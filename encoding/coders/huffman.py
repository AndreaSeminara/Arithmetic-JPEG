import numpy as np
from utils.tables import (
    STD_DC_LUMA_BITS,
    STD_DC_LUMA_VALS,
    STD_AC_LUMA_BITS,
    STD_AC_LUMA_VALS,
    STD_DC_CHROMA_BITS,
    STD_DC_CHROMA_VALS,
    STD_AC_CHROMA_BITS,
    STD_AC_CHROMA_VALS,
)
from .base import EntropyEncoder, EntropyDecoder


def build_huffman_dict(bits, huffval):
    # Costruisce il dizionario Huffman a partire dalle tabelle T.81.
    # BITS: quanti codici ci sono per ogni lunghezza (da 1 a 16 bit)
    # HUFFVAL: i simboli reali ordinati per probabilità
    if sum(bits) != len(huffval):
        raise ValueError(
            "Tabelle inconsistenti: il totale di BITS non corrisponde al numero di simboli in HUFFVAL."
        )

    huff_dict = {}
    code = 0
    idx = 0

    # Generazione canonica dei codici Huffman:
    # Si parte da 0 e si shifta a sinistra (aggiungendo un bit) ad ogni step di lunghezza
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            huff_dict[huffval[idx]] = bin(code)[2:].zfill(length)
            code += 1
            idx += 1
        code <<= 1  # Shift a sinistra per allungare il codice al prossimo ciclo

    return huff_dict


class Huffman(EntropyEncoder, EntropyDecoder):
    # Implementazione classica JPEG con tabelle fisse.

    def __init__(self):
        self.bit_str = ""

        # Tabelle per codificare (Simbolo -> Bits)
        self.dc_luma = build_huffman_dict(STD_DC_LUMA_BITS, STD_DC_LUMA_VALS)
        self.ac_luma = build_huffman_dict(STD_AC_LUMA_BITS, STD_AC_LUMA_VALS)
        self.dc_chroma = build_huffman_dict(STD_DC_CHROMA_BITS, STD_DC_CHROMA_VALS)
        self.ac_chroma = build_huffman_dict(STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS)

        # Tabelle per decodificare (Bits -> Simbolo) invertendo le chiavi
        self.dc_luma_dec = {v: k for k, v in self.dc_luma.items()}
        self.ac_luma_dec = {v: k for k, v in self.ac_luma.items()}
        self.dc_chroma_dec = {v: k for k, v in self.dc_chroma.items()}
        self.ac_chroma_dec = {v: k for k, v in self.ac_chroma.items()}

    def encode(self, blocks, is_luma=True):
        self.bit_str = ""
        prev_dc = 0

        dc_table = self.dc_luma if is_luma else self.dc_chroma
        ac_table = self.ac_luma if is_luma else self.ac_chroma

        for block in blocks:
            #  DC: si salva solo la differenza col blocco precedente (DPCM)
            dc_val = int(block[0])
            diff = dc_val - prev_dc
            prev_dc = dc_val

            dc_size, dc_bits = self._get_cat_and_bits(diff)
            self.bit_str += dc_table[dc_size]  # Prefisso Huffman
            self.bit_str += dc_bits  # Bit effettivi del valore

            #  AC: Run-Length Encoding (zeri consecutivi)
            run = 0
            for ac_val in block[1:]:
                ac_val = int(ac_val)

                if ac_val == 0:
                    run += 1
                    if run == 16:
                        # ZRL: 16 zeri consecutivi, il contatore viene azzerato
                        self.bit_str += ac_table[0xF0]
                        run = 0
                else:
                    ac_size, ac_bits = self._get_cat_and_bits(ac_val)
                    ac_key = (run << 4) | ac_size  # Uniamo RUN e SIZE in un solo byte

                    self.bit_str += ac_table[ac_key]
                    self.bit_str += ac_bits
                    run = 0

            if run > 0:
                # EOB: Fine del blocco, il resto è tutto zero
                self.bit_str += ac_table[0x00]

        return self._pack_bytes()

    def _get_cat_and_bits(self, val):
        # Data un'ampiezza, trova in che categoria "Size" ricade e genera la stringa binaria
        if val == 0:
            return 0, ""

        abs_val = abs(val)
        size = abs_val.bit_length()

        if val > 0:
            bits = bin(val)[2:]
        else:
            # Numeri negativi: complemento a 1
            comp = (1 << size) + val - 1
            bits = bin(comp)[2:].zfill(size)

        return size, bits

    def _pack_bytes(self):
        # Converte la sequenza continua di bit ("0" e "1") in byte effettivi.
        # Nello standard JPEG, il padding finale si fa con "1" (bit a 1).
        rem = len(self.bit_str) % 8
        if rem != 0:
            self.bit_str += "1" * (8 - rem)

        b_array = bytearray()
        for i in range(0, len(self.bit_str), 8):
            b_array.append(int(self.bit_str[i : i + 8], 2))

        return bytes(b_array)

    def _decode_val(self, size, bits):
        if size == 0:
            return 0
        if bits[0] == "1":
            return int(bits, 2)
        return int(bits, 2) - (1 << size) + 1

    def decode(self, byte_stream, num_blocks, is_luma=True, custom_tables=None):
        # Nota: custom_tables è qui per rispettare l'interfaccia Base, ma Huffman usa le fisse.
        bit_str = "".join(f"{b:08b}" for b in byte_stream)
        idx = 0

        dc_table = self.dc_luma_dec if is_luma else self.dc_chroma_dec
        ac_table = self.ac_luma_dec if is_luma else self.ac_chroma_dec

        blocks = []
        prev_dc = 0

        for _ in range(num_blocks):
            block = np.zeros(64, dtype=np.float32)

            #  Lettura DC
            code = ""
            while True:
                # Leggiamo un bit alla volta finché non troviamo una corrispondenza nel dizionario Huffman
                code += bit_str[idx]
                idx += 1
                if code in dc_table:
                    dc_size = dc_table[code]
                    break

            if dc_size > 0:
                dc_bits = bit_str[idx : idx + dc_size]
                idx += dc_size
                dc_diff = self._decode_val(dc_size, dc_bits)
            else:
                dc_diff = 0

            prev_dc += dc_diff
            block[0] = prev_dc

            #  Lettura AC
            ac_idx = 1
            while ac_idx < 64:
                code = ""
                while True:
                    code += bit_str[idx]
                    idx += 1
                    if code in ac_table:
                        ac_val = ac_table[code]
                        break

                if ac_val == 0x00:
                    break  # EOB: Fine del blocco
                elif ac_val == 0xF0:
                    ac_idx += 16  # ZRL: Salto di 16 zeri
                else:
                    run = ac_val >> 4
                    size = ac_val & 0x0F
                    ac_idx += run

                    if size > 0:
                        ac_bits = bit_str[idx : idx + size]
                        idx += size
                        block[ac_idx] = self._decode_val(size, ac_bits)
                    ac_idx += 1

            blocks.append(block)

        # Calcola i byte consumati per indicare al chiamante dove riprendere
        bytes_consumed = (idx + 7) // 8
        return blocks, bytes_consumed
