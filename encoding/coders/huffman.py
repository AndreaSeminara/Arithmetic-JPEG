import numpy as np
from typing import List, Tuple
from utils.tables import (
    STD_DC_LUMA_BITS,
    STD_DC_LUMA_VALS,
    STD_AC_LUMA_BITS,
    STD_AC_LUMA_VALS,
)
from utils.tables import (
    STD_DC_CHROMA_BITS,
    STD_DC_CHROMA_VALS,
    STD_AC_CHROMA_BITS,
    STD_AC_CHROMA_VALS,
)
from .base import EntropyEncoder


def build_huffman_dict(bits: List[int], huffval: List[int]):
    """
    Algoritmo Standard T.81
    Restituisce un dizionario { valore: "codice_binario_stringa" }.
    """
    # La somma delle lunghezze deve uguagliare il numero di foglie
    if sum(bits) != len(huffval):
        raise ValueError(
            f"Incoerenza Tabelle Huffman: BITS dichiara {sum(bits)} codici, "
            f"ma HUFFVAL contiene {len(huffval)} valori."
        )

    huff_dict = {}
    code = 0
    idx = 0

    # Scorre le lunghezze da 1 a 16 bit
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            huff_dict[huffval[idx]] = bin(code)[2:].zfill(length)
            code += 1
            idx += 1
        code <<= 1

    return huff_dict


class HuffmanEncoder(EntropyEncoder):
    def __init__(self):
        self.bit_string = ""

        # Pre-generazione dei 4 dizionari di codifica
        self.dc_luma_table = build_huffman_dict(STD_DC_LUMA_BITS, STD_DC_LUMA_VALS)
        self.ac_luma_table = build_huffman_dict(STD_AC_LUMA_BITS, STD_AC_LUMA_VALS)
        self.dc_chroma_table = build_huffman_dict(
            STD_DC_CHROMA_BITS, STD_DC_CHROMA_VALS
        )
        self.ac_chroma_table = build_huffman_dict(
            STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS
        )

    def encode(self, blocks: List[np.ndarray], is_luma: bool = True):
        self.bit_string = ""
        prev_dc = 0

        # Selezione delle tabelle in base al tipo di canale
        dc_table = self.dc_luma_table if is_luma else self.dc_chroma_table
        ac_table = self.ac_luma_table if is_luma else self.ac_chroma_table

        for block in blocks:
            # --- CODIFICA DC ---
            dc_value = int(block[0])
            diff = dc_value - prev_dc
            prev_dc = dc_value

            dc_size, dc_bits = self._get_category_and_bits(diff)

            # Usa la tabella corretta
            self.bit_string += dc_table[dc_size]
            self.bit_string += dc_bits

            # --- CODIFICA AC ---
            run_length = 0
            for ac_value in block[1:]:
                ac_value = int(ac_value)

                if ac_value == 0:
                    run_length += 1
                    if run_length == 16:
                        self.bit_string += ac_table[0xF0]
                        run_length = 0
                else:
                    ac_size, ac_bits = self._get_category_and_bits(ac_value)
                    ac_key = (run_length << 4) | ac_size

                    self.bit_string += ac_table[ac_key]
                    self.bit_string += ac_bits
                    run_length = 0

            if run_length > 0:
                self.bit_string += ac_table[0x00]

        return self._pack_bits_to_bytes()

    def _get_category_and_bits(self, value: int) -> Tuple[int, str]:
        """
        Implementa le Tabelle dello standard T.81.
        Restituisce la Size (quanti bit servono) e la rappresentazione
        in bit (il valore vero e proprio) del coefficiente.
        """
        if value == 0:
            return 0, ""

        # La 'size' (categoria) è calcolata tramite la lunghezza in bit del valore assoluto
        abs_val = abs(value)
        size = abs_val.bit_length()

        if value > 0:
            # Per valori positivi, i bit sono la rappresentazione binaria standard
            bits = bin(value)[2:]
        else:
            # Valori negativi: complemento a 1 (formula T.81)
            positive_complement = (1 << size) + value - 1
            bits = bin(positive_complement)[2:].zfill(size)

        return size, bits

    def _pack_bits_to_bytes(self) -> bytes:
        """
        Converte la stringa di bit in un oggetto 'bytes'.
        Gestisce il padding finale con '1' come previsto dallo standard T.81.
        """
        remainder = len(self.bit_string) % 8
        if remainder != 0:
            self.bit_string += "1" * (8 - remainder)

        byte_array = bytearray()
        for i in range(0, len(self.bit_string), 8):
            byte_chunk = self.bit_string[i : i + 8]
            byte_array.append(int(byte_chunk, 2))

        return bytes(byte_array)
