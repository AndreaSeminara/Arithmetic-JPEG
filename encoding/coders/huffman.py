import numpy as np
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
from .base import EntropyEncoder, EntropyDecoder


def build_huffman_dict(bits: list[int], huffval: list[int]) -> dict[int, str]:
    """Costruisce una tabella Huffman standard"""
    if sum(bits) != len(huffval):
        raise ValueError(
            f"Incoerenza Tabelle Huffman: BITS dichiara {sum(bits)} codici, "
            f"ma HUFFVAL contiene {len(huffval)} valori."
        )

    huff_dict = {}
    code = 0
    idx = 0

    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            huff_dict[huffval[idx]] = bin(code)[2:].zfill(length)
            code += 1
            idx += 1
        code <<= 1

    return huff_dict


class HuffmanEncoder(EntropyEncoder):
    def __init__(self) -> None:
        """Inizializza le tabelle Huffman"""
        self.bit_string = ""

        self.dc_luma_table = build_huffman_dict(STD_DC_LUMA_BITS, STD_DC_LUMA_VALS)
        self.ac_luma_table = build_huffman_dict(STD_AC_LUMA_BITS, STD_AC_LUMA_VALS)
        self.dc_chroma_table = build_huffman_dict(
            STD_DC_CHROMA_BITS, STD_DC_CHROMA_VALS
        )
        self.ac_chroma_table = build_huffman_dict(
            STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS
        )

    def encode(self, blocks: list[np.ndarray], is_luma: bool = True) -> bytes:
        """Codifica i blocchi con Huffman"""
        self.bit_string = ""
        prev_dc = 0

        dc_table = self.dc_luma_table if is_luma else self.dc_chroma_table
        ac_table = self.ac_luma_table if is_luma else self.ac_chroma_table

        for block in blocks:
            dc_value = int(block[0])
            diff = dc_value - prev_dc
            prev_dc = dc_value

            dc_size, dc_bits = self._get_category_and_bits(diff)

            self.bit_string += dc_table[dc_size]
            self.bit_string += dc_bits

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

    def _get_category_and_bits(self, value: int) -> tuple[int, str]:
        """
        Implementa le Tabelle dello standard T.81
        Restituisce la Size, cioè quanti bit servono, e i bit del coefficiente
        """
        if value == 0:
            return 0, ""

        abs_val = abs(value)
        size = abs_val.bit_length()

        if value > 0:
            bits = bin(value)[2:]
        else:
            positive_complement = (1 << size) + value - 1
            bits = bin(positive_complement)[2:].zfill(size)

        return size, bits

    def _pack_bits_to_bytes(self) -> bytes:
        """Converte la stringa di bit in byte"""
        remainder = len(self.bit_string) % 8
        if remainder != 0:
            self.bit_string += "1" * (8 - remainder)

        byte_array = bytearray()
        for i in range(0, len(self.bit_string), 8):
            byte_chunk = self.bit_string[i : i + 8]
            byte_array.append(int(byte_chunk, 2))

        return bytes(byte_array)


class HuffmanDecoder(EntropyDecoder):
    def __init__(self) -> None:
        """Inizializza le tabelle Huffman inverse"""
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

        self.dc_luma_table = {
            v: k
            for k, v in build_huffman_dict(STD_DC_LUMA_BITS, STD_DC_LUMA_VALS).items()
        }
        self.ac_luma_table = {
            v: k
            for k, v in build_huffman_dict(STD_AC_LUMA_BITS, STD_AC_LUMA_VALS).items()
        }
        self.dc_chroma_table = {
            v: k
            for k, v in build_huffman_dict(
                STD_DC_CHROMA_BITS, STD_DC_CHROMA_VALS
            ).items()
        }
        self.ac_chroma_table = {
            v: k
            for k, v in build_huffman_dict(
                STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS
            ).items()
        }

    def _decode_value(self, size: int, bits: str) -> int:
        """Restituisce il valore decodificato"""
        if size == 0:
            return 0
        if bits[0] == "1":
            return int(bits, 2)
        else:
            return int(bits, 2) - (1 << size) + 1

    def decode(
        self, byte_stream: bytes, num_blocks: int, is_luma: bool = True
    ) -> tuple[list[np.ndarray], int]:
        """Restituisce i blocchi decodificati e i byte consumati"""
        bit_string = "".join(f"{byte:08b}" for byte in byte_stream)
        bit_idx = 0

        dc_table = self.dc_luma_table if is_luma else self.dc_chroma_table
        ac_table = self.ac_luma_table if is_luma else self.ac_chroma_table

        blocks = []
        prev_dc = 0

        for _ in range(num_blocks):
            block = np.zeros(64, dtype=np.float32)

            # Lettura DC
            code = ""
            while True:
                code += bit_string[bit_idx]
                bit_idx += 1
                if code in dc_table:
                    dc_size = dc_table[code]
                    break

            if dc_size > 0:
                dc_bits = bit_string[bit_idx : bit_idx + dc_size]
                bit_idx += dc_size
                dc_diff = self._decode_value(dc_size, dc_bits)
            else:
                dc_diff = 0

            prev_dc += dc_diff
            block[0] = prev_dc

            # Lettura AC
            ac_idx = 1
            while ac_idx < 64:
                code = ""
                while True:
                    code += bit_string[bit_idx]
                    bit_idx += 1
                    if code in ac_table:
                        ac_val = ac_table[code]
                        break

                if ac_val == 0x00:
                    break
                elif ac_val == 0xF0:
                    ac_idx += 16
                else:
                    run = ac_val >> 4
                    size = ac_val & 0x0F
                    ac_idx += run

                    if size > 0:
                        ac_bits = bit_string[bit_idx : bit_idx + size]
                        bit_idx += size
                        block[ac_idx] = self._decode_value(size, ac_bits)
                    ac_idx += 1

            blocks.append(block)

        bytes_consumed = (bit_idx + 7) // 8
        return blocks, bytes_consumed
