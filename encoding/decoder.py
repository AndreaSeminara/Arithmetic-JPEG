import numpy as np
from .coders.huffman import HuffmanDecoder


def decode_blocks(
    compressed_stream: bytes, blocks_layout: dict[str, int], method: str = "huffman"
) -> dict[str, list[np.ndarray]]:
    """Restituisce i blocchi decodificati per canale"""
    if method == "huffman":
        decoder = HuffmanDecoder()
    # elif method == "arithmetic":
    #     decoder = ArithmeticDecoder()
    # elif method == "qm":
    #     decoder = QMDecoder()
    else:
        raise ValueError(f"Metodo di decodifica '{method}' non supportato.")

    decoded_blocks_by_channel = {}
    current_stream = compressed_stream

    for channel, num_blocks in blocks_layout.items():
        is_luma = channel == "Y"

        blocks, bytes_consumed = decoder.decode(
            current_stream, num_blocks=num_blocks, is_luma=is_luma
        )
        decoded_blocks_by_channel[channel] = blocks

        current_stream = current_stream[bytes_consumed:]

    return decoded_blocks_by_channel
