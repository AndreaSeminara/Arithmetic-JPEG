import numpy as np
from .coders.huffman import Huffman

# from .coders.arithmetic import ArithmeticEncoder
# from .coders.qm import QMCoder


def encode_blocks(
    blocks_by_channel: dict[str, list[np.ndarray]], method: str = "all"
) -> bytes | dict[str, bytes]:
    """Restituisce il flusso compresso dei blocchi per canale"""
    if method == "all":
        return {
            "huffman": _encode_single_method(blocks_by_channel, "huffman"),
            # "arithmetic": _encode_single_method(blocks_by_channel, "arithmetic"),
            # "qm": _encode_single_method(blocks_by_channel, "qm"),
        }
    else:
        return _encode_single_method(blocks_by_channel, method)


def decode_blocks(
    compressed_stream: bytes, blocks_layout: dict[str, int], method: str = "huffman"
) -> dict[str, list[np.ndarray]]:
    """Restituisce i blocchi decodificati per canale"""
    if method == "huffman":
        decoder = Huffman()
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


def _encode_single_method(
    blocks_by_channel: dict[str, list[np.ndarray]], method: str
) -> bytes:
    """Usa l'encoder scelto per codificare i blocchi e restituire il flusso compresso"""
    if method == "huffman":
        encoder = Huffman()
    # elif method == "arithmetic":
    #     encoder = ArithmeticEncoder()
    # elif method == "qm":
    #     encoder = QMCoder()
    else:
        raise ValueError(f"Metodo di codifica non supportato: {method}")

    full_compressed_stream = b""

    for channel_name, blocks in blocks_by_channel.items():
        is_luma = channel_name == "Y"

        full_compressed_stream += encoder.encode(blocks, is_luma=is_luma)

    return full_compressed_stream
