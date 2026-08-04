import numpy as np
from .coders.huffman import HuffmanEncoder

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


def _encode_single_method(
    blocks_by_channel: dict[str, list[np.ndarray]], method: str
) -> bytes:
    """Usa l'encoder scelto per codificare i blocchi e restituire il flusso compresso"""
    if method == "huffman":
        encoder = HuffmanEncoder()
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
