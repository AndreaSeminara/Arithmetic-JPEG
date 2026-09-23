import numpy as np

from .coders.arithmetic import ArithmeticStandard, ArithmeticStatic
from .coders.huffman import Huffman
from .coders.qm import QMCoder


def encode_blocks(
    blocks_by_channel: dict[str, list[np.ndarray]], method: str = "all"
) -> bytes | dict[str, bytes]:
    """Restituisce il flusso compresso dei blocchi per canale"""
    if method == "all":
        streams, tables = {}, {}
        for m in ["huffman", "arithmetic_tables", "arithmetic_static", "qm"]:
            s, t = _encode_single_method(blocks_by_channel, m)
            streams[m] = s
            if t is not None:
                tables[m] = t
        return streams, tables
    else:
        return _encode_single_method(blocks_by_channel, method)


def decode_blocks(
    compressed_stream: bytes,
    blocks_layout: dict[str, int],
    method: str = "huffman",
    custom_tables: dict = None,
) -> dict[str, list[np.ndarray]]:
    """Restituisce i blocchi decodificati per canale"""
    if method == "huffman":
        decoder = Huffman()
    elif method == "arithmetic_tables":
        decoder = ArithmeticStandard()
    elif method == "arithmetic_static":
        decoder = ArithmeticStatic()
    elif method == "qm":
        decoder = QMCoder()
    else:
        raise ValueError(f"Metodo di decodifica '{method}' non supportato.")

    decoded_blocks_by_channel = {}
    current_stream = compressed_stream

    for channel, num_blocks in blocks_layout.items():
        is_luma = channel == "Y"

        if method == "arithmetic_static":
            # Passiamo al decoder solo le tabelle del canale corrente (Y, Cb o Cr)
            ch_tables = custom_tables[channel] if custom_tables else None
            blocks, bytes_consumed = decoder.decode(
                current_stream,
                num_blocks=num_blocks,
                is_luma=is_luma,
                custom_tables=ch_tables,
            )
        else:
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
    elif method == "arithmetic_tables":
        encoder = ArithmeticStandard()
    elif method == "arithmetic_static":
        encoder = ArithmeticStatic()
    elif method == "qm":
        encoder = QMCoder()
    else:
        raise ValueError(f"Metodo di codifica non supportato: {method}")

    full_compressed_stream = b""
    custom_tables = {} if method == "arithmetic_static" else None

    for channel_name, blocks in blocks_by_channel.items():
        is_luma = channel_name == "Y"

        if method == "arithmetic_static":
            stream, tables = encoder.encode(blocks, is_luma=is_luma)
            full_compressed_stream += stream
            custom_tables[channel_name] = tables  # Salviamo le tabelle di Y, Cb e Cr
        else:
            full_compressed_stream += encoder.encode(blocks, is_luma=is_luma)

    return full_compressed_stream, custom_tables
