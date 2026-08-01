from .coders.huffman import HuffmanEncoder

# from .coders.arithmetic import ArithmeticEncoder
# from .coders.qm import QMCoder


def encode_blocks(blocks_by_channel: dict, method: str = "all"):
    """
    Codifica i blocchi di dati separati per canale.

    Args:
        blocks_by_channel (dict): Dizionario {nome_canale: lista_blocchi}
                                  es. {'Y': [...], 'Cb': [...], 'Cr': [...]}
        method (str): Metodo di compressione ("huffman", "arithmetic", "qm" o "all").

    Returns:
        bytes o dict: Flusso di dati compresso o dizionario di flussi.
    """
    if method == "all":
        return {
            "huffman": _encode_single_method(blocks_by_channel, "huffman"),
            # "arithmetic": _encode_single_method(blocks_by_channel, "arithmetic"),
            # "qm": _encode_single_method(blocks_by_channel, "qm"),
        }
    else:
        return _encode_single_method(blocks_by_channel, method)


def _encode_single_method(blocks_by_channel: dict, method: str) -> bytes:
    """
    Inizializza l'encoder corretto e processa i canali in sequenza.
    """
    if method == "huffman":
        encoder = HuffmanEncoder()
    # elif method == "arithmetic":
    #     encoder = ArithmeticEncoder()
    # elif method == "qm":
    #     encoder = QMCoder()
    else:
        raise ValueError(f"Metodo di codifica non supportato: {method}")

    full_compressed_stream = b""

    # Itero su ciascun canale
    for channel_name, blocks in blocks_by_channel.items():
        is_luma = channel_name == "Y"

        full_compressed_stream += encoder.encode(blocks, is_luma=is_luma)

    return full_compressed_stream
