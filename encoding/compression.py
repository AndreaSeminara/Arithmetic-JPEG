import numpy as np
import pickle
import struct
from typing import Any

from .coders.arithmetic import ArithmeticStandard, ArithmeticStatic
from .coders.huffman import Huffman
from .coders.qm import QMCoder

MAGIC_NUMBER = b"AS"
HEADER_FORMAT = ">2sHHBI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def save_custom_jpeg(
    filepath: str,
    width: int,
    height: int,
    algo_flag: int,
    bitstream: bytes,
    custom_tables: Any = None,
) -> None:
    """Salva il bitstream in un file binario proprietario con header.

    Header (ordine):
    - Magic Number (2B)
    - Width (2B)
    - Height (2B)
    - Algo Flag (1B)
    - Lunghezza payload tabelle (4B)
    - Payload tabelle (variabile, solo per flag == 2)
    - Bitstream (variabile)
    """
    if not (0 <= width <= 0xFFFF and 0 <= height <= 0xFFFF):
        raise ValueError("Width e Height devono essere nel range [0, 65535].")

    if algo_flag not in (0, 1, 2, 3):
        raise ValueError("algo_flag non valido. Valori ammessi: 0, 1, 2, 3.")

    if not isinstance(bitstream, (bytes, bytearray)):
        raise TypeError("bitstream deve essere di tipo bytes o bytearray.")

    bitstream_bytes = bytes(bitstream)
    tables_payload = b""
    if algo_flag == 2:
        extracted_tables = None

        if custom_tables is not None:
            extracted_tables = custom_tables

        if extracted_tables is None:
            raise ValueError(
                "Per algo_flag == 2 (Aritmetica Statica) custom_tables non puo essere None."
            )
        tables_payload = pickle.dumps(
            extracted_tables, protocol=pickle.HIGHEST_PROTOCOL
        )

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_NUMBER,
        width,
        height,
        algo_flag,
        len(tables_payload),
    )

    try:
        with open(filepath, "wb") as file_obj:
            file_obj.write(header)
            if tables_payload:
                file_obj.write(tables_payload)
            file_obj.write(bitstream_bytes)
    except OSError as exc:
        raise OSError(
            f"Errore durante il salvataggio del file '{filepath}': {exc}"
        ) from exc


def load_custom_jpeg(filepath: str) -> tuple[int, int, int, Any, bytes]:
    """Carica un file binario proprietario e restituisce metadati e bitstream.

    Ritorna:
    (width, height, algo_flag, custom_tables, bitstream)
    """

    def _read_exact(file_obj, size: int) -> bytes:
        data = file_obj.read(size)
        if len(data) != size:
            raise ValueError("File corrotto o incompleto: header/payload troncato.")
        return data

    try:
        with open(filepath, "rb") as file_obj:
            header_data = _read_exact(file_obj, HEADER_SIZE)
            magic, width, height, algo_flag, tables_len = struct.unpack(
                HEADER_FORMAT, header_data
            )

            if magic != MAGIC_NUMBER:
                raise ValueError(
                    "Magic Number non valido: il file non e in formato custom JPEG."
                )

            if algo_flag not in (0, 1, 2, 3):
                raise ValueError(f"Algo Flag non valido nel file: {algo_flag}.")

            custom_tables = None
            if algo_flag == 2:
                if tables_len == 0:
                    raise ValueError(
                        "Header non valido: tabelle mancanti per Aritmetica Statica (flag == 2)."
                    )
                tables_payload = _read_exact(file_obj, tables_len)
                try:
                    custom_tables = pickle.loads(tables_payload)
                except Exception as exc:
                    raise ValueError(
                        "Impossibile deserializzare le custom_tables dal payload."
                    ) from exc
            elif tables_len != 0:
                raise ValueError(
                    "Header non valido: tables_len deve essere 0 se algo_flag != 2."
                )

            bitstream = file_obj.read()
            return width, height, algo_flag, custom_tables, bitstream

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File non trovato: '{filepath}'.") from exc
    except OSError as exc:
        raise OSError(
            f"Errore durante la lettura del file '{filepath}': {exc}"
        ) from exc


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
