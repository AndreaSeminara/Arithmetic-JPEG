import pickle
import struct
from typing import Any

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
