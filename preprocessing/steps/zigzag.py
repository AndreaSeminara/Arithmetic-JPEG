import numpy as np
from utils import ZIGZAG_INDEX


def zigzag_scan(block_2d: np.ndarray) -> np.ndarray:
    # L'ordine zig-zag mette le basse frequenze per prime
    # e lascia quelle alte in fondo, dove di solito ci sono lunghe serie di zeri:
    # perfetto per il run-length encoding che viene dopo.
    block_flat = block_2d.flatten()
    block_1d = np.zeros(64, dtype=np.float32)

    for i in range(64):
        block_1d[i] = block_flat[ZIGZAG_INDEX[i]]

    return block_1d


def inverse_zigzag_scan(block_1d: np.ndarray) -> np.ndarray:
    block_flat = np.zeros(64, dtype=np.float32)

    for i in range(64):
        block_flat[ZIGZAG_INDEX[i]] = block_1d[i]

    return block_flat.reshape((8, 8))
