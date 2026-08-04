import numpy as np
from utils import ZIGZAG_INDEX


def zigzag_scan(block_2d: np.ndarray) -> np.ndarray:
    """Appiattisce un blocco 8x8 in un array tramite una scansione Zig-Zag"""
    block_flat = block_2d.flatten()
    block_1d = np.zeros(64, dtype=np.float32)

    for i in range(64):
        block_1d[i] = block_flat[ZIGZAG_INDEX[i]]

    return block_1d


def inverse_zigzag_scan(block_1d: np.ndarray) -> np.ndarray:
    """Ricostruisce un blocco 8x8 partendo da un array che segue l'ordine Zig-Zag"""
    block_flat = np.zeros(64, dtype=np.float32)

    for i in range(64):
        block_flat[ZIGZAG_INDEX[i]] = block_1d[i]

    return block_flat.reshape((8, 8))
