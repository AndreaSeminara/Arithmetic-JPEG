import numpy as np

ZIGZAG_INDICES = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
])

def zigzag_scan(block):
    """
    Esegue lo Zig-Zag scan su un blocco 8x8.

    Args:
        block (np.ndarray): Blocco 8x8 da scansionare.
    Returns:
        np.ndarray: Array 1D contenente gli elementi del blocco in ordine Zig-Zag.
    """
    if block.shape != (8, 8):
        raise ValueError("Il blocco deve essere di dimensione 8x8.")

    zigzag_array = block.ravel()[ZIGZAG_INDICES]

    return zigzag_array