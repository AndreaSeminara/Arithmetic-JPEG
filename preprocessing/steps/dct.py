import numpy as np


def _build_1d_dct_matrix(N: int = 8) -> np.ndarray:
    """Costruisce la matrice DCT 1D"""
    C = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        for x in range(N):
            alpha = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
            C[u, x] = alpha * np.cos((2 * x + 1) * u * np.pi / (2 * N))
    return C


DCT_MTX = _build_1d_dct_matrix(8)


def dct(block: np.ndarray) -> np.ndarray:
    """Applica la DCT a un blocco 8x8"""
    if block.shape != (8, 8):
        raise ValueError("Il blocco deve essere di dimensione 8x8.")

    # Sottraggo 128 per centrare i valori intorno a zero
    block = block.astype(np.float32) - 128.0

    # Applico la DCT: DCT = C * block * C^T
    dct_block = np.dot(np.dot(DCT_MTX, block), DCT_MTX.T)

    return dct_block


def inv_dct(block: np.ndarray) -> np.ndarray:
    """Applica l'Inverse DCT a un blocco 8x8"""
    if block.shape != (8, 8):
        raise ValueError("Il blocco deve essere di dimensione 8x8.")

    # Applico la DCT Inversa: IDCT = C^T * block * C
    idct_block = np.dot(np.dot(DCT_MTX.T, block), DCT_MTX)

    # Aggiungo 128 per riportare i valori nel range originale [0, 255]
    reconstructed_block = idct_block + 128.0

    return reconstructed_block
