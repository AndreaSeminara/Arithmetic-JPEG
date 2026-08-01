import numpy as np


def _build_1d_dct_matrix(N: int = 8) -> np.ndarray:
    """
    Costruisce la matrice di trasformazione DCT 1D di dimensione NxN
    """
    C = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        for x in range(N):
            # Fattore di normalizzazione alpha
            alpha = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
            # Formula standard DCT-II
            C[u, x] = alpha * np.cos((2 * x + 1) * u * np.pi / (2 * N))
    return C


DCT_MTX = _build_1d_dct_matrix(8)


def dct(block):
    """
    Applica la Trasformata Discreta del Coseno (DCT) a un blocco 8x8.

    Args:
        block (np.ndarray): Blocco 8x8 di dati dell'immagine.
    Returns:
        np.ndarray: Blocco 8x8 trasformato con DCT.
    """
    if block.shape != (8, 8):
        raise ValueError("Il blocco deve essere di dimensione 8x8.")

    # Sottraggo 128 per centrare i valori intorno a zero
    block = block.astype(np.float32) - 128.0

    # Applico la DCT: DCT = C * block * C^T
    dct_block = np.dot(np.dot(DCT_MTX, block), DCT_MTX.T)

    return dct_block
