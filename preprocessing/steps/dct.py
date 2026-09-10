import numpy as np


def _build_1d_dct_matrix(N: int = 8) -> np.ndarray:
    """Precalcola la matrice C della DCT-II ortogonale (T.81 §A.3.3).
    Sfrutta la separabilità 2D: DCT(block) = C @ block @ C^T"""
    C = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        for x in range(N):
            alpha = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
            C[u, x] = alpha * np.cos((2 * x + 1) * u * np.pi / (2 * N))
    return C


DCT_MTX = _build_1d_dct_matrix(8)


def dct(block: np.ndarray) -> np.ndarray:
    if block.shape != (8, 8):
        raise ValueError("Il blocco deve essere di dimensione 8x8.")

    # I pixel stanno in [0, 255]: li centrare attorno allo zero migliora la DCT
    # e riduce l'ampiezza del coefficiente DC, che altrimenti dominerebbe tutto.
    block = block.astype(np.float32) - 128.0

    return np.dot(np.dot(DCT_MTX, block), DCT_MTX.T)


def inv_dct(block: np.ndarray) -> np.ndarray:
    if block.shape != (8, 8):
        raise ValueError("Il blocco deve essere di dimensione 8x8.")

    # C^T @ block @ C inverte esattamente la trasformata,
    # poi rimettiamo il bias di 128 che avevamo tolto in fase di codifica.
    return np.dot(np.dot(DCT_MTX.T, block), DCT_MTX) + 128.0
