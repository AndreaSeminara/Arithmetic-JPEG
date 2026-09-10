import numpy as np

DIM_BLOCK = 8


def get_nxn_blocks(
    channel_data: np.ndarray, block_size: int = DIM_BLOCK
) -> tuple[list[np.ndarray], int, int]:
    """Taglia il canale in blocchi block_size x block_size.
    Ritorna la lista di blocchi, il padding aggiunto in altezza e quello in larghezza."""
    if channel_data.ndim != 2:
        raise ValueError("Il canale dell'immagine deve essere una matrice 2D.")

    padded_channel = padding(channel_data, block_size)

    orig_height, orig_width = channel_data.shape
    height, width = padded_channel.shape
    blocks = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = padded_channel[i : i + block_size, j : j + block_size]
            blocks.append(block)

    return blocks, height - orig_height, width - orig_width


def reassemble_blocks(
    blocks: list[np.ndarray], image_shape: tuple[int, int], pad_h: int, pad_w: int
) -> np.ndarray:
    h, w = image_shape
    padded_h, padded_w = h + pad_h, w + pad_w

    reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)

    idx = 0
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            reconstructed[i : i + 8, j : j + 8] = blocks[idx]
            idx += 1

    return reconstructed[:h, :w]


def padding(channel_data: np.ndarray, block_size: int = DIM_BLOCK) -> np.ndarray:
    height, width = channel_data.shape
    new_height = (height + block_size - 1) // block_size * block_size
    new_width = (width + block_size - 1) // block_size * block_size

    pad_h = new_height - height
    pad_w = new_width - width

    if pad_h == 0 and pad_w == 0:
        return channel_data

    # mode="edge" replica l'ultimo pixel invece di mettere zeri, così la DCT
    # non vede un bordo artificiale che creerebbe artefatti nella compressione.
    padded_channel = np.pad(channel_data, ((0, pad_h), (0, pad_w)), mode="edge")

    return padded_channel
