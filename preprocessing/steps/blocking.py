import numpy as np

DIM_BLOCK = 8


def get_nxn_blocks(
    channel_data: np.ndarray, block_size: int = DIM_BLOCK
) -> tuple[list[np.ndarray], int, int]:
    """Divide un canale in blocchi nxn"""
    if channel_data.ndim != 2:
        raise ValueError("Il canale dell'immagine deve essere una matrice 2D.")

    # Applica il padding per garantire che le dimensioni siano multipli di block_size
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
    """Ricostruisce l'immagine e rimuove il padding"""
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
    """Aggiunge padding per ottenere dimensioni multiple di block_size"""
    height, width = channel_data.shape
    new_height = (height + block_size - 1) // block_size * block_size
    new_width = (width + block_size - 1) // block_size * block_size

    pad_h = new_height - height
    pad_w = new_width - width

    if pad_h == 0 and pad_w == 0:
        return channel_data

    padded_channel = np.pad(channel_data, ((0, pad_h), (0, pad_w)), mode="edge")

    return padded_channel
