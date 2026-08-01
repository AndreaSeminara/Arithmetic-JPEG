import numpy as np

DIM_BLOCK = 8


def get_nxn_blocks(channel_data, block_size=DIM_BLOCK):
    """
    Divide un canale di immagine in blocchi nxn.

    Args:
        channel_data (np.ndarray): Canale dell'immagine da dividere in blocchi.
        block_size (int): Dimensione dei blocchi (default: 8).

    Returns:
        list: Lista di blocchi nxn.
    """
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


def padding(channel_data, block_size=DIM_BLOCK):
    """
    Applica il padding a un canale di immagine per garantire che le dimensioni siano multipli di block_size.

    Args:
        channel_data (np.ndarray): Canale dell'immagine a cui applicare il padding.
        block_size (int): Dimensione dei blocchi (default: 8).

    Returns:
        np.ndarray: Canale dell'immagine con padding applicato.
    """
    height, width = channel_data.shape
    new_height = (height + block_size - 1) // block_size * block_size
    new_width = (width + block_size - 1) // block_size * block_size

    pad_h = new_height - height
    pad_w = new_width - width

    if pad_h == 0 and pad_w == 0:
        return channel_data

    padded_channel = np.pad(channel_data, ((0, pad_h), (0, pad_w)), mode="edge")

    return padded_channel
