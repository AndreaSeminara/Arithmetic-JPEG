import numpy as np
from PIL import Image
from tqdm import tqdm

from .steps import (
    rgb_to_ycbcr,
    get_nxn_blocks,
    dct,
    quantize_block,
    zigzag_scan,
)
from .steps import (
    ycbcr_to_rgb,
    reassemble_blocks,
    inv_dct,
    dequantize_block,
    inverse_zigzag_scan,
)
from encoding import encode_blocks, decode_blocks


def run_pipeline(
    img: Image.Image, grayscale: bool = False, method: str = "all"
) -> tuple[bytes | dict[str, bytes], dict[str, Image.Image]]:
    """Esegue la pipeline completa di compressione e ricostruzione"""
    print("Esecuzione della pipeline di preprocessing...\n")

    if img is None:
        print("Errore: Immagine non valida. Assicurati di fornire un'immagine valida.")
        return None

    # --- Preprocessing ---

    if grayscale or img.mode == "L":
        print("Immagine in scala di grigi...")
        img_gray = img.convert("L")
        channels = {"Y": np.array(img_gray, dtype=np.float32)}
    else:
        print("Immagine a colori...")
        if img.mode != "RGB":
            img = img.convert("RGB")
        channels = rgb_to_ycbcr(img)

    processed_blocks_by_channel = {}

    for channel_name, channel_data in channels.items():
        channel_blocks = []
        blocks, pad_h, pad_w = get_nxn_blocks(channel_data)

        for block in blocks:
            dct_block = dct(block)

            # Y usa la matrice di quantizzazione della luminanza, Cb/Cr quella della crominanza
            is_luma = channel_name == "Y"
            q_block = quantize_block(dct_block, is_luma=is_luma)

            zz_array = zigzag_scan(q_block)
            channel_blocks.append(zz_array)

        processed_blocks_by_channel[channel_name] = channel_blocks

    print("Pipeline di preprocessing completata con successo.\n")

    print(
        f"\nAvvio della fase di codifica e decodifica"
        if method != "all"
        else f"\nAvvio della fase di codifica e decodifica (Tutti i metodi)\n"
    )

    methods_to_run = (
        ["huffman", "arithmetic_tables", "arithmetic_static", "qm"]
        if method == "all"
        else [method]
    )

    compressed_streams = {}
    custom_tables_dict = {}
    reconstructed_images = {}

    blocks_layout = {
        ch: len(blocks) for ch, blocks in processed_blocks_by_channel.items()
    }

    for m in methods_to_run:
        if m == "huffman":
            algo_name = "Huffman"
        elif m == "arithmetic_tables":
            algo_name = "Aritmetica Standard"
        elif m == "arithmetic_static":
            algo_name = "Aritmetica Statica"
        elif m == "qm":
            algo_name = "QM"
        else:
            algo_name = m.upper()

        pbar = tqdm(
            total=100,
            desc=f"Codifica {algo_name}",
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}%",
        )

        # --- Codifica ---
        stream, tables = encode_blocks(processed_blocks_by_channel, method=m)
        compressed_streams[m] = stream
        if tables is not None:
            custom_tables_dict[m] = tables

        pbar.update(50)

        # --- Decodifica ---
        tables_for_decode = tables if m == "arithmetic_static" else None
        decoded_blocks_by_channel = decode_blocks(
            stream, blocks_layout, method=m, custom_tables=tables_for_decode
        )

        reconstructed_channels = {}
        for channel_name, blocks in decoded_blocks_by_channel.items():
            is_luma = channel_name == "Y"
            spatial_blocks = []

            for block_1d in blocks:
                block_2d_q = inverse_zigzag_scan(block_1d)
                block_2d_dct = dequantize_block(block_2d_q, is_luma=is_luma)
                spatial_block = inv_dct(block_2d_dct)
                spatial_blocks.append(spatial_block)

            channel_matrix = reassemble_blocks(
                spatial_blocks,
                image_shape=(img.height, img.width),
                pad_h=pad_h,
                pad_w=pad_w,
            )
            channel_matrix = channel_matrix[: img.height, : img.width]
            reconstructed_channels[channel_name] = channel_matrix

        if grayscale or img.mode == "L":
            final_array = np.clip(reconstructed_channels["Y"], 0, 255).astype(np.uint8)
            final_img = Image.fromarray(final_array, mode="L")
        else:
            final_img = ycbcr_to_rgb(
                reconstructed_channels["Y"],
                reconstructed_channels["Cb"],
                reconstructed_channels["Cr"],
            )

        reconstructed_images[m] = final_img

        pbar.update(50)
        pbar.close()

    if method == "all":
        final_stream = compressed_streams
        final_tables = custom_tables_dict
    else:
        final_stream = compressed_streams[method]
        final_tables = custom_tables_dict.get(method, None)

    return final_stream, reconstructed_images, final_tables
