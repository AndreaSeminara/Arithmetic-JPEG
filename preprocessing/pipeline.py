import numpy as np
from PIL import Image
from .steps import (
    extract_ycbcr_channels,
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
        channels = extract_ycbcr_channels(img)

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
        f"\nAvvio della fase di codifica"
        if method != "all"
        else f"\nAvvio della fase di codifica con il metodo: {method}\n"
    )

    compressed_stream, custom_tables = encode_blocks(
        processed_blocks_by_channel, method=method
    )
    preview = (
        compressed_stream[:20]
        if not isinstance(compressed_stream, dict)
        else "Dizionario di flussi"
    )
    print(f"\nCodifica completata con successo: {preview}")

    # --- Decodifica e ricostruzione ---
    reconstructed_images = {}
    streams = (
        compressed_stream
        if isinstance(compressed_stream, dict)
        else {method: compressed_stream}
    )

    blocks_layout = {
        ch: len(blocks) for ch, blocks in processed_blocks_by_channel.items()
    }

    for encoding_name, stream in streams.items():
        print(f"\nAvvio decodifica per il metodo: {encoding_name.upper()}")

        # Per l'aritmetica statica servono le tabelle di frequenza costruite durante la codifica
        tables_for_decode = None
        if encoding_name == "arithmetic_static":
            tables_for_decode = (
                custom_tables.get(encoding_name) if method == "all" else custom_tables
            )
        decoded_blocks_by_channel = decode_blocks(
            stream, blocks_layout, method=encoding_name, custom_tables=tables_for_decode
        )

        reconstructed_channels = {}

        for channel_name, blocks in decoded_blocks_by_channel.items():
            print(f"  Ricostruzione geometrica canale {channel_name}...")
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

        print(f"  Conversione colore e creazione oggetto immagine...")

        if grayscale or img.mode == "L":
            final_array = np.clip(reconstructed_channels["Y"], 0, 255).astype(np.uint8)
            final_img = Image.fromarray(final_array, mode="L")
        else:
            final_img = ycbcr_to_rgb(
                reconstructed_channels["Y"],
                reconstructed_channels["Cb"],
                reconstructed_channels["Cr"],
            )

        reconstructed_images[encoding_name] = final_img

    return compressed_stream, reconstructed_images, custom_tables
