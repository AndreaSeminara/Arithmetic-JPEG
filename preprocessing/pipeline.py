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

    # ==================================================
    #               Fase di Codifica
    # ==================================================

    # STEP 1
    # Se l'immagine è in scala di grigi o se l'opzione grayscale è attiva, convertila in scala di grigi
    if grayscale or img.mode == "L":
        print("Immagine in scala di grigi...")

        img_gray = img.convert("L")
        channels = {"Y": np.array(img_gray, dtype=np.float32)}
    else:
        # Se l'immagine è a colori, converti in YCbCr e estrai i canali
        print("Immagine a colori...")
        if img.mode != "RGB":
            img = img.convert("RGB")

        channels = extract_ycbcr_channels(img)

    processed_blocks_by_channel = {}

    for channel_name, channel_data in channels.items():
        channel_blocks = []
        # STEP 2
        # Dividi ogni canale in blocchi 8x8
        blocks, pad_h, pad_w = get_nxn_blocks(channel_data)

        for block in blocks:
            # STEP 3
            # Trasformata Discreta del Coseno (DCT) su ogni blocco
            dct_block = dct(block)

            # STEP 4
            # Quantizzazione su ogni blocco DCT
            # Distinguo tra canale Y (luminanza) e canali Cb/Cr (crominanza) per usare la matrice di quantizzazione corretta
            is_luma = channel_name == "Y"
            q_block = quantize_block(dct_block, is_luma=is_luma)

            # STEP 5
            # Zig-Zag scan su ogni blocco quantizzato
            zz_array = zigzag_scan(q_block)

            # Aggiungi il blocco alla lista del canale
            channel_blocks.append(zz_array)

        processed_blocks_by_channel[channel_name] = channel_blocks

    print("Pipeline di preprocessing completata con successo.\n")

    print(
        f"\nAvvio della fase di codifica"
        if method != "all"
        else f"\nAvvio della fase di codifica con il metodo: {method}\n"
    )

    compressed_stream = encode_blocks(processed_blocks_by_channel, method=method)

    preview = (
        compressed_stream[:20]
        if not isinstance(compressed_stream, dict)
        else "Dizionario di flussi"
    )
    print(f"\nCodifica completata con successo: {preview}")

    # ==================================================
    #       Fase di Decodifica e Ricostruzione
    # ==================================================
    reconstructed_images = {}
    streams = (
        compressed_stream
        if isinstance(compressed_stream, dict)
        else {method: compressed_stream}
    )

    # Calcoliamo quanti blocchi ci sono per ogni canale
    blocks_layout = {
        ch: len(blocks) for ch, blocks in processed_blocks_by_channel.items()
    }

    for encoding_name, stream in streams.items():
        print(f"\nAvvio decodifica per il metodo: {encoding_name.upper()}")

        # STEP 1
        # Decodifica dei blocchi
        decoded_blocks_by_channel = decode_blocks(
            stream, blocks_layout, method=encoding_name
        )

        reconstructed_channels = {}

        for channel_name, blocks in decoded_blocks_by_channel.items():
            print(f"  Ricostruzione geometrica canale {channel_name}...")
            is_luma = channel_name == "Y"
            spatial_blocks = []

            for block_1d in blocks:
                # Step 2
                # Inverse Zig-Zag
                block_2d_q = inverse_zigzag_scan(block_1d)

                # Step 3
                # Dequantizzazione
                block_2d_dct = dequantize_block(block_2d_q, is_luma=is_luma)

                # Step 4
                # Inverse DCT
                spatial_block = inv_dct(block_2d_dct)

                spatial_blocks.append(spatial_block)

            # Step 5
            # Unione dei blocchi 8x8
            channel_matrix = reassemble_blocks(
                spatial_blocks,
                image_shape=(img.height, img.width),
                pad_h=pad_h,
                pad_w=pad_w,
            )
            reconstructed_channels[channel_name] = channel_matrix

        print(f"  Conversione colore e creazione oggetto immagine...")

        # STEP 6
        # Conversione da YCbCr a RGB (o scala di grigi)
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

    return compressed_stream, reconstructed_images
