import numpy as np
from .steps import (
    extract_ycbcr_channels,
    get_nxn_blocks,
    dct,
    quantize_block,
    zigzag_scan,
)
from encoding import encode_blocks


def run_pipeline(img, grayscale: bool = False, method: str = "all"):
    """
    Esegue la pipeline di preprocessing sull'immagine fornita.

    Args:
        img (PIL.Image.Image): Immagine da elaborare.
        grayscale (bool): Se True, l'immagine viene convertita in scala di grigi se non lo è già.
        method (str): Metodo di compressione da utilizzare ("huffman", "arithmetic" o "qm").
    """
    print("Esecuzione della pipeline di preprocessing...\n")

    if img is None:
        print("Errore: Immagine non valida. Assicurati di fornire un'immagine valida.")
        return None

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

    print(f"\nCodifica completata con successo : {compressed_stream[:50]}")

    return None
