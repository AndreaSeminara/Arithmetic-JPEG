# Entry Point


import argparse
import os
from PIL import Image


def main(image_path):
    print(f"-- Inizio Pipeline Arithmetic-JPEG --")
    print(f"Tentativo di caricamento dell'immagine : {image_path}")

    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata in {image_path}")
        return

    img = Image.open(image_path)

    altezza, larghezza = img.size
    print(f"Immagine caricata con successo: {image_path}")
    print(
        f"Info Immagine: Dimensioni: {larghezza}x{altezza}, Formato: {img.format}, Colore: {img.mode}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Arithmetic-JPEG")
    parser.add_argument(
        "--image_path",
        type=str,
        default="images/lena.png",
        help="Percorso dell'immagine da elaborare (default: images/lena.png)",
    )
    args = parser.parse_args()
    main(image_path=args.image_path)
