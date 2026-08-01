import argparse
import os
from PIL import Image
from preprocessing import run_pipeline
from utils import modes


def main(image_path, grayscale: bool = False, method: str = "huffman"):
    print(f"-- Inizio Pipeline Arithmetic-JPEG --")
    print(f"Tentativo di caricamento dell'immagine : {image_path}")

    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata in {image_path}")
        return

    img = Image.open(image_path)

    larghezza, altezza = img.size
    print(f"Immagine caricata con successo: {image_path}\n")
    print(
        f"Info Immagine: Dimensioni: {larghezza}x{altezza}, Formato: {img.format}, Colore: {img.mode}\n"
    )
    print(f"Inizio elaborazione dell'immagine...\n")

    run_pipeline(img, grayscale=grayscale, method=method)

    print(f"-- Fine Pipeline Arithmetic-JPEG --")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Arithmetic-JPEG")
    parser.add_argument(
        "--image_path",
        type=str,
        default="images/lena.png",
        help="Percorso dell'immagine da elaborare (default: images/lena.png)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Se specificato, elabora l'immagine in bianco e nero",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="0",
        choices=["0", "1", "2", "3"],
        help="Algoritmo di codifica entropica da utilizzare",
    )
    args = parser.parse_args()
    main(
        image_path=args.image_path,
        grayscale=args.grayscale,
        method=modes[int(args.method)],
    )
