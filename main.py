import argparse
import os
from PIL import Image
from preprocessing import run_pipeline
from utils import modes


def main(image_path: str, grayscale: bool = False, method: str = "huffman") -> None:
    print("-- Inizio pipeline Arithmetic-JPEG --")
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

    compressed_stream, reconstructed_images = run_pipeline(
        img, grayscale=grayscale, method=method
    )

    print("\nSalvataggio immagini ricostruite in corso...")

    output_dir = os.path.join("images", "output")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]

    for alg_name, final_img in reconstructed_images.items():
        output_filename = os.path.join(output_dir, f"{base_name}_{alg_name}.jpeg")

        final_img.save(output_filename, format="JPEG", quality=100)
        print(f"  [OK] Immagine ricostruita salvata in: {output_filename}")

    print("\n-- Fine pipeline Arithmetic-JPEG --\n")


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
