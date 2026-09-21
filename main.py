import argparse
import os
from PIL import Image
from encoding import load_custom_jpeg, save_custom_jpeg
from preprocessing import run_pipeline
from utils import modes

ALGORITHMS_FLAGS = {
    "huffman": 0,
    "arithmetic_tables": 1,
    "arithmetic_static": 2,
    "qm": 3,
}


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

    compressed_stream, reconstructed_images, custom_tables = run_pipeline(
        img, grayscale=grayscale, method=method
    )

    print("\nSalvataggio immagini ricostruite e bitstream in corso...")

    output_dir = os.path.join("images", "output")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    streams = (
        compressed_stream
        if isinstance(compressed_stream, dict)
        else {method: compressed_stream}
    )

    for alg_name, stream in streams.items():
        algo_flag = ALGORITHMS_FLAGS.get(alg_name)
        if algo_flag is None:
            print(
                f"  [WARN] Algoritmo sconosciuto '{alg_name}', salvataggio .myjpeg ignorato"
            )
            continue

        # Estraiamo le tabelle di frequenza reali per l'aritmetica statica
        tables_for_save = None
        if algo_flag == 2:  # Arithmetic Static
            tables_for_save = (
                custom_tables.get(alg_name) if method == "all" else custom_tables
            )
            
            # Salva anche il .bin puro per mostrare la differenza di dimensione
            bin_filename = os.path.join(output_dir, f"{base_name}_{alg_name}.bin")
            try:
                with open(bin_filename, "wb") as f_bin:
                    f_bin.write(stream)
                print(f"  [OK] Bitstream puro salvato in: {bin_filename}")
            except Exception as exc_bin:
                print(f"  [ERRORE] Salvataggio .bin fallito ({bin_filename}): {exc_bin}")

        container_filename = os.path.join(output_dir, f"{base_name}_{alg_name}.myjpeg")

        try:
            save_custom_jpeg(
                filepath=container_filename,
                width=larghezza,
                height=altezza,
                algo_flag=algo_flag,
                bitstream=stream,
                custom_tables=tables_for_save,
            )

            loaded_w, loaded_h, loaded_flag, loaded_tables, loaded_stream = (
                load_custom_jpeg(container_filename)
            )

            if algo_flag == 2:
                if (
                    loaded_w != larghezza
                    or loaded_h != altezza
                    or loaded_flag != algo_flag
                    or loaded_tables is None
                ):
                    print(
                        f"  [WARN] Verifica del contenitore statico fallita per: {container_filename}"
                    )
                else:
                    print(
                        f"  [OK] Bitstream salvato in container: {container_filename}"
                    )
            elif (
                loaded_w != larghezza
                or loaded_h != altezza
                or loaded_flag != algo_flag
                or loaded_stream != stream
            ):
                print(
                    f"  [WARN] Verifica round-trip non perfetta per: {container_filename}"
                )
            else:
                print(f"  [OK] Bitstream salvato in container: {container_filename}")
        except Exception as exc:
            print(
                f"  [ERRORE] Salvataggio container fallito ({container_filename}): {exc}"
            )

    for alg_name, final_img in reconstructed_images.items():
        output_filename = os.path.join(output_dir, f"{base_name}_{alg_name}.png")

        final_img.save(output_filename, format="PNG")
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
        choices=["0", "1", "2", "3", "4"],
        help="Algoritmo di codifica entropica da utilizzare",
    )
    args = parser.parse_args()
    main(
        image_path=args.image_path,
        grayscale=args.grayscale,
        method=modes[int(args.method)],
    )
