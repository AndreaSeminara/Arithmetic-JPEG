import numpy as np
from PIL import Image


def rgb_to_ycbcr(img: Image.Image) -> Image.Image:
    # PIL usa la conversione BT.601 studio swing, che è quella prevista dallo standard JPEG
    if img.mode != "RGB":
        raise ValueError("L'immagine deve essere in formato RGB.")

    return img.convert("YCbCr")


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> Image.Image:
    y_uint8 = np.clip(y, 0, 255).astype(np.uint8)
    cb_uint8 = np.clip(cb, 0, 255).astype(np.uint8)
    cr_uint8 = np.clip(cr, 0, 255).astype(np.uint8)

    y_img = Image.fromarray(y_uint8, mode="L")
    cb_img = Image.fromarray(cb_uint8, mode="L")
    cr_img = Image.fromarray(cr_uint8, mode="L")

    ycbcr_img = Image.merge("YCbCr", (y_img, cb_img, cr_img))

    return ycbcr_img.convert("RGB")


def extract_ycbcr_channels(img: Image.Image) -> dict[str, np.ndarray]:
    if img is None:
        raise ValueError(
            "Immagine non valida. Assicurati di fornire un'immagine valida."
        )

    img = rgb_to_ycbcr(img)

    y, cb, cr = img.split()
    return {
        "Y": np.array(y, dtype=np.float32),
        "Cb": np.array(cb, dtype=np.float32),
        "Cr": np.array(cr, dtype=np.float32),
    }
