import numpy as np
from PIL import Image


def rgb_to_ycbcr(img):
    """
    Converte un'immagine RGB in YCbCr.

    Args:
        img (PIL.Image.Image): Immagine in formato RGB.

    Returns:
        PIL.Image.Image: Immagine convertita in YCbCr.
    """
    if img.mode != "RGB":
        raise ValueError("L'immagine deve essere in formato RGB.")

    return img.convert("YCbCr")


def extract_ycbcr_channels(img):
    """
    Estrae i canali Y, Cb e Cr da un'immagine YCbCr.

    Args:
        img (PIL.Image.Image): Immagine in formato YCbCr
    """
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
