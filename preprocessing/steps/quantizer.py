import numpy as np
from utils.tables import STD_LUMA_QMAT, STD_CHROMA_QMAT


def quantize_block(block, is_luma=True):
    """
    Quantizza un blocco DCT utilizzando la tabella di quantizzazione standard JPEG.

    Args:
        block (np.ndarray): Blocco DCT da quantizzare.
        is_luma (bool): Se True, utilizza la tabella di quantizzazione per la luminanza (Y).
                        Se False, utilizza la tabella di quantizzazione per la crominanza (Cb, Cr).

    Returns:
        np.ndarray: Blocco quantizzato.
    """
    qmat = STD_LUMA_QMAT if is_luma else STD_CHROMA_QMAT

    # Quantizzazione: dividi elemento per elemento e arrotonda al numero intero più vicino
    quantized_block = np.round(block / qmat).astype(np.int32)

    return quantized_block
