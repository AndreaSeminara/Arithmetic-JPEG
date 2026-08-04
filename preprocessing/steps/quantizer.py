import numpy as np
from utils.tables import STD_LUMA_QMAT, STD_CHROMA_QMAT


def quantize_block(block: np.ndarray, is_luma: bool = True) -> np.ndarray:
    """Quantizza un blocco DCT"""
    qmat = STD_LUMA_QMAT if is_luma else STD_CHROMA_QMAT

    quantized_block = np.round(block / qmat).astype(np.int32)

    return quantized_block


def dequantize_block(block: np.ndarray, is_luma: bool = True) -> np.ndarray:
    """Dequantizza un blocco"""
    from utils.tables import STD_LUMA_QMAT, STD_CHROMA_QMAT

    qmat = STD_LUMA_QMAT if is_luma else STD_CHROMA_QMAT
    return block * qmat
