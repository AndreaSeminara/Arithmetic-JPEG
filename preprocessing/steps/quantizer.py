import numpy as np
from utils.tables import STD_LUMA_QMAT, STD_CHROMA_QMAT


def quantize_block(block: np.ndarray, is_luma: bool = True) -> np.ndarray:
    qmat = STD_LUMA_QMAT if is_luma else STD_CHROMA_QMAT
    return np.round(block / qmat).astype(np.int32)


def dequantize_block(block: np.ndarray, is_luma: bool = True) -> np.ndarray:

    qmat = STD_LUMA_QMAT if is_luma else STD_CHROMA_QMAT
    return block * qmat
