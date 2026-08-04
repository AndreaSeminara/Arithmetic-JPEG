from abc import ABC, abstractmethod
import numpy as np


class EntropyEncoder(ABC):
    """Definisce la base dei codificatori entropici JPEG"""

    @abstractmethod
    def encode(self, blocks: list[np.ndarray]) -> bytes:
        """Restituisce i byte compressi dei blocchi in input"""
        pass


class EntropyDecoder(ABC):
    @abstractmethod
    def decode(
        self, byte_stream: bytes, num_blocks: int, is_luma: bool = True
    ) -> tuple[list[np.ndarray], int]:
        """Restituisce i blocchi decodificati e i byte consumati"""
        pass
