from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EntropyEncoder(ABC):
    """
    Interfaccia astratta per i codificatori entropici JPEG.
    Tutti i codificatori specifici (Huffman, Arithmetic, QM) devono implementare questa interfaccia.
    """

    @abstractmethod
    def encode(self, blocks: List[np.ndarray]) -> bytes:
        """
        Riceve una lista di blocchi 8x8 e restituisce il flusso di byte compresso.

        Args:
            blocks: Lista di array NumPy 1D da 64 elementi.

        Returns:
            bytes: Flusso binario compresso pronto da scrivere su file.
        """
        pass
