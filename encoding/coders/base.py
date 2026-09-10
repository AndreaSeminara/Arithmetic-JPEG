from abc import ABC, abstractmethod
import numpy as np

# Interfacce base per garantire il polimorfismo.
# In questo modo la pipeline principale non si preoccupa dell'algoritmo specifico:
# chiama solo encode() o decode() e passa i blocchi, che sia Huffman, QM o Aritmetica.

class EntropyEncoder(ABC):
    
    @abstractmethod
    def encode(self, blocks, is_luma=True):
        # Prende in input la lista dei blocchi 8x8 (già passati per DCT, quantizzazione e zigzag)
        # e sputa fuori il flusso di byte raw (il bitstream compresso).
        pass


class EntropyDecoder(ABC):
    
    @abstractmethod
    def decode(self, byte_stream, num_blocks, is_luma=True, custom_tables=None):
        # Legge il flusso compresso e deve restituire due cose:
        # 1. blocks: La lista dei blocchi 8x8 estratti e pronti per la dequantizzazione.
        # 2. bytes_consumed: Quanti byte esatti si è "mangiato". Fondamentale per 
        #    non far sballare il puntatore quando passiamo dal canale Y al canale Cb/Cr.
        pass