# Arithmetic-JPEG

An educational JPEG compressor comparing the standard Huffman baseline with Static Arithmetic and Adaptive Arithmetic (QM-Coder) entropy coding

## Project Structure

```
arithmetic_jpeg/
│
├── main.py # Access point
│
├── preprocessing/      # Elaborazione dell'immagine
│   ├── __init__.py
│   ├── pipeline.py     # Avvia la pipeline JPEG una volta ricevuta un'immagine
│   │
│   └── steps/          # Step della pipeline
│       ├── __init__.py
│       ├── color.py        # Step 1: Conversione RGB <-> YCbCr (se necessaria)
│       ├── blocking.py     # Step 2: Divisione in blocchi 8x8
│       ├── dct.py          # Step 3: Trasformata Discreta del Coseno
│       ├── quantizer.py    # Step 4: Quantizzazione
│       └── zigzag.py       # Step 5: Scanning Zig Zag
│
└── encoding/           # Codifiche
│   ├── __init__.py
│   ├── encoder.py      # Access Point per gestire quale algoritmo di codifica usare
│   ├── decoder.py      # Access Point per gestire quale algoritmo di decodifca usare
│   │
│   └── coders/         # Algoritmi di Codifica
│       ├── __init__.py
│       ├── base.py         # Classe astratta per i codificatori
│       └── huffman.py      # Huffman
└── utils/              # Utility Generale
│   ├── __init__.py
│   ├── modes.py            # Variabile Globale per Modalità d'Uso del Comando
│   ├── tables.py           # Tabelle dello Standard T.81 per Huffman e Quantizzazione
```
