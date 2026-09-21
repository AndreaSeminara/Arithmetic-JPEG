# Arithmetic-JPEG

Compressore JPEG che confronta quattro diversi codificatori Huffman, Aritmetica Statica, Aritmetica con Tabelle e QM-Coder 
L'obiettivo è capire come cambia l'efficienza di compressione ptra i vari metodi mantenendo invariata tutta la pipeline JPEG sottostante: conversione YCbCr, divisione in blocchi 8×8, DCT, quantizzazione e scan zig-zag.

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
│   ├── compression.py      # Access Point per gestire quale algoritmo di codifica usare
│   │
│   └── coders/         # Algoritmi di Codifica
│       ├── __init__.py
│       ├── base.py         # Classe astratta per i codificatori
│       ├── huffman.py      # Huffman
│       ├── arithmetic.py   # Aritmetica - Statica e con Tabelle
│       └── qm.py           # QM-Coder
|
└── utils/              # Utility Generale
│   ├── __init__.py
│   ├── modes.py            # Variabile Globale per Modalità d'Uso del Comando
│   ├── tables.py           # Tabelle dello Standard T.81 per Huffman e Quantizzazione
|
└── images/             # Immagine di Input e Risultati in Output
```

## Installazione

**Con Conda:**
```bash
conda env create -f environment.yml
conda activate arithmetic_jpeg
```

**Con pip:**
```bash
pip install -r requirements.txt
```

## Utilizzo

```bash
python main.py [--image_path PATH] [--method N] [--grayscale]
```

### Flag disponibili

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--image_path PATH` | `images/lena.png` | Percorso dell'immagine da elaborare |
| `--method N` | `0` | Algoritmo di codifica entropica (vedi tabella sotto) |
| `--grayscale` | — | Se presente, elabora l'immagine in scala di grigi |

### Valori di `--method`

| N | Algoritmo |
|---|-----------|
| `0` | Tutti (esegue e confronta tutti e quattro) |
| `1` | Huffman |
| `2` | Aritmetica con Tabelle |
| `3` | Aritmetica Statica |
| `4` | QM-Coder |

I risultati vengono salvati in `images/output/`, con un file `.png` (immagine ricostruita) e un file `.myjpeg` (bitstream) per ogni algoritmo.


