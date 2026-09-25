import numpy as np
import struct

from utils.tables import (
    STD_DC_LUMA_BITS,
    STD_DC_LUMA_VALS,
    STD_AC_LUMA_BITS,
    STD_AC_LUMA_VALS,
    STD_DC_CHROMA_BITS,
    STD_DC_CHROMA_VALS,
    STD_AC_CHROMA_BITS,
    STD_AC_CHROMA_VALS,
)
from .base import EntropyEncoder, EntropyDecoder

#  Core Aritmetico (Aritmetica intera a 32-bit)
# Lavoriamo con interi enormi invece dei float per evitare che
# l'arrotondamento faccia sbagliare il decoder (drifting)


class ArithEncoderCore:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF  # Rappresenta 1.0 in aritmetica intera a 32 bit
        self.underflow = 0  # Contatore di underflow: bit in sospeso quando l'intervallo cade nella zona centrale
        self.bit_str = ""

    def update(self, bounds):
        # Restringe l'intervallo in base alla probabilità del simbolo corrente.
        # bounds contiene: (frequenza_min, frequenza_max, totale_frequenze)
        low_c, high_c, tot_c = bounds
        rng = self.high - self.low + 1

        # Calcoliamo i nuovi limiti con una divisione intera
        low_off = (rng * low_c) // tot_c
        high_off = (rng * high_c) // tot_c

        self.high = self.low + high_off - 1
        self.low = self.low + low_off

        # Dopo aver ristretto l'intervallo, si verifica se è necessario rinormalizzare
        self._renorm()

    def _renorm(self):
        # Se low e high ricadono nella stessa metà dell'intervallo globale,
        # il bit più significativo è ormai noto: viene emesso e l'intervallo
        # viene raddoppiato (shift a sinistra) per guadagnare di nuovo precisione.
        HALF = 0x80000000
        QTR = 0x40000000
        while True:
            if self.high < HALF:
                # Entrambi sotto la metà: il bit sicuro è 0
                self._emit(0)
                self.low = (self.low * 2) & 0xFFFFFFFF
                self.high = ((self.high * 2) + 1) & 0xFFFFFFFF
            elif self.low >= HALF:
                # Entrambi sopra la metà: il bit sicuro è 1
                self._emit(1)
                self.low = ((self.low - HALF) * 2) & 0xFFFFFFFF
                self.high = (((self.high - HALF) * 2) + 1) & 0xFFFFFFFF
            elif self.low >= QTR and self.high < (HALF + QTR):
                # Rischio underflow: l'intervallo cade nella zona centrale (0.5).
                # Il bit non è ancora determinabile: si incrementa il contatore e si espande l'intervallo.
                self.underflow += 1
                self.low = ((self.low - QTR) * 2) & 0xFFFFFFFF
                self.high = (((self.high - QTR) * 2) + 1) & 0xFFFFFFFF
            else:
                # L'intervallo è ancora abbastanza largo: si esce dal ciclo.
                break

    def _emit(self, bit):
        # Scrive il bit confermato e aggiunge il complemento per
        # tutti i bit di underflow rimasti in sospeso
        self.bit_str += str(bit)
        self.bit_str += str(1 - bit) * self.underflow
        self.underflow = 0

    def finish(self):
        # Prima di chiudere, si forza l'emissione dell'ultimo bit di underflow.
        self.underflow += 1
        if self.low >= 0x40000000:
            self._emit(1)
        else:
            self._emit(0)

        # Aggiunge 32 zeri finali così il decoder ha abbastanza bit da leggere
        # senza andare oltre il segmento corrente.
        self.bit_str += "0" * 32
        return self.bit_str


class ArithDecoderCore:
    def __init__(self, bit_str):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.value = 0
        self.bit_str = bit_str
        self.idx = 0

        # Carica i primi 32 bit del flusso nel registro value per avviare la decodifica.
        for _ in range(32):
            self.value = (self.value * 2) + self._read_bit()

    def _read_bit(self):
        if self.idx < len(self.bit_str):
            bit = int(self.bit_str[self.idx])
            self.idx += 1
            return bit
        return 0

    def get_offset(self):
        # Restituisce l'offset del valore rispetto al limite inferiore e la larghezza dell'intervallo.
        rng = self.high - self.low + 1
        target = self.value - self.low
        return target, rng

    def update(self, bounds):
        # Aggiorna i limiti esattamente come fa l'encoder
        low_c, high_c, tot_c = bounds
        rng = self.high - self.low + 1

        self.high = self.low + ((rng * high_c) // tot_c) - 1
        self.low = self.low + ((rng * low_c) // tot_c)
        self._renorm()

    def _renorm(self):
        # Riproduce gli stessi shift dell'encoder per rimanere sincronizzato,
        # ma in più legge i nuovi bit in arrivo.
        HALF = 0x80000000
        QTR = 0x40000000
        while True:
            if self.high < HALF:
                self.low = (self.low * 2) & 0xFFFFFFFF
                self.high = ((self.high * 2) + 1) & 0xFFFFFFFF
                self.value = ((self.value * 2) + self._read_bit()) & 0xFFFFFFFF
            elif self.low >= HALF:
                self.low = ((self.low - HALF) * 2) & 0xFFFFFFFF
                self.high = (((self.high - HALF) * 2) + 1) & 0xFFFFFFFF
                self.value = (((self.value - HALF) * 2) + self._read_bit()) & 0xFFFFFFFF
            elif self.low >= QTR and self.high < (HALF + QTR):
                self.low = ((self.low - QTR) * 2) & 0xFFFFFFFF
                self.high = (((self.high - QTR) * 2) + 1) & 0xFFFFFFFF
                self.value = (((self.value - QTR) * 2) + self._read_bit()) & 0xFFFFFFFF
            else:
                break


#  Standard Aritmetico (T.81)


class ArithmeticStandard(EntropyEncoder, EntropyDecoder):
    def __init__(self):
        # Carica le tabelle delle probabilità standard del JPEG
        self.dc_luma = self._build_cdf(STD_DC_LUMA_BITS, STD_DC_LUMA_VALS)
        self.ac_luma = self._build_cdf(STD_AC_LUMA_BITS, STD_AC_LUMA_VALS)
        self.dc_chroma = self._build_cdf(STD_DC_CHROMA_BITS, STD_DC_CHROMA_VALS)
        self.ac_chroma = self._build_cdf(STD_AC_CHROMA_BITS, STD_AC_CHROMA_VALS)

    def _get_cat(self, val):
        # Mappa il valore del coefficiente nella sua categoria VLI (dimensione in bit)
        if val == 0:
            return 0, ""
        size = abs(val).bit_length()
        if val > 0:
            bits = bin(val)[2:]
        else:
            comp = (1 << size) + val - 1
            bits = bin(comp)[2:].zfill(size)
        return size, bits

    def _dec_val(self, size, bits):
        # Operazione inversa: dai bit riottiene il coefficiente originale
        if size == 0:
            return 0
        if bits[0] == "1":
            return int(bits, 2)
        return int(bits, 2) - (1 << size) + 1

    def _build_cdf(self, bits, vals):
        # Costruisce la funzione di ripartizione (CDF) dalle tabelle.
        # Servono le somme cumulative per calcolare le probabilità frazionarie.
        freqs = {}
        idx = 0
        for length in range(1, 17):
            for _ in range(bits[length - 1]):
                freqs[vals[idx]] = 1 << (16 - length)
                idx += 1

        tot = sum(freqs.values())
        cdf = {}
        cum = 0
        for sym, freq in freqs.items():
            cdf[sym] = (cum, cum + freq, tot)
            cum += freq
        return cdf

    def encode(self, blocks, is_luma=True):
        core = ArithEncoderCore()
        prev_dc = 0
        dc_cdf = self.dc_luma if is_luma else self.dc_chroma
        ac_cdf = self.ac_luma if is_luma else self.ac_chroma

        for block in blocks:
            # Il DC si codifica come differenza col blocco precedente (DPCM)
            dc_val = int(block[0])
            diff = dc_val - prev_dc
            prev_dc = dc_val

            dc_size, dc_bits = self._get_cat(diff)
            core.update(dc_cdf[dc_size])

            # Invia i bit grezzi considerandoli equiprobabili (prob. 1/2)
            for bit in dc_bits:
                core.update((0, 1, 2) if bit == "0" else (1, 2, 2))

            # I coefficienti AC usano il run-length encoding
            run = 0
            for ac_val in block[1:]:
                ac_val = int(ac_val)
                if ac_val == 0:
                    run += 1
                    if run == 16:
                        # ZRL (Zero Run Length): blocco di 16 zeri consecutivi
                        core.update(ac_cdf[0xF0])
                        run = 0
                else:
                    # Codifica il coefficiente accoppiando il numero di zeri (run)
                    # alla dimensione (size) del valore
                    ac_size, ac_bits = self._get_cat(ac_val)
                    ac_key = (run << 4) | ac_size
                    core.update(ac_cdf[ac_key])

                    for bit in ac_bits:
                        core.update((0, 1, 2) if bit == "0" else (1, 2, 2))
                    run = 0

            if run > 0:
                # EOB (End of Block): da qui in poi sono tutti zeri
                core.update(ac_cdf[0x00])

        bit_str = core.finish()

        # Allinea i bit per riempire bytes interi (padding pulito)
        rem = len(bit_str) % 8
        if rem != 0:
            bit_str += "0" * (8 - rem)

        b_array = bytearray()
        for i in range(0, len(bit_str), 8):
            b_array.append(int(bit_str[i : i + 8], 2))

        data = bytes(b_array)
        # Aggiunge 4 byte di header con la lunghezza del dato, così il decoder può isolare il suo segmento
        return struct.pack(">I", len(data)) + data

    def _find_sym(self, target, rng, cdf):
        # Trova il simbolo il cui intervallo comprende la posizione target corrente
        for sym, (low_c, high_c, tot_c) in cdf.items():
            low_off = (rng * low_c) // tot_c
            high_off = (rng * high_c) // tot_c
            if low_off <= target < high_off:
                return sym
        return list(cdf.keys())[-1]

    def decode(self, byte_stream, num_blocks, is_luma=True):
        # Legge la lunghezza dai primi 4 byte e isola il segmento del canale corrente
        data_len = struct.unpack(">I", byte_stream[:4])[0]
        ch_bytes = byte_stream[4 : 4 + data_len]

        bit_str = "".join(f"{b:08b}" for b in ch_bytes)
        core = ArithDecoderCore(bit_str)

        dc_cdf = self.dc_luma if is_luma else self.dc_chroma
        ac_cdf = self.ac_luma if is_luma else self.ac_chroma

        blocks = []
        prev_dc = 0

        for _ in range(num_blocks):
            block = np.zeros(64, dtype=np.float32)

            #  Lettura del DC
            target, rng = core.get_offset()
            dc_size = self._find_sym(target, rng, dc_cdf)
            core.update(dc_cdf[dc_size])

            dc_bits = ""
            for _ in range(dc_size):
                target, rng = core.get_offset()
                if target < (rng // 2):
                    core.update((0, 1, 2))
                    dc_bits += "0"
                else:
                    core.update((1, 2, 2))
                    dc_bits += "1"

            prev_dc += self._dec_val(dc_size, dc_bits)
            block[0] = prev_dc

            #  Lettura degli AC
            idx = 1
            while idx < 64:
                target, rng = core.get_offset()
                ac_key = self._find_sym(target, rng, ac_cdf)
                core.update(ac_cdf[ac_key])

                if ac_key == 0x00:
                    break  # Trovato EOB, il resto del blocco rimane a zero
                elif ac_key == 0xF0:
                    idx += 16  # ZRL: si avanzano 16 posizioni
                else:
                    run = ac_key >> 4
                    size = ac_key & 0x0F
                    idx += run

                    if idx < 64:
                        ac_bits = ""
                        for _ in range(size):
                            target, rng = core.get_offset()
                            if target < (rng // 2):
                                core.update((0, 1, 2))
                                ac_bits += "0"
                            else:
                                core.update((1, 2, 2))
                                ac_bits += "1"

                        block[idx] = self._dec_val(size, ac_bits)
                    idx += 1
            blocks.append(block)

        return blocks, 4 + data_len


#  Aritmetica Statica (Dinamica su immagine)


class ArithmeticStatic(EntropyEncoder, EntropyDecoder):
    def __init__(self):
        self.dc_cdf = {}
        self.ac_cdf = {}

    def _get_cat(self, val):
        if val == 0:
            return 0, ""
        size = abs(val).bit_length()
        if val > 0:
            bits = bin(val)[2:]
        else:
            comp = (1 << size) + val - 1
            bits = bin(comp)[2:].zfill(size)
        return size, bits

    def _dec_val(self, size, bits):
        if size == 0:
            return 0
        if bits[0] == "1":
            return int(bits, 2)
        return int(bits, 2) - (1 << size) + 1

    def _build_cdf(self, freqs):
        # A differenza dello standard, qui costruiamo la ripartizione
        # direttamente dai conteggi reali estratti dall'immagine
        tot = sum(freqs.values())
        cdf = {}
        cum = 0
        for sym in sorted(freqs.keys()):
            count = freqs[sym]
            cdf[sym] = (cum, cum + count, tot)
            cum += count
        return cdf

    def encode(self, blocks, is_luma=True):
        ch_name = "Y" if is_luma else "CbCr"
        dc_freqs = {}
        ac_freqs = {}
        prev_dc = 0

        # PASSATA 1: Prima scansione per raccogliere le frequenze dei simboli
        for block in blocks:
            dc_val = int(block[0])
            diff = dc_val - prev_dc
            prev_dc = dc_val
            dc_size, _ = self._get_cat(diff)
            dc_freqs[dc_size] = dc_freqs.get(dc_size, 0) + 1

            run = 0
            for ac_val in block[1:]:
                ac_val = int(ac_val)
                if ac_val == 0:
                    run += 1
                    if run == 16:
                        ac_freqs[0xF0] = ac_freqs.get(0xF0, 0) + 1
                        run = 0
                else:
                    ac_size, _ = self._get_cat(ac_val)
                    key = (run << 4) | ac_size
                    ac_freqs[key] = ac_freqs.get(key, 0) + 1
                    run = 0
            if run > 0:
                ac_freqs[0x00] = ac_freqs.get(0x00, 0) + 1

        # Modelliamo le probabilità su misura per questa specifica immagine
        self.dc_cdf = self._build_cdf(dc_freqs)
        self.ac_cdf = self._build_cdf(ac_freqs)

        # PASSATA 2: Codifica effettiva con le probabilità calcolate dalla prima scansione
        core = ArithEncoderCore()
        prev_dc = 0

        for block in blocks:
            dc_val = int(block[0])
            diff = dc_val - prev_dc
            prev_dc = dc_val

            dc_size, dc_bits = self._get_cat(diff)
            core.update(self.dc_cdf[dc_size])

            for bit in dc_bits:
                core.update((0, 1, 2) if bit == "0" else (1, 2, 2))

            run = 0
            for ac_val in block[1:]:
                ac_val = int(ac_val)
                if ac_val == 0:
                    run += 1
                    if run == 16:
                        core.update(self.ac_cdf[0xF0])
                        run = 0
                else:
                    ac_size, ac_bits = self._get_cat(ac_val)
                    key = (run << 4) | ac_size
                    core.update(self.ac_cdf[key])

                    for bit in ac_bits:
                        core.update((0, 1, 2) if bit == "0" else (1, 2, 2))
                    run = 0
            if run > 0:
                core.update(self.ac_cdf[0x00])

        bit_str = core.finish()
        rem = len(bit_str) % 8
        if rem != 0:
            bit_str += "0" * (8 - rem)

        b_array = bytearray()
        for i in range(0, len(bit_str), 8):
            b_array.append(int(bit_str[i : i + 8], 2))

        data = bytes(b_array)
        payload = struct.pack(">I", len(data)) + data

        # Restituisce il payload compresso e le tabelle di frequenza necessarie per la decodifica
        return payload, {"dc": dc_freqs, "ac": ac_freqs}

    def _find_sym(self, target, rng, cdf):
        for sym, (low_c, high_c, tot_c) in cdf.items():
            low_off = (rng * low_c) // tot_c
            high_off = (rng * high_c) // tot_c
            if low_off <= target < high_off:
                return sym
        return list(cdf.keys())[-1]

    def decode(self, byte_stream, num_blocks, is_luma=True, custom_tables=None):
        if custom_tables is None:
            raise ValueError("Mancano le tabelle per l'aritmetica statica")

        data_len = struct.unpack(">I", byte_stream[:4])[0]
        ch_bytes = byte_stream[4 : 4 + data_len]

        # Le probabilità vengono ricostruite dalle frequenze salvate nel file .myjpeg
        self.dc_cdf = self._build_cdf(custom_tables["dc"])
        self.ac_cdf = self._build_cdf(custom_tables["ac"])

        bit_str = "".join(f"{b:08b}" for b in ch_bytes)
        core = ArithDecoderCore(bit_str)

        blocks = []
        prev_dc = 0
        ch_name = "Y" if is_luma else "CbCr"

        for _ in range(num_blocks):
            block = np.zeros(64, dtype=np.float32)

            #  Lettura DC
            target, rng = core.get_offset()
            dc_size = self._find_sym(target, rng, self.dc_cdf)
            core.update(self.dc_cdf[dc_size])

            dc_bits = ""
            for _ in range(dc_size):
                target, rng = core.get_offset()
                if target < (rng // 2):
                    core.update((0, 1, 2))
                    dc_bits += "0"
                else:
                    core.update((1, 2, 2))
                    dc_bits += "1"

            prev_dc += self._dec_val(dc_size, dc_bits)
            block[0] = prev_dc

            #  Lettura AC
            idx = 1
            while idx < 64:
                target, rng = core.get_offset()
                ac_key = self._find_sym(target, rng, self.ac_cdf)
                core.update(self.ac_cdf[ac_key])

                if ac_key == 0x00:
                    break
                elif ac_key == 0xF0:
                    idx += 16
                else:
                    run = ac_key >> 4
                    size = ac_key & 0x0F
                    idx += run

                    if idx < 64:
                        ac_bits = ""
                        for _ in range(size):
                            target, rng = core.get_offset()
                            if target < (rng // 2):
                                core.update((0, 1, 2))
                                ac_bits += "0"
                            else:
                                core.update((1, 2, 2))
                                ac_bits += "1"

                        block[idx] = self._dec_val(size, ac_bits)
                    idx += 1
            blocks.append(block)

        return blocks, 4 + data_len
