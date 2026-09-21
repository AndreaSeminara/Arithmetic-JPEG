import struct
import numpy as np
from typing import List, Tuple
from .base import EntropyEncoder, EntropyDecoder
from utils.tables import QM_ST_TABLE


#  QM-CODER 
#
#  Il QM-Coder è il codificatore aritmetico binario specificato dallo
#  standard JPEG. Lavora su decisioni binarie (0/1) e stima le probabilità
#  tramite una macchina a stati finiti che si adatta automaticamente alla statistica del segnale.
#
#  Registri:
#   - A  : ampiezza dell'intervallo di codifica (Q16, 0x10000 = 1.0)
#   - C  : registro di codifica, accumula i bit prima dell'emissione
#   - CT : contatore dei bit liberi in C prima del prossimo BYTEOUT
#   - B  : ultimo byte emesso, tenuto in sospeso per gestire il carry
#   - SC : numero di byte 0xFF accumulati in attesa di carry resolution
#   - ZC : numero di byte 0x00 consecutivi rimandati (ottimizzazione)


class QMCoderCore:
    _ST_TABLE = QM_ST_TABLE

    def __init__(self, debug=False):
        self.debug = debug
        # Ogni contesto ha il proprio indice di stato (st) e senso MPS (mps).
        # 600 contesti sono sufficienti per luma + chroma (300 ciascuno).
        self.st = bytearray(600)
        self.mps = bytearray(600)

        # Registri Encoder
        self.A = 0
        self.C = 0
        self.CT = 0
        # B = -1 è la sentinella "nessun byte in sospeso":
        # evita il flag _first_byte usato in implementazioni più semplici.
        self.B = -1
        self.ZC = 0
        self.SC = 0
        self._out = []

        # Registri Decoder
        self._in = b""
        self._in_idx = 0

    #  Inizializzazione (T.81 §D.1.7 / §D.2.7)

    def init_enc(self):
        self.A = 0x10000  # Intervallo pieno = 1.0 in fixed-point Q16
        self.C = 0
        # CT = 11 anziché 12: i primi 3 bit di C sono spacer di guardia
        # che impediscono al carry di propagarsi oltre il registro (T.81 §D.1.5).
        # Con B = -1 il primo BYTEOUT non emette nulla, equivale all'approccio
        # standard con BP = BPST - 1.
        self.CT = 11
        self.B = -1
        self.ZC = 0
        self.SC = 0
        self._out = []

    def init_dec(self, stream: bytes):
        """Carica i primi byte nel registro C per avviare la decodifica (T.81 §D.2.7)."""
        self._in = stream
        self._in_idx = 0
        self.B = self._next_byte()
        self.C = self.B << 16
        self._byte_in()
        self.C = (self.C << 8) & 0xFFFFFFFF
        self.CT -= 8
        self.A = 0x10000

    def _next_byte(self) -> int:
        """Legge il prossimo byte dallo stream; restituisce 0 se esaurito."""
        if self._in_idx < len(self._in):
            b = self._in[self._in_idx]
            self._in_idx += 1
            return b
        return 0

    #  Codifica binaria adattiva (T.81 Fig. D.7)

    def encode_bin(self, cx: int, decision: int):
        """Codifica una decisione binaria nel contesto cx.

        L'intervallo [0, A) viene suddiviso in due sotto-intervalli:
          - MPS (Most Probable Symbol): ampiezza A - Qe
          - LPS (Least Probable Symbol): ampiezza Qe

        Quando A - Qe < Qe si verifica il "conditional exchange":
        i sotto-intervalli vengono scambiati per mantenere la coerenza
        tra la codifica e la stima di probabilità (T.81 §D.1.3).
        """
        state = self.st[cx]
        Qe, NMPS, NLPS, SWITCH = self._ST_TABLE[state]

        self.A -= Qe
        if decision == self.mps[cx]:
            # --- Ramo MPS ---
            if self.A < 0x8000:
                # Conditional exchange: se l'intervallo MPS (A) è diventato
                # più piccolo di quello LPS (Qe), li scambiamo.
                if self.A < Qe:
                    self.C = (self.C + self.A) & 0xFFFFFFFF
                    self.A = Qe
                self.st[cx] = NMPS
                self._renorm_e()
            # Se A >= 0x8000 l'intervallo è ancora nel range valido:
            # non serve né aggiornare lo stato né rinormalizzare.
        else:
            # --- Ramo LPS ---
            # Conditional exchange: se A >= Qe il sotto-intervallo LPS
            # è nella parte bassa, altrimenti è già nella parte alta.
            if self.A >= Qe:
                self.C = (self.C + self.A) & 0xFFFFFFFF
                self.A = Qe
            # SWITCH: quando Qe è molto alto, il senso di MPS/LPS
            # si inverte perché il simbolo "raro" è diventato il più frequente.
            if SWITCH:
                self.mps[cx] ^= 1
            self.st[cx] = NLPS
            self._renorm_e()  # Dopo un LPS la rinormalizzazione è sempre necessaria

    #  Rinormalizzazione Encoder (T.81 §D.1.5 — RENORME)

    def _renorm_e(self):
        """Raddoppia l'intervallo A finché non torna nel range [0x8000, 0x10000).

        Ad ogni shift a sinistra di C un bit viene "consumato" da CT.
        Quando CT arriva a 0 il byte accumulato in C viene emesso con BYTEOUT.
        """
        while self.A < 0x8000:
            self.A = (self.A << 1) & 0xFFFF
            self.C = (self.C << 1) & 0xFFFFFFFF
            self.CT -= 1
            if self.CT == 0:
                self._byte_out()

    #  Emissione byte con gestione carry (T.81 §D.1.6 — BYTEOUT)

    def _byte_out(self):
        """Emette un byte dal registro C gestendo carry e byte stuffing.

        Il registro C accumula bit nella parte alta. I bit 19-26 contengono
        il byte candidato per l'emissione. Il bit 27+ indica un carry.

        Tre casi possibili:
          1. Carry (t > 0xFF): il carry si propaga nel byte B pendente.
             Tutti gli 0xFF accumulati nello stack (SC) diventano 0x00
             perché 0xFF + carry = 0x100 → byte = 0x00 con carry propagato.
          2. Byte 0xFF (temp == 0xFF): non possiamo ancora emetterlo perché
             un carry futuro potrebbe modificarlo. Lo accumuliamo in SC.
          3. Byte normale: possiamo emettere B e svuotare lo stack.
             Ogni 0xFF nello stack viene emesso con byte stuffing (0xFF 0x00)
             per evitare che il decoder lo scambi per un marker JPEG.

        ZC ottimizza l'output: i byte 0x00 consecutivi vengono contati
        e scritti solo quando necessario, evitando leading zeros inutili.
        """
        t = self.C >> 19
        temp = t & 0xFF

        if t > 0xFF:
            # ---- Carry rilevato ----
            if self.B >= 0:
                # Svuota eventuali 0x00 accumulati prima del byte pendente
                if self.ZC:
                    for _ in range(self.ZC):
                        self._out.append(0x00)
                    self.ZC = 0

                self._out.append(self.B + 1)

                # Se B+1 diventa 0xFF serve byte stuffing
                if self.B + 1 == 0xFF:
                    self._out.append(0x00)

            # Il carry trasforma tutti gli 0xFF pendenti in 0x00
            self.ZC += self.SC
            self.SC = 0

            self.B = temp

        elif temp == 0xFF:
            # ---- Byte 0xFF: accumulalo nello stack ----
            # Non possiamo emetterlo ora perché un carry futuro
            # potrebbe propagarsi attraverso di esso.
            self.SC += 1

        else:
            # ---- Byte normale: ora possiamo emettere tutto ----
            if self.B == 0:
                # Gli 0x00 vengono contati in ZC per evitare emissioni premature
                self.ZC += 1

            elif self.B >= 0:
                if self.ZC:
                    for _ in range(self.ZC):
                        self._out.append(0x00)
                    self.ZC = 0

                self._out.append(self.B)

            # Lo stack di 0xFF può ora essere emesso definitivamente
            # con byte stuffing: ogni 0xFF è seguito da 0x00 (T.81 §D.1.4)
            if self.SC:
                if self.ZC:
                    for _ in range(self.ZC):
                        self._out.append(0x00)
                    self.ZC = 0

                for _ in range(self.SC):
                    self._out.append(0xFF)
                    self._out.append(0x00)
                self.SC = 0

            self.B = temp

        # Indipendentemente dal ramo, puliamo i bit alti di C
        # e resettiamo il contatore a 8 bit per il prossimo byte.
        self.C &= 0x7FFFF
        self.CT = 8


    #  Finalizzazione (T.81 §D.1.8 — FLUSH)

    def flush(self) -> bytes:
        """Chiude il flusso di codifica emettendo gli ultimi byte.

        Sceglie un valore finale di C all'interno dell'intervallo [C, C+A)
        che massimizzi i trailing zero, così da minimizzare i byte di output.
        Poi gestisce l'eventuale carry finale e scrive solo i byte significativi.
        """
        # Arrotonda C al limite superiore dell'intervallo, azzerando i bit bassi
        temp = (self.A - 1 + self.C) & 0xFFFF0000

        if temp < self.C:
            # L'arrotondamento ha superato l'intervallo: rientra con +0x8000
            self.C = (temp + 0x8000) & 0xFFFFFFFF
        else:
            self.C = temp

        # Shifta C per i bit rimanenti nel contatore CT
        self.C = (self.C << self.CT) & 0xFFFFFFFF

        # --- Gestione carry finale ---
        # I bit alti di C (27-23) indicano se c'è stato un carry
        if self.C & 0xF8000000:
            # Carry presente: stessa logica di _byte_out ramo carry
            if self.B >= 0:
                if self.ZC:
                    for _ in range(self.ZC):
                        self._out.append(0x00)
                    self.ZC = 0

                self._out.append(self.B + 1)

                if self.B + 1 == 0xFF:
                    self._out.append(0x00)

            # Gli 0xFF pendenti diventano 0x00 per il carry
            self.ZC += self.SC
            self.SC = 0

        else:
            # Nessun carry: emetti B e lo stack come in _byte_out
            if self.B == 0:
                self.ZC += 1

            elif self.B >= 0:
                if self.ZC:
                    for _ in range(self.ZC):
                        self._out.append(0x00)
                    self.ZC = 0

                self._out.append(self.B)

            if self.SC:
                if self.ZC:
                    for _ in range(self.ZC):
                        self._out.append(0x00)
                    self.ZC = 0

                for _ in range(self.SC):
                    self._out.append(0xFF)
                    self._out.append(0x00)

        # --- Emissione degli ultimi byte significativi ---
        # Emettiamo solo i byte che contengono informazione utile,
        # evitando trailing 0x00 che il decoder non ha bisogno di leggere.
        if self.C & 0x7FFF800:
            if self.ZC:
                for _ in range(self.ZC):
                    self._out.append(0x00)
                self.ZC = 0

            b = (self.C >> 19) & 0xFF
            self._out.append(b)

            if b == 0xFF:
                self._out.append(0x00)

            # Secondo byte residuo, se significativo
            if self.C & 0x7F800:
                b = (self.C >> 11) & 0xFF
                self._out.append(b)

                if b == 0xFF:
                    self._out.append(0x00)

        return bytes(self._out)

    #  Decodifica binaria adattiva (T.81 Fig. D.14)

    def decode_bin(self, cx: int) -> int:
        """Decodifica una decisione binaria dal contesto cx.

        Procedura speculare a encode_bin: partiziona l'intervallo nello stesso
        modo e determina il simbolo in base alla posizione di C nell'intervallo.
        Il conditional exchange garantisce la coerenza con l'encoder.
        """
        state = self.st[cx]
        Qe, NMPS, NLPS, SWITCH = self._ST_TABLE[state]

        self.A -= Qe
        C_high = self.C >> 16
        if C_high < self.A:
            # C è nella sub-regione MPS
            if self.A < 0x8000:
                if self.A < Qe:
                    # Conditional exchange: A < Qe → è in realtà LPS
                    decision = self.mps[cx] ^ 1
                    if SWITCH:
                        self.mps[cx] ^= 1
                    self.st[cx] = NLPS
                else:
                    decision = self.mps[cx]
                    self.st[cx] = NMPS
                self._renorm_d()
            else:
                # A è ancora nel range: nessuna rinormalizzazione necessaria
                decision = self.mps[cx]
        else:
            # C è nella sub-regione LPS: riposiziona C nell'intervallo LPS
            self.C = (self.C - (self.A << 16)) & 0xFFFFFFFF
            if self.A < Qe:
                # Conditional exchange: A < Qe → è in realtà MPS
                self.A = Qe
                decision = self.mps[cx]
                self.st[cx] = NMPS
            else:
                self.A = Qe
                decision = self.mps[cx] ^ 1
                if SWITCH:
                    self.mps[cx] ^= 1
                self.st[cx] = NLPS
            self._renorm_d()
        return decision

    #  Codifica/Decodifica a probabilità fissa (T.81 §F.1.4.4.1)
    #  Usata per il segno dei coefficienti AC: la distribuzione dei segni
    #  è approssimativamente uniforme (p ≈ 0.5), quindi non serve adattarla.
    #  Si usa Qe = 0x5A1D (stato 0) senza aggiornamento di stato.

    def encode_bin_fixed(self, decision: int):
        """Codifica un bit con probabilità fissa 0.5 (senza adattamento)."""
        Qe = 0x5A1D
        self.A -= Qe
        if decision == 0:
            if self.A < 0x8000:
                if self.A < Qe:
                    self.C = (self.C + self.A) & 0xFFFFFFFF
                    self.A = Qe
                self._renorm_e()
        else:
            if self.A >= Qe:
                self.C = (self.C + self.A) & 0xFFFFFFFF
                self.A = Qe
            self._renorm_e()

    def decode_bin_fixed(self) -> int:
        """Decodifica un bit con probabilità fissa 0.5 (senza adattamento)."""
        Qe = 0x5A1D
        self.A -= Qe
        C_high = self.C >> 16

        if C_high < self.A:
            if self.A < 0x8000:
                if self.A < Qe:
                    decision = 1
                else:
                    decision = 0
                self._renorm_d()
            else:
                decision = 0
        else:
            self.C = (self.C - (self.A << 16)) & 0xFFFFFFFF
            if self.A < Qe:
                self.A = Qe
                decision = 0
            else:
                self.A = Qe
                decision = 1
            self._renorm_d()
        return decision

    #  Rinormalizzazione Decoder (T.81 §D.2.6 — RENORMD)

    def _renorm_d(self):
        """Raddoppia A e C finché A non torna nel range valido.

        Speculare a _renorm_e: legge nuovi bit dallo stream tramite
        BYTEIN quando CT si esaurisce.
        """
        while self.A < 0x8000:
            if self.CT == 0:
                self._byte_in()
            self.A = (self.A << 1) & 0xFFFF
            self.C = (self.C << 1) & 0xFFFFFFFF
            self.CT -= 1

    #  Ingresso byte Decoder (T.81 §D.2.5 — BYTEIN)

    def _byte_in(self):
        """Legge un byte dallo stream e lo carica nel registro C.

        Gestisce il byte stuffing JPEG: se il byte letto è 0xFF,
        il successivo deve essere 0x00 (stuff byte) e viene scartato.
        Eventuali 0xFF consecutivi sono fill bytes e vengono saltati.
        Un byte diverso da 0x00 dopo 0xFF indica un marker JPEG inatteso.
        """
        b = self._next_byte()

        if b == 0xFF:
            b2 = self._next_byte()

            # Salta eventuali fill bytes 0xFF consecutivi
            while b2 == 0xFF:
                b2 = self._next_byte()

            if b2 == 0x00:
                # Byte stuffing: il vero byte è 0xFF, lo 0x00 viene scartato
                b = 0xFF
            else:
                raise ValueError("Unexpected JPEG marker in QM payload")

        self.C = (self.C + (b << 8)) & 0xFFFFFFFF
        self.CT = 8


#  BINARIZER — Modello statistico per coefficienti DCT (T.81 Annex F)
#
#  Il Binarizer converte i coefficienti DCT quantizzati in una sequenza
#  di decisioni binarie, ciascuna associata a un contesto specifico.
#  Il QM-Coder codifica poi ogni decisione adattando la probabilità
#  tramite la sua macchina a stati.
#
#  I contesti sono organizzati per tipo di coefficiente:
#   - DC: differenza col blocco precedente (DPCM), con contesti
#     condizionati dalla differenza DC precedente (Da)
#   - AC: per ogni posizione zig-zag k (1-63), tre decisioni base:
#     EOB (fine blocco), zero/non-zero, e magnitudine
#
#  Mapping contesti (per canale, offset 0 per luma, 300 per chroma):
#   DC:  [0..3]×5 = 20 contesti base + [20..48] per categoria/magnitudine
#   AC:  3 contesti × 63 posizioni = 189 base + contesti di categoria


class Binarizer:
    def __init__(self, is_luma: bool = True):
        self.prev_dc = 0
        self.prev_dc_diff = 0
        self.DC_CTX_OFFSET = 0 if is_luma else 300
        self.AC_CTX_OFFSET = self.DC_CTX_OFFSET + 49

    def reset_dc(self):
        self.prev_dc = 0
        self.prev_dc_diff = 0

    #  Classificazione DC (T.81 Tab. F.1)

    def _dc_context(self, Da: int) -> int:
        """Seleziona il contesto base S0 in base alla differenza DC precedente (Da).

        Lo standard definisce 5 categorie di conditioning basate su Da
        e i parametri L, U (configurabili via marker DAC, qui U=2):
          - Da == 0         → contesto 0  (nessuna differenza)
          - 0 < Da ≤ U      → contesto 4  (piccola positiva)
          - Da > U           → contesto 12 (grande positiva)
          - -U ≤ Da < 0      → contesto 8  (piccola negativa)
          - Da < -U          → contesto 16 (grande negativa)

        I contesti sono distanziati di 4 per lasciare spazio a SS, SP, SN
        all'interno di ciascuna categoria.
        """
        if Da == 0:
            return 0

        if Da > 0:
            return 4 if Da <= 2 else 12

        return 8 if Da >= -2 else 16

    #  Codifica DC (T.81 §F.1.4.3, Fig. F.3 / F.8 / F.9)

    def binarize_dc(self, core: QMCoderCore, value: int):
        """Binarizza il coefficiente DC in una sequenza di decisioni.

        Struttura della codifica:
          1. Differenza DPCM = value - prev_dc
          2. Decisione zero/non-zero (contesto S0, condizionato da Da)
          3. Segno (contesto SS = S0 + 1)
          4. Categoria di magnitudine (contesti SP/SN → X1, X2, ...)
             codificata come albero binario: si emettono '1' finché
             v >> 1 ≠ 0, poi uno '0' terminale (Fig. F.8)
          5. Bit di rifinitura della magnitudine (contesti M, offset +14)
             specificano la posizione esatta all'interno della categoria (Fig. F.9)
        """
        diff = value - self.prev_dc
        self.prev_dc = value

        ctx_base = self.DC_CTX_OFFSET
        S0 = self._dc_context(self.prev_dc_diff)

        self.prev_dc_diff = diff

        # 1. Decisione zero/non-zero
        if diff == 0:
            core.encode_bin(ctx_base + S0, 0)
            return

        core.encode_bin(ctx_base + S0, 1)

        # 2. Segno: 0 = positivo, 1 = negativo
        sign = 1 if diff < 0 else 0
        core.encode_bin(ctx_base + S0 + 1, sign)

        # 3. Selezione contesto SP (positivo) o SN (negativo) per la magnitudine
        st = ctx_base + S0 + (3 if sign else 2)

        # 4. Categoria di magnitudine (Fig. F.8)
        #    v = |diff| - 1: la magnitudine meno 1, usata per la codifica esponenziale.
        #    m traccia la potenza di 2 della categoria corrente.
        v = abs(diff) - 1
        m = 0

        if v:
            # La magnitudine è ≥ 2 → serve almeno una categoria
            core.encode_bin(st, 1)

            m = 1
            v2 = v

            # Contesti X1, X2, ... per le categorie successive (a partire dall'indice 20)
            st = ctx_base + 20

            while True:
                v2 >>= 1
                if v2 == 0:
                    break

                core.encode_bin(st, 1)
                m <<= 1
                st += 1

        # Terminatore '0': segnala che la categoria è stata determinata
        core.encode_bin(st, 0)

        # 5. Bit di rifinitura (Fig. F.9)
        #    Specificano la posizione esatta del valore all'interno della categoria.
        #    I contesti M sono a offset +14 rispetto ai contesti X.
        st += 14

        m2 = m
        while True:
            m2 >>= 1
            if m2 == 0:
                break

            bit = 1 if (m2 & v) else 0
            core.encode_bin(st, bit)

    #  Decodifica DC (T.81 §F.2.4.3, Fig. F.18 / F.23 / F.24)

    def debinarize_dc(self, core: QMCoderCore) -> int:
        """Decodifica il coefficiente DC. Speculare a binarize_dc."""
        ctx_base = self.DC_CTX_OFFSET
        S0 = self._dc_context(self.prev_dc_diff)

        # 1. Zero/non-zero
        if core.decode_bin(ctx_base + S0) == 0:
            self.prev_dc_diff = 0
            return self.prev_dc

        # 2. Segno
        sign = core.decode_bin(ctx_base + S0 + 1)

        # 3. SP / SN
        st = ctx_base + S0 + (3 if sign else 2)

        # 4. Categoria di magnitudine (Fig. F.23)
        #    m accumula la potenza di 2 della categoria decodificata
        m = core.decode_bin(st)

        if m:
            st = ctx_base + 20

            while core.decode_bin(st):
                m <<= 1

                if m >= 0x8000:
                    raise ValueError("DC magnitude overflow")

                st += 1

        # 5. Bit di rifinitura (Fig. F.24)
        #    v parte da m e accumula i bit di rifinitura tramite OR
        v = m

        st += 14

        m2 = m
        while True:
            m2 >>= 1
            if m2 == 0:
                break

            if core.decode_bin(st):
                v |= m2

        # Il valore finale è v + 1 (la magnitudine era stata decrementata di 1)
        v += 1

        if sign:
            v = -v

        self.prev_dc_diff = v
        self.prev_dc += v

        return self.prev_dc

    #  Codifica AC (T.81 §F.1.4.4, Fig. F.4 / F.8 / F.9)

    def binarize_ac(self, core: QMCoderCore, k: int, is_eob: bool, value: int = 0):
        """Binarizza un coefficiente AC alla posizione zig-zag k.

        Per ogni posizione k (1-63) ci sono tre contesti base (Tab. F.5):
          - SE = 3*(k-1)    : decisione EOB (fine del blocco)
          - S0 = SE + 1     : decisione zero/non-zero
          - S1 = SE + 2     : prima decisione di magnitudine

        La magnitudine usa la stessa struttura di Fig. F.8/F.9 del DC,
        con una differenza importante: le prime DUE decisioni di categoria
        usano lo STESSO contesto S1, e solo dalla terza si passa ai
        contesti X1/X2 (base 189 per k ≤ Kx, 217 per k > Kx, dove Kx=5).

        Il segno viene codificato a probabilità fissa 0.5 (encode_bin_fixed)
        perché la distribuzione dei segni AC è approssimativamente uniforme.
        """
        ctx_base = self.AC_CTX_OFFSET
        k_idx = k - 1

        SE = 3 * k_idx
        S0 = SE + 1
        S1 = SE + 2

        # 1. Decisione EOB: da qui in poi il blocco è tutto zeri
        if is_eob:
            core.encode_bin(ctx_base + SE, 1)
            return

        core.encode_bin(ctx_base + SE, 0)

        # 2. Decisione zero/non-zero
        if value == 0:
            core.encode_bin(ctx_base + S0, 0)
            return

        core.encode_bin(ctx_base + S0, 1)

        # 3. Segno a probabilità fissa 0.5 (T.81 §F.1.4.4.1)
        sign = 1 if value < 0 else 0
        core.encode_bin_fixed(sign)

        # 4. Categoria di magnitudine (Fig. F.8)
        v = abs(value) - 1
        m = 0

        st = ctx_base + S1

        if v:
            # Prima decisione di categoria: usa contesto S1
            core.encode_bin(st, 1)

            m = 1
            v2 = v

            # Seconda decisione di categoria: RIUSA lo stesso contesto S1
            # (peculiarità degli AC rispetto ai DC — T.81 Tab. F.5)
            v2 >>= 1

            if v2:
                core.encode_bin(st, 1)

                m <<= 1

                # Dalla terza decisione in poi si usano i contesti X1/X2
                # Kx = 5: soglia tra i due gruppi di contesti (T.81 §F.1.4.4.2)
                st = ctx_base + (189 if k <= 5 else 217)

                while True:
                    v2 >>= 1

                    if v2 == 0:
                        break

                    core.encode_bin(st, 1)
                    m <<= 1
                    st += 1

        # Terminatore '0' della categoria
        core.encode_bin(st, 0)

        # 5. Bit di rifinitura (Fig. F.9), offset +14
        st += 14

        m2 = m
        while True:
            m2 >>= 1

            if m2 == 0:
                break

            bit = 1 if (m2 & v) else 0
            core.encode_bin(st, bit)

    #  Decodifica AC (T.81 §F.2.4.4, Fig. F.19 / F.23 / F.24)

    def debinarize_ac(self, core: QMCoderCore, k: int) -> Tuple[bool, int]:
        """Decodifica un coefficiente AC alla posizione k. Speculare a binarize_ac."""
        ctx_base = self.AC_CTX_OFFSET
        k_idx = k - 1

        SE = 3 * k_idx
        S0 = SE + 1
        S1 = SE + 2

        # 1. EOB
        if core.decode_bin(ctx_base + SE):
            return True, 0

        # 2. Zero/non-zero
        if core.decode_bin(ctx_base + S0) == 0:
            return False, 0

        # 3. Segno a probabilità fissa
        sign = core.decode_bin_fixed()

        st = ctx_base + S1

        # 4. Categoria di magnitudine (Fig. F.23)
        m = core.decode_bin(st)

        if m:
            # Seconda decisione: stesso contesto S1
            if core.decode_bin(st):
                m <<= 1

                # Dalla terza in poi: contesti X1/X2
                st = ctx_base + (189 if k <= 5 else 217)

                while core.decode_bin(st):
                    m <<= 1

                    if m >= 0x8000:
                        raise ValueError("AC magnitude overflow")

                    st += 1

        # 5. Bit di rifinitura (Fig. F.24)
        v = m

        st += 14

        m2 = m
        while True:
            m2 >>= 1

            if m2 == 0:
                break

            if core.decode_bin(st):
                v |= m2

        v += 1

        if sign:
            v = -v

        return False, v


import struct


#  QMCoder — Interfaccia di alto livello per la pipeline JPEG
#
#  Combina il core aritmetico (QMCoderCore) con il modello statistico
#  (Binarizer) per codificare/decodificare i blocchi 8×8 di coefficienti
#  DCT quantizzati e riordinati in zig-zag.


class QMCoder(EntropyEncoder, EntropyDecoder):
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.name = "QM-Coder"

    def encode(self, blocks: List[np.ndarray], is_luma: bool = True) -> bytes:
        """Codifica una lista di blocchi 8×8 (in ordine zig-zag) in un flusso di byte.

        Per ogni blocco:
          1. Codifica il coefficiente DC (differenza DPCM col blocco precedente)
          2. Codifica i coefficienti AC da k=1 a k=63
          3. Emette un simbolo EOB se il blocco termina prima di k=63

        Il flusso risultante è preceduto da 4 byte (big-endian) che indicano
        la lunghezza del payload, così il decoder può isolare il segmento
        del canale corrente (Y, Cb o Cr).
        """
        core = QMCoderCore(debug=self.debug)
        core.init_enc()
        binarizer = Binarizer(is_luma=is_luma)

        for block in blocks:
            # DC: il primo coefficiente del blocco
            binarizer.binarize_dc(core, int(block[0]))

            # AC: trova l'ultimo coefficiente non-zero per sapere dove emettere EOB
            last_nonzero = 63
            while last_nonzero > 0 and block[last_nonzero] == 0:
                last_nonzero -= 1

            for k in range(1, last_nonzero + 1):
                binarizer.binarize_ac(core, k, False, int(block[k]))

            # EOB: segnala che il resto del blocco è tutto zeri
            if last_nonzero < 63:
                binarizer.binarize_ac(core, last_nonzero + 1, True)

        stream = core.flush()
        return struct.pack(">I", len(stream)) + stream

    def decode(
        self, stream: bytes, num_blocks: int, is_luma: bool = True
    ) -> Tuple[List[np.ndarray], int]:
        """Decodifica il flusso di byte in una lista di blocchi 8×8.

        Legge i primi 4 byte per determinare la lunghezza del payload,
        poi decodifica num_blocks blocchi usando il processo inverso dell'encoder.
        Restituisce i blocchi decodificati e il numero di byte consumati.
        """
        data_len = struct.unpack(">I", stream[:4])[0]
        ch_bytes = stream[4 : 4 + data_len]

        core = QMCoderCore(debug=self.debug)
        core.init_dec(ch_bytes)
        binarizer = Binarizer(is_luma=is_luma)

        blocks_out = []
        for _ in range(num_blocks):
            block = np.zeros(64, dtype=np.int32)

            block[0] = binarizer.debinarize_dc(core)

            k = 1
            while k < 64:
                is_eob, ac_val = binarizer.debinarize_ac(core, k)
                if is_eob:
                    break
                block[k] = ac_val
                k += 1

            blocks_out.append(block)

        return blocks_out, 4 + data_len
