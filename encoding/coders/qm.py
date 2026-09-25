import struct
import numpy as np
from typing import List, Tuple
from .base import EntropyEncoder, EntropyDecoder
from utils.tables import QM_ST_TABLE

#  QM-CODER
#
#  Il QM-Coder è un codificatore aritmetico binario adattivo usato in JPEG.
#  Ogni decisione binaria (0 o 1) viene codificata sfruttando una stima
#  della probabilità che si aggiorna automaticamente blocco dopo blocco.
#
#  I registri principali sono:
#   - A  : ampiezza dell'intervallo di codifica. 0x10000 rappresenta l'intervallo pieno (= 1.0 scalato a intero)
#   - C  : registro di codifica che accumula i bit man mano che li produciamo
#   - CT : quanti bit possiamo ancora inserire in C prima di dover emettere un byte
#   - B  : l'ultimo byte "in sospeso", tenuto da parte per gestire eventuali carry
#   - SC : quanti byte 0xFF consecutivi stiamo aspettando di confermare
#   - ZC : quanti byte 0x00 consecutivi sono stati rimandati (evitiamo di scriverli subito)


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
        # B = -1 indica "nessun byte in sospeso"
        self.B = -1
        self.ZC = 0
        self.SC = 0
        self._out = []

        # Registri Decoder
        self._in = b""
        self._in_idx = 0

    #  Inizializzazione

    def init_enc(self):
        self.A = 0x10000  # Intervallo pieno
        self.C = 0
        # CT parte da 11, i 3 bit più alti di C fungono da buffer di guardia per evitare carry
        self.CT = 11
        # B = -1 indica che non ci sono byte in sospeso
        self.B = -1
        self.ZC = 0
        self.SC = 0
        self._out = []

    def init_dec(self, stream: bytes):
        """Carica i primi byte nel registro C per avviare la decodifica"""
        self._in = stream
        self._in_idx = 0
        self.B = self._next_byte()
        self.C = self.B << 16
        self._byte_in()
        self.C = (self.C << 8) & 0xFFFFFFFF
        self.CT -= 8
        self.A = 0x10000

    def _next_byte(self) -> int:
        """Legge il prossimo byte dallo stream. Restituisce 0 se esaurito"""
        if self._in_idx < len(self._in):
            b = self._in[self._in_idx]
            self._in_idx += 1
            return b
        return 0

    #  Codifica binaria adattiva

    def encode_bin(self, cx: int, decision: int):
        """Codifica una decisione binaria nel contesto cx.

        L'idea di base è semplice: l'intervallo corrente [0, A) viene diviso
        in due parti in base alla probabilità stimata:
          - MPS (Most Probable Symbol): occupa A - Qe
          - LPS (Least Probable Symbol): occupa Qe

        Quando A - Qe scende sotto Qe avviene il "conditional exchange":
        i due sotto-intervalli vengono scambiati, in modo che l'MPS occupi
        sempre la fetta più grande.
        """
        state = self.st[cx]
        Qe, NMPS, NLPS, SWITCH = self._ST_TABLE[state]

        self.A -= Qe
        if decision == self.mps[cx]:
            # Ramo MPS
            if self.A < 0x8000:
                # L'intervallo è diventato troppo piccolo e va rinormalizzato.
                # Prima però controlliamo se serve il conditional exchange:
                # se A < Qe l'MPS è più stretto dell'LPS, quindi li scambiamo.
                if self.A < Qe:
                    self.C = (self.C + self.A) & 0xFFFFFFFF
                    self.A = Qe
                self.st[cx] = NMPS
                self._renorm_e()
            # Se A è ancora >= 0x8000 siamo a posto, nessuna azione necessaria
        else:
            # Ramo LPS: spostiamo C verso la parte alta dell'intervallo
            # in modo da selezionare la sotto-regione LPS.
            if self.A >= Qe:
                self.C = (self.C + self.A) & 0xFFFFFFFF
                self.A = Qe
            # SWITCH: se Qe è molto alto, il simbolo "raro" sta diventando
            # il più comune, quindi invertiamo la polarità MPS/LPS.
            if SWITCH:
                self.mps[cx] ^= 1
            self.st[cx] = NLPS
            self._renorm_e()  # dopo un LPS la rinormalizzazione è sempre necessaria

    #  Rinormalizzazione Encoder

    def _renorm_e(self):
        """Raddoppia l'intervallo A finché non rientra nel range [0x8000, 0x10000).

        Ad ogni shift a sinistra di C, CT si decrementa di 1.
        Quando CT raggiunge 0, il byte accumulato viene emesso tramite BYTEOUT.
        """
        while self.A < 0x8000:
            self.A = (self.A << 1) & 0xFFFF
            self.C = (self.C << 1) & 0xFFFFFFFF
            self.CT -= 1
            if self.CT == 0:
                self._byte_out()

    #  Emissione byte con gestione carry

    def _byte_out(self):
        """Emette un byte dal registro C gestendo carry e byte stuffing.

        Il registro C accumula bit nella parte alta. I bit 19-26 contengono
        il byte candidato per l'emissione. Il bit 27+ indica un carry.

        Tre casi possibili:
          1. Carry (t > 0xFF): il carry si propaga nel byte B pendente.
             Tutti gli 0xFF in SC diventano 0x00
             perché 0xFF + carry = 0x100 → byte = 0x00 con carry propagato.
          2. Byte 0xFF (temp == 0xFF): non viene emesso subito perché
             un carry futuro potrebbe modificarlo. Viene accumulato in SC.
          3. Byte normale: B viene emesso e lo stack SC viene svuotato.
             Ogni 0xFF in SC è seguito da 0x00 (byte stuffing)
             per evitare che il decoder lo scambi per un marker JPEG.

        ZC ottimizza l'output: i byte 0x00 consecutivi vengono contati
        e scritti solo quando necessario, evitando zeri iniziali non significativi.
        """
        t = self.C >> 19
        temp = t & 0xFF

        if t > 0xFF:
            # Carry rilevato
            if self.B >= 0:
                # Emette gli eventuali 0x00 accumulati prima del byte pendente
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
            # Byte 0xFF: accumulalo nello stack
            # Non possiamo emetterlo ora perché un carry futuro
            # potrebbe propagarsi attraverso di esso
            self.SC += 1

        else:
            # Byte normale: ora possiamo emettere tutto
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
            # con byte stuffing: ogni 0xFF è seguito da 0x00
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

    #  Finalizzazione

    def flush(self) -> bytes:
        """Chiude il flusso di codifica emettendo gli ultimi byte.

        Sceglie un valore finale di C all'interno dell'intervallo [C, C+A)
        con il maggior numero possibile di zeri finali, minimizzando i byte di output.
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

        #  Gestione carry finale
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

        #  Emissione degli ultimi byte significativi
        # Vengono scritti solo i byte con informazione utile,
        # omettendo gli 0x00 finali che il decoder non deve leggere.
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

    #  Decodifica binaria adattiva

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

    #  Codifica/Decodifica a probabilità fissa
    #  Usata per il segno dei coefficienti AC: la distribuzione dei segni
    #  è approssimativamente uniforme (p ≈ 0.5), quindi non serve adattarla.
    #  Si usa Qe = 0x5A1D (stato 0) senza aggiornamento di stato.

    def encode_bin_fixed(self, decision: int):
        """Codifica un bit con probabilità fissa 0.5 (senza adattamento)"""
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
        """Decodifica un bit con probabilità fissa 0.5 (senza adattamento)"""
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

    #  Rinormalizzazione Decoder

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

    #  Ingresso byte Decoder

    def _byte_in(self):
        """Legge un byte dallo stream e lo carica nel registro C.

        Gestisce il byte stuffing JPEG: se il byte letto è 0xFF,
        il successivo deve essere 0x00 (stuff byte) e viene ignorato.
        Eventuali 0xFF consecutivi sono byte di riempimento e vengono scartati.
        Un byte diverso da 0x00 dopo 0xFF indica un marker JPEG nel payload.
        """
        b = self._next_byte()

        if b == 0xFF:
            b2 = self._next_byte()

            # Scarta eventuali byte di riempimento 0xFF consecutivi
            while b2 == 0xFF:
                b2 = self._next_byte()

            if b2 == 0x00:
                # Byte stuffing: il vero byte è 0xFF, lo 0x00 viene scartato
                b = 0xFF
            else:
                raise ValueError("Unexpected JPEG marker in QM payload")

        self.C = (self.C + (b << 8)) & 0xFFFFFFFF
        self.CT = 8


#  BINARIZER — Modello statistico per coefficienti DCT
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

    #  Classificazione DC

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

    #  Codifica DC

    def binarize_dc(self, core: QMCoderCore, value: int):
        """Binarizza il coefficiente DC in una sequenza di decisioni.

        Il DC viene codificato come differenza rispetto al blocco precedente (DPCM),
        e la differenza a sua volta viene decomposta in più decisioni binarie:
          1. diff = value - prev_dc
          2. È zero o no? (contesto S0, che dipende dalla diff precedente)
          3. Segno della diff (contesto SS = S0 + 1)
          4. Categoria della magnitudine: quanti bit servono per rappresentarla.
             Emettiamo una serie di '1' fino a trovare la categoria giusta,
             poi uno '0' come terminatore.
          5. Bit di rifinitura: precisano il valore esatto all'interno della categoria.
             I contesti per questi bit sono a offset +14 rispetto alla categoria.
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

        # 4. Categoria di magnitudine
        #    Lavoriamo su v = |diff| - 1 (scaliamo di 1 perché la magnitudine minima è 1).
        #    m tiene traccia della potenza di 2 che corrisponde alla categoria.
        v = abs(diff) - 1
        m = 0

        if v:
            # La magnitudine è ≥ 2 → serve almeno una categoria
            core.encode_bin(st, 1)

            m = 1
            v2 = v

            # Per le categorie superiori usiamo contesti dedicati a partire dall'offset 20
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

        # 5. Bit di rifinitura
        #    Ora che conosciamo la categoria, emettiamo i bit che specificano
        #    il valore esatto. I contesti per questa fase sono a +14 dall'ultimo.
        st += 14

        m2 = m
        while True:
            m2 >>= 1
            if m2 == 0:
                break

            bit = 1 if (m2 & v) else 0
            core.encode_bin(st, bit)

    #  Decodifica DC

    def debinarize_dc(self, core: QMCoderCore) -> int:
        """Decodifica il coefficiente DC. Speculare a binarize_dc"""
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

        # 4. Categoria di magnitudine
        #    Leggiamo le decisioni finché non troviamo il terminatore '0'.
        #    m ci dice in quale categoria siamo (potenza di 2 corrispondente).
        m = core.decode_bin(st)

        if m:
            st = ctx_base + 20

            while core.decode_bin(st):
                m <<= 1

                if m >= 0x8000:
                    raise ValueError("DC magnitude overflow")

                st += 1

        # 5. Bit di rifinitura
        #    Partiamo da m e affiniamo il valore bit a bit tramite OR.
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

    #  Codifica AC

    def binarize_ac(self, core: QMCoderCore, k: int, is_eob: bool, value: int = 0):
        """Binarizza un coefficiente AC alla posizione zig-zag k.

        Per ogni posizione k (1-63) usiamo tre contesti base:
          - SE = 3*(k-1)    : è questo il simbolo EOB? (fine blocco)
          - S0 = SE + 1     : il coefficiente è zero o no?
          - S1 = SE + 2     : prima decisione sulla grandezza del valore

        La codifica della magnitudine è simile al DC, con una piccola differenza:
        le prime due decisioni di categoria condividono lo stesso contesto S1.
        Dalla terza in poi si usano contesti separati (uno per k <= 5,
        uno per k > 5, perché i coefficienti alle basse frequenze hanno
        distribuzioni diverse da quelli ad alta frequenza).

        Il segno viene codificato a probabilità fissa 0.5 perché i segni
        dei coefficienti AC sono praticamente equidistribuiti.
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

        # 3. Segno a probabilità fissa 0.5
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

            # Seconda decisione di categoria: riusiamo lo stesso contesto S1
            # (a differenza del DC, qui le prime due decisioni condividono il contesto)
            v2 >>= 1

            if v2:
                core.encode_bin(st, 1)

                m <<= 1

                # Dalla terza decisione in poi usiamo un contesto diverso
                # a seconda della posizione k: basse frequenze (k <= 5) o alte (k > 5)
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

        # 5. Bit di rifinitura (contesti a +14 dall'ultimo usato)
        st += 14

        m2 = m
        while True:
            m2 >>= 1

            if m2 == 0:
                break

            bit = 1 if (m2 & v) else 0
            core.encode_bin(st, bit)

    #  Decodifica AC

    def debinarize_ac(self, core: QMCoderCore, k: int) -> Tuple[bool, int]:
        """Decodifica un coefficiente AC alla posizione k. Speculare a binarize_ac"""
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

        # 4. Categoria di magnitudine
        m = core.decode_bin(st)

        if m:
            # Seconda decisione: stesso contesto S1 (come in encoding)
            if core.decode_bin(st):
                m <<= 1

                # Dalla terza in poi: contesto dipende dalla posizione k
                st = ctx_base + (189 if k <= 5 else 217)

                while core.decode_bin(st):
                    m <<= 1

                    if m >= 0x8000:
                        raise ValueError("AC magnitude overflow")

                    st += 1

        # 5. Bit di rifinitura
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
