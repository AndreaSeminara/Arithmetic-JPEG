import struct
import numpy as np
from typing import List, Tuple
from .base import EntropyEncoder, EntropyDecoder
from utils.tables import QM_ST_TABLE

class QMCoderCore:
    _ST_TABLE = QM_ST_TABLE
    
    def __init__(self, debug=False):
        self.debug = debug
        self.st = bytearray(600)
        self.mps = bytearray(600)
        
        # Registri Encoder
        self.A = 0
        self.C = 0
        self.CT = 0
        self.B = 0
        self.SC = 0
        self._out = []
        self._first_byte = True

        # Registri Decoder
        self._in = b''
        self._in_idx = 0

    def init_enc(self):
        # A è la larghezza dell'intervallo in fixed-point Q16: 0x10000 rappresenta 1.0
        self.A = 0x10000
        self.C = 0
        # CT conta quanti bit possono ancora essere scritti in C prima di emettere un byte.
        # Vale 11 all'inizio e non 8 perché i primi 3 bit di C fungono da buffer di guardia (T.81 §D.1.5)
        self.CT = 11
        # B è il byte tenuto in sospeso finché non sappiamo se arriverà un carry
        self.B = 0
        # SC conta i byte 0xFF rimandati per gestire l'underflow
        self.SC = 0
        self._out = []
        self._first_byte = True

    def init_dec(self, stream: bytes):
        self._in = stream
        self._in_idx = 0
        # I primi byte vengono caricati nel registro C per inizializzare il decoder.
        # La procedura è speculare all'init dell'encoder (T.81 §D.3.1)
        self.B = self._next_byte()
        self.C = (self.B << 16)
        self._byte_in()
        self.C = (self.C << 8) & 0xFFFFFFFF
        self.CT -= 8
        self.A = 0x10000

    def _next_byte(self) -> int:
        if self._in_idx < len(self._in):
            b = self._in[self._in_idx]
            self._in_idx += 1
            return b
        return 0

    def encode_bin(self, cx: int, decision: int):
        # MPS = Most Probable Symbol (il simbolo con probabilità maggiore),
        # LPS = il simbolo meno probabile. Lo stato cx descrive la probabilità corrente
        # e si aggiorna automaticamente dopo ogni simbolo (T.81 tab. D.3).
        state = self.st[cx]
        Qe, NMPS, NLPS, SWITCH = self._ST_TABLE[state]
        
        self.A -= Qe
        if decision == self.mps[cx]:
            if self.A < 0x8000:
                if self.A < Qe:
                    self.C = (self.C + self.A) & 0xFFFFFFFF
                    self.A = Qe
                self.st[cx] = NMPS
                self._renorm_e()
        else:
            if self.A >= Qe:
                self.C = (self.C + self.A) & 0xFFFFFFFF
                self.A = Qe
            if SWITCH:
                self.mps[cx] ^= 1
            self.st[cx] = NLPS
            self._renorm_e()

    def _renorm_e(self):
        while self.A < 0x8000:
            self.A = (self.A << 1) & 0xFFFF
            self.C = (self.C << 1) & 0xFFFFFFFF
            self.CT -= 1
            if self.CT == 0:
                self._byte_out()

    def _byte_out(self):
        # Estrae il byte da emettere dalla parte alta del registro C.
        # T.81 §D.1.4: se il byte è 0xFF dobbiamo inserire subito uno 0x00 dopo (byte stuffing),
        # altrimenti il decoder lo scambierebbe per un marker JPEG.
        # SC tiene i 0xFF in sospeso finché non sappiamo se arriverà un carry.
        t = self.C >> 19
        temp = t & 0xFF
        
        if t > 0xFF:
            self.B += 1
            if self.B == 0xFF:
                self._out.append(0xFF)
                self._out.append(0x00)
                for _ in range(self.SC):
                    self._out.append(0x00)
            else:
                self._out.append(self.B)
                for _ in range(self.SC):
                    self._out.append(0x00)
            self.B = temp
            self.SC = 0
        else:
            if temp == 0xFF:
                self.SC += 1
            else:
                if self._first_byte:
                    self._first_byte = False
                else:
                    self._out.append(self.B)
                for _ in range(self.SC):
                    self._out.append(0xFF)
                    self._out.append(0x00)
                self.B = temp
                self.SC = 0
                
        self.C &= 0x7FFFF
        self.CT = 8

    def flush(self) -> bytes:
        temp = (self.C + self.A - 1) & 0xFFFF0000
        if temp < self.C:
            self.C = (temp + 0x8000) & 0xFFFFFFFF
        else:
            self.C = temp
        self.C = (self.C << self.CT) & 0xFFFFFFFF
        
        # Svuotiamo i byte rimasti nel registro C con due BYTEOUT,
        # più una terza chiamata per garantire al decoder abbastanza bit da leggere senza eccedere il segmento.
        self._byte_out()
        self._byte_out()
        self._byte_out()
        
        return bytes(self._out)

    def decode_bin(self, cx: int) -> int:
        state = self.st[cx]
        Qe, NMPS, NLPS, SWITCH = self._ST_TABLE[state]
        
        self.A -= Qe
        C_high = self.C >> 16
        if C_high < self.A:
            if self.A < 0x8000:
                if self.A < Qe:
                    decision = self.mps[cx] ^ 1
                    if SWITCH: self.mps[cx] ^= 1
                    self.st[cx] = NLPS
                else:
                    decision = self.mps[cx]
                    self.st[cx] = NMPS
                self._renorm_d()
            else:
                decision = self.mps[cx]
        else:
            self.C = (self.C - (self.A << 16)) & 0xFFFFFFFF
            if self.A < Qe:
                self.A = Qe
                decision = self.mps[cx]
                self.st[cx] = NMPS
            else:
                self.A = Qe
                decision = self.mps[cx] ^ 1
                if SWITCH: self.mps[cx] ^= 1
                self.st[cx] = NLPS
            self._renorm_d()
        return decision

    def encode_bin_fixed(self, decision: int):
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


    def _renorm_d(self):
        while self.A < 0x8000:
            if self.CT == 0:
                self._byte_in()
            self.A = (self.A << 1) & 0xFFFF
            self.C = (self.C << 1) & 0xFFFFFFFF
            self.CT -= 1

    def _byte_in(self):
        b = self._next_byte()
        if b == 0xFF:
            b2 = self._next_byte()
            if b2 == 0x00:
                self.C = (self.C + 0xFF00) & 0xFFFFFFFF
                self.CT = 8
            else:
                self.C = (self.C + 0xFF00) & 0xFFFFFFFF
                self.CT = 8
        else:
            self.C = (self.C + (b << 8)) & 0xFFFFFFFF
            self.CT = 8


class Binarizer:
    def __init__(self, is_luma: bool = True):
        self.prev_dc = 0
        self.prev_dc_diff = 0
        # 49 states for DC, 245 for AC
        self.DC_CTX_OFFSET = 0 if is_luma else 300
        self.AC_CTX_OFFSET = self.DC_CTX_OFFSET + 49

    def reset_dc(self):
        self.prev_dc = 0
        self.prev_dc_diff = 0

    def _dc_context(self, Da: int) -> int:
        if Da == 0:
            return 0
        abs_Da = abs(Da)
        if abs_Da <= 2:
            return 4 if Da > 0 else 8
        else:
            return 12 if Da > 0 else 16

    def binarize_dc(self, core: QMCoderCore, value: int):
        diff = value - self.prev_dc
        self.prev_dc = value
        
        ctx_base = self.DC_CTX_OFFSET
        S0 = self._dc_context(self.prev_dc_diff)
        self.prev_dc_diff = diff
        
        # se la differenza è zero, si codifica con il simbolo 0 e si ritorna
        if diff == 0:
            core.encode_bin(ctx_base + S0, 0)
            return

        core.encode_bin(ctx_base + S0, 1)

        # codifichiamo il segno: 1 = negativo, 0 = positivo
        sign = 1 if diff < 0 else 0
        SS = S0 + 1
        core.encode_bin(ctx_base + SS, sign)

        # Sz è la magnitudine scalata di -1 per la codifica unary che segue
        Sz = abs(diff) - 1

        # seleziona il contesto di magnitudine in base al segno della differenza
        S = (S0 + 3) if diff < 0 else (S0 + 2)
        if Sz < 1:
            core.encode_bin(ctx_base + S, 0)
            return

        core.encode_bin(ctx_base + S, 1)

        # I contesti 20-34 codificano la categoria di magnitudine in unario (T.81 tab. D.3)
        cat = 1
        X = 20
        while True:
            if Sz < (1 << cat):
                core.encode_bin(ctx_base + X, 0)
                break
            core.encode_bin(ctx_base + X, 1)
            cat += 1
            X += 1
            if X > 34:
                break

        # bit di rifinitura della magnitudine: offset di 14 rispetto ai contesti di categoria (T.81 tab. D.3)
        if cat > 0:
            M = X + 14
            for i in range(cat - 1, -1, -1):
                b = (Sz >> i) & 1
                core.encode_bin(ctx_base + M, b)

    def debinarize_dc(self, core: QMCoderCore) -> int:
        # Speculare a binarize_dc: leggiamo i simboli nello stesso ordine in cui sono stati scritti
        ctx_base = self.DC_CTX_OFFSET
        S0 = self._dc_context(self.prev_dc_diff)

        if core.decode_bin(ctx_base + S0) == 0:
            self.prev_dc_diff = 0
            return self.prev_dc

        sign = core.decode_bin(ctx_base + S0 + 1)
        S = (S0 + 3) if sign == 1 else (S0 + 2)

        if core.decode_bin(ctx_base + S) == 0:
            Sz = 0
        else:
            # stessi contesti 20-34 dell'encoder per la categoria (T.81 tab. D.3)
            cat = 1
            X = 20
            while True:
                if core.decode_bin(ctx_base + X) == 0:
                    break
                cat += 1
                X += 1
                if X > 34:
                    break

            Sz_bits = 0
            M = X + 14
            for i in range(cat - 1, -1, -1):
                b = core.decode_bin(ctx_base + M)
                Sz_bits = (Sz_bits << 1) | b

            Sz = Sz_bits

        diff = -(Sz + 1) if sign == 1 else (Sz + 1)
        self.prev_dc_diff = diff
        self.prev_dc += diff
        return self.prev_dc

    def binarize_ac(self, core: QMCoderCore, k: int, is_eob: bool, value: int = 0):
        ctx_base = self.AC_CTX_OFFSET
        k_idx = k - 1
        # Ogni posizione AC k ha tre contesti: EOB, zero/nonzero, e prima categoria (T.81 tab. D.4)
        SE = 3 * k_idx
        S0 = 3 * k_idx + 1
        S1 = 3 * k_idx + 2

        # EOB: da qui in poi il blocco è tutto zeri
        if is_eob:
            core.encode_bin(ctx_base + SE, 1)
            return
        core.encode_bin(ctx_base + SE, 0)

        # coefficiente nullo a questa posizione
        if value == 0:
            core.encode_bin(ctx_base + S0, 0)
            return
        core.encode_bin(ctx_base + S0, 1)

        # il segno lo trattiamo a probabilità fissa 0.5: non varia abbastanza per addestrare il modello
        sign = 1 if value < 0 else 0
        core.encode_bin_fixed(sign)

        Sz = abs(value) - 1

        if Sz < 1:
            core.encode_bin(ctx_base + S1, 0)
            return
        core.encode_bin(ctx_base + S1, 1)

        # I contesti di categoria cambiano in base alla posizione k:
        # coefficienti 1-5 usano l'indice base 189, gli altri 217 (T.81 tab. D.4)
        cat = 1
        X = 189 if (k + 1) <= 5 else 217
        while True:
            if Sz < (1 << cat):
                core.encode_bin(ctx_base + X, 0)
                break
            core.encode_bin(ctx_base + X, 1)
            cat += 1
            X += 1
            if (X - 189) > 13 and (X - 217) > 13:
                break

        # bit di rifinitura, offset di 14 dopo i contesti di categoria (T.81 tab. D.4)
        if cat > 0:
            M = X + 14
            for i in range(cat - 1, -1, -1):
                b = (Sz >> i) & 1
                core.encode_bin(ctx_base + M, b)

    def debinarize_ac(self, core: QMCoderCore, k: int) -> Tuple[bool, int]:
        # Speculare a binarize_ac: leggiamo EOB, zero, segno e magnitudine nello stesso ordine
        ctx_base = self.AC_CTX_OFFSET
        k_idx = k - 1
        SE = 3 * k_idx
        S0 = 3 * k_idx + 1
        S1 = 3 * k_idx + 2

        if core.decode_bin(ctx_base + SE) == 1:
            return True, 0  # EOB: fine del blocco

        if core.decode_bin(ctx_base + S0) == 0:
            return False, 0  # coefficiente zero

        sign = core.decode_bin_fixed()

        if core.decode_bin(ctx_base + S1) == 0:
            Sz = 0
        else:
            # stessi contesti dell'encoder per la categoria (T.81 tab. D.4)
            cat = 1
            X = 189 if (k + 1) <= 5 else 217
            while True:
                if core.decode_bin(ctx_base + X) == 0:
                    break
                cat += 1
                X += 1
                if (X - 189) > 13 and (X - 217) > 13:
                    break

            Sz_bits = 0
            M = X + 14
            for i in range(cat - 1, -1, -1):
                b = core.decode_bin(ctx_base + M)
                Sz_bits = (Sz_bits << 1) | b

            Sz = Sz_bits

        value = -(Sz + 1) if sign == 1 else (Sz + 1)
        return False, value



import struct
class QMCoder(EntropyEncoder, EntropyDecoder):
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.name = "QM-Coder"

    def encode(self, blocks: List[np.ndarray], is_luma: bool = True) -> bytes:
        core = QMCoderCore(debug=self.debug)
        core.init_enc()
        binarizer = Binarizer(is_luma=is_luma)
        
        for block in blocks:
            binarizer.binarize_dc(core, int(block[0]))
            
            last_nonzero = 63
            while last_nonzero > 0 and block[last_nonzero] == 0:
                last_nonzero -= 1
                
            for k in range(1, last_nonzero + 1):
                binarizer.binarize_ac(core, k, False, int(block[k]))
                
            if last_nonzero < 63:
                binarizer.binarize_ac(core, last_nonzero + 1, True)
                
        stream = core.flush()
        # Prepende 4 byte di lunghezza al payload così il decoder sa quanti byte leggere
        return struct.pack(">I", len(stream)) + stream

    def decode(self, stream: bytes, num_blocks: int, is_luma: bool = True) -> Tuple[List[np.ndarray], int]:
        # I primi 4 byte contengono la lunghezza del payload di questo canale
        data_len = struct.unpack(">I", stream[:4])[0]
        ch_bytes = stream[4: 4 + data_len]
        
        core = QMCoderCore(debug=self.debug)
        core.init_dec(ch_bytes)
        binarizer = Binarizer(is_luma=is_luma)
        
        blocks_out = []
        for _ in range(num_blocks):
            block = np.zeros(64, dtype=np.int32)
            
            try:
                block[0] = binarizer.debinarize_dc(core)
                
                k = 1
                while k < 64:
                    is_eob, ac_val = binarizer.debinarize_ac(core, k)
                    if is_eob:
                        break
                    block[k] = ac_val
                    k += 1
            except Exception:
                break
                
            blocks_out.append(block)
            
        # I byte consumati totali sono la lunghezza del payload + 4 byte di header
        return blocks_out, 4 + data_len
