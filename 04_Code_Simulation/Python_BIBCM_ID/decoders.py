"""
decoders.py  –  Soft-Input Soft-Output Decoders for BIBCM-ID
=============================================================
Converted from  SisoDecode.c  and  dualdec.m

Provides:
  1. SISODecoder   – BCJR (MAP) decoder for convolutional codes
                     (pure Python re-implementation of the C MEX file)
  2. DualDecoder   – ML-based soft decoder for block codes (Hamming)

Both are designed to be callable from TF via tf.py_function for
integration into end-to-end trainable pipelines.
"""

import numpy as np
from typing import Tuple


# ====================================================================
# 1.  Trellis construction helpers
# ====================================================================

def _parity_counter(symbol: int, length: int) -> int:
    """
    Exact port of CML's parity_counter().
    Returns 1 if symbol has odd parity, 0 if even.
    """
    temp_parity = 0
    for _ in range(length):
        temp_parity ^= (symbol & 1)
        symbol >>= 1
    return temp_parity


def _rsc_enc_bit(input_bit: int, state_in: int,
                 g: list, KK: int, nn: int):
    """
    Exact port of CML's rsc_enc_bit().
    Returns (output_symbol, next_state).
    """
    # systematic output
    out = input_bit

    # determine feedback bit
    a_k = input_bit ^ _parity_counter(g[0] & state_in, KK)

    # create a word made up of state and feedback bit
    state = (a_k << (KK - 1)) ^ state_in

    # AND the word with the generators
    for i in range(1, nn):
        out = (out << 1) + _parity_counter(state & g[i], KK)

    # shift the state to make the new state
    next_state = state >> 1
    return out, next_state


def _build_trellis_rsc(g_encoder_octal: list, K: int, n: int):
    """
    Build RSC trellis transition tables.
    Exact port of CML's rsc_transit().

    Parameters
    ----------
    g_encoder_octal : list of n octal generator polynomials
                      (g[0] = feedback, g[1..] = feedforward)
    K : constraint length
    n : number of outputs per input bit

    Returns
    -------
    out0, state0 : output/next-state when input = 0, shape (num_states,)
    out1, state1 : output/next-state when input = 1, shape (num_states,)
    """
    m = K - 1
    num_states = 1 << m

    out0 = np.zeros(num_states, dtype=int)
    out1 = np.zeros(num_states, dtype=int)
    state0 = np.zeros(num_states, dtype=int)
    state1 = np.zeros(num_states, dtype=int)

    for s in range(num_states):
        out0[s], state0[s] = _rsc_enc_bit(0, s, g_encoder_octal, K, n)
        out1[s], state1[s] = _rsc_enc_bit(1, s, g_encoder_octal, K, n)

    return out0, state0, out1, state1


def _build_trellis_rsc_from_binary(g_binary: np.ndarray):
    """
    Build trellis from the binary generator matrix g (shape n x K).
    Row 0 = feedback polynomial, rows 1..n-1 = feedforward.

    Converts to integer representation exactly as SisoDecode.c does,
    then calls _build_trellis_rsc.
    """
    n, K = g_binary.shape
    g_octal = []
    for i in range(n):
        val = 0
        for j in range(K):
            if g_binary[i, j] != 0:
                val += (1 << (K - 1 - j))
        g_octal.append(val)
    return _build_trellis_rsc(g_octal, K, n)


# ====================================================================
# 2.  BCJR (Log-MAP) SISO Decoder
# ====================================================================

from numba import njit
import os

@njit(nogil=True, cache=True)
def _bcjr_maxstar(a: float, b: float, dec_type: int) -> float:
    if dec_type == 1:
        return max(a, b)
    else:
        if a > b:
            return a + np.log1p(np.exp(b - a))
        else:
            return b + np.log1p(np.exp(a - b))

@njit(nogil=True, cache=True)
def _bcjr_decode_njit(input_u: np.ndarray, input_c: np.ndarray,
                      out0: np.ndarray, out1: np.ndarray,
                      state0: np.ndarray, state1: np.ndarray,
                      data_len: int, code_len: int, total_len: int,
                      n: int, num_states: int, dec_type: int):
    NINF = -1e10

    # --- Branch metrics ---
    gamma0 = np.zeros((total_len, num_states))
    gamma1 = np.zeros((total_len, num_states))

    for t in range(total_len):
        for s in range(num_states):
            bm0 = 0.0
            o0 = out0[s]
            for i in range(n):
                bit = (o0 >> i) & 1
                if bit == 1:
                    bm0 += input_c[t * n + (n - 1 - i)]
            
            bm1 = 0.0
            o1 = out1[s]
            for i in range(n):
                bit = (o1 >> i) & 1
                if bit == 1:
                    bm1 += input_c[t * n + (n - 1 - i)]
            
            if t < data_len:
                bm1 += input_u[t]
            
            gamma0[t, s] = bm0
            gamma1[t, s] = bm1

    # --- Forward recursion (alpha) ---
    alpha = np.full((total_len + 1, num_states), NINF)
    alpha[0, 0] = 0.0

    for t in range(total_len):
        for s in range(num_states):
            ns0 = state0[s]
            val0 = alpha[t, s] + gamma0[t, s]
            
            ns1 = state1[s]
            val1 = alpha[t, s] + gamma1[t, s]
            
            alpha[t + 1, ns0] = _bcjr_maxstar(alpha[t + 1, ns0], val0, dec_type)
            alpha[t + 1, ns1] = _bcjr_maxstar(alpha[t + 1, ns1], val1, dec_type)

    # --- Backward recursion (beta) ---
    beta = np.full((total_len + 1, num_states), NINF)
    beta[total_len, 0] = 0.0

    for t in range(total_len - 1, -1, -1):
        for s in range(num_states):
            ns0 = state0[s]
            ns1 = state1[s]
            
            val0 = beta[t + 1, ns0] + gamma0[t, s]
            val1 = beta[t + 1, ns1] + gamma1[t, s]
            
            beta[t, s] = _bcjr_maxstar(_bcjr_maxstar(beta[t, s], val0, dec_type), val1, dec_type)

    # --- Compute output LLRs ---
    output_u = np.zeros(data_len)
    output_c = np.zeros(code_len)

    for t in range(total_len):
        llr_1 = NINF
        llr_0 = NINF
        
        for s in range(num_states):
            ns0 = state0[s]
            ns1 = state1[s]
            
            val0 = alpha[t, s] + gamma0[t, s] + beta[t + 1, ns0]
            val1 = alpha[t, s] + gamma1[t, s] + beta[t + 1, ns1]
            
            llr_0 = _bcjr_maxstar(llr_0, val0, dec_type)
            llr_1 = _bcjr_maxstar(llr_1, val1, dec_type)
            
        if t < data_len:
            output_u[t] = llr_1 - llr_0
            
        for j in range(n):
            if t * n + j < code_len:
                llr_bit1 = NINF
                llr_bit0 = NINF
                i = n - 1 - j
                
                for s in range(num_states):
                    ns0 = state0[s]
                    ns1 = state1[s]
                    
                    bit0_out = (out0[s] >> i) & 1
                    val0 = alpha[t, s] + gamma0[t, s] + beta[t + 1, ns0]
                    if bit0_out == 1:
                        llr_bit1 = _bcjr_maxstar(llr_bit1, val0, dec_type)
                    else:
                        llr_bit0 = _bcjr_maxstar(llr_bit0, val0, dec_type)
                        
                    bit1_out = (out1[s] >> i) & 1
                    val1 = alpha[t, s] + gamma1[t, s] + beta[t + 1, ns1]
                    if bit1_out == 1:
                        llr_bit1 = _bcjr_maxstar(llr_bit1, val1, dec_type)
                    else:
                        llr_bit0 = _bcjr_maxstar(llr_bit0, val1, dec_type)
                        
                output_c[t * n + j] = llr_bit1 - llr_bit0
                
    return output_u, output_c

class SISODecoder:
    """
    Soft-Input Soft-Output decoder using the BCJR algorithm.

    Implements the Log-MAP algorithm with the Jacobian correction
    (log-sum-exp) as the default decoding method.

    This is a pure-Python port of the C MEX function SisoDecode.c
    from the Iterative Solutions Coded Modulation Library.
    """

    # Decoder type constants
    LINEAR_APPROX = 0
    MAX_LOG_MAP = 1
    LOG_MAP = 4

    def __init__(self, g: np.ndarray, code_type: int = 0,
                 dec_type: int = 0):
        """
        Parameters
        ----------
        g         : generator matrix, shape (n, K), binary
        code_type : 0 = RSC (default), 1 = NSC
        dec_type  : 0 = linear approx to log-MAP (default)
                    1 = max-log-MAP
                    4 = log-MAP with correction
        """
        self.g = g
        self.n, self.K = g.shape
        self.m = self.K - 1
        self.num_states = 1 << self.m
        self.code_type = code_type
        self.dec_type = dec_type

        # Build trellis
        self.out0, self.state0, self.out1, self.state1 = \
            _build_trellis_rsc_from_binary(g)



    def decode(self, input_u: np.ndarray,
               input_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SISO decode.

        Parameters
        ----------
        input_u : a-priori LLR for data bits, shape (data_len,)
        input_c : a-priori LLR for code bits, shape (code_len,)
                  code_len = n * (data_len + m)

        Returns
        -------
        output_u : extrinsic LLR for data bits, shape (data_len,)
        output_c : extrinsic LLR for code bits, shape (code_len,)
        """
        data_len = len(input_u)
        code_len = len(input_c)
        total_len = data_len + self.m  # includes tail

        return _bcjr_decode_njit(
            input_u, input_c,
            self.out0, self.out1, self.state0, self.state1,
            data_len, code_len, total_len,
            self.n, self.num_states, self.dec_type
        )


# ====================================================================
# 3.  Dual Decoder for Block Codes  — port of dualdec.m
# ====================================================================

@njit(nogil=True, cache=True)
def _dual_decode_frame_njit(llr: np.ndarray, C: np.ndarray, E: np.ndarray, n_code: int):
    num_cw = len(llr) // n_code
    output = np.zeros_like(llr)
    
    for i in range(num_cw):
        start = i * n_code
        end = (i + 1) * n_code
        x = llr[start:end]
        
        # p = (1 - exp(x)) / (1 + exp(x)) = -tanh(x/2)
        # Clip for numerical stability
        x_clipped = np.copy(x)
        for j in range(n_code):
            if x_clipped[j] > 30:
                x_clipped[j] = 30
            elif x_clipped[j] < -30:
                x_clipped[j] = -30
                
        p = (1.0 - np.exp(x_clipped)) / (1.0 + np.exp(x_clipped))
        
        v2 = 1.0 + 1e-12
        for cw_idx in range(C.shape[0]):
            prod_val = 1.0
            has_nonzero = False
            for j in range(n_code):
                if C[cw_idx, j] != 0:
                    prod_val *= p[j]
                    has_nonzero = True
            if has_nonzero:
                v2 += prod_val
                
        vv = np.zeros(n_code)
        for jj in range(n_code):
            v1 = p[jj]
            for cw_idx in range(C.shape[0]):
                prod_val = 1.0
                has_nonzero = False
                for j in range(n_code):
                    mod_cw = (C[cw_idx, j] + E[jj, j]) % 2
                    if mod_cw != 0:
                        prod_val *= p[j]
                        has_nonzero = True
                if has_nonzero:
                    v1 += prod_val
            
            v2_plus_v1 = v2 + v1
            v2_minus_v1 = v2 - v1
            
            if abs(v2_minus_v1) < 1e-30:
                if v2_minus_v1 != 0:
                    if v2_minus_v1 > 0:
                        v2_minus_v1 = 1e-30
                    else:
                        v2_minus_v1 = -1e-30
                else:
                    v2_minus_v1 = 1e-30
            if abs(v2_plus_v1) < 1e-30:
                v2_plus_v1 = 1e-30
                
            vv[jj] = (np.log(abs(v2**2 - v1**2) + 1e-30) - 
                      np.log(abs((v2 + v1)**2) + 1e-30))
            
        output[start:end] = vv
        
    return output

class DualDecoder:
    """
    Dual-domain soft decoder for linear block codes.

    Uses the dual code (via the parity-check matrix H) to compute
    extrinsic LLRs. Exact implementation of dualdec.m.
    """

    def __init__(self, H: np.ndarray):
        """
        Parameters
        ----------
        H : parity-check matrix, shape (n_checks, n_code)
        """
        self.H = H
        self.n_checks, self.n_code = H.shape

        # Pre-compute all dual codewords (non-zero syndrome patterns)
        self._precompute_dual_codewords()

    def _precompute_dual_codewords(self):
        """Build all codewords of the dual code from H."""
        n_checks = self.n_checks
        # All non-zero binary vectors of length n_checks
        num_patterns = 2 ** n_checks - 1
        patterns = np.array(
            [[(i >> (n_checks - 1 - j)) & 1 for j in range(n_checks)]
             for i in range(1, num_patterns + 1)],
            dtype=int
        )  # shape (2^n_checks - 1, n_checks)

        # Dual codewords: C = patterns @ H mod 2
        self.C = (patterns @ self.H) % 2  # shape (num_patterns, n_code)

        self.E = np.eye(self.n_code, dtype=np.int64)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Soft decode one codeword block.
        """
        return self.decode_frame(x)

    def decode_frame(self, llr: np.ndarray) -> np.ndarray:
        """
        Decode a full frame of multiple codewords.
        """
        return _dual_decode_frame_njit(llr, self.C, self.E, self.n_code)
