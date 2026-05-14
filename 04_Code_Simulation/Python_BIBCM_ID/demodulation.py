"""
demodulation.py  –  Soft Demodulation (TF/Keras layers)
=======================================================
Converted from  softdemod_16qam.m  and  softdem.m

Provides:
  1. softdemod_qam()  –  compute channel metrics  (NumPy)
  2. soft_demapper()   –  compute extrinsic LLRs   (NumPy)
  3. SoftDemodLayer    –  Keras layer for channel metrics
  4. SoftDemapperLayer –  Keras layer for LLR computation with a-priori info
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ====================================================================
# 1.  Channel metric computation  (NumPy)   — softdemod_16qam.m
# ====================================================================

def softdemod_qam(rx: np.ndarray, N0: float,
                  S: np.ndarray) -> np.ndarray:
    """
    Compute per-symbol, per-constellation-point log-likelihood metrics.

    Parameters
    ----------
    rx : complex array, shape (num_symbols,)
    N0 : noise power spectral density (scalar)
    S  : complex constellation, shape (M,)

    Returns
    -------
    llr_n : real array, shape (M, num_symbols)
            llr_n[k, t] = -|rx[t] - S[k]|^2 / N0
    """
    M = len(S)
    frame_len = len(rx)
    # Vectorised: (M, frame_len)
    diff = rx[np.newaxis, :] - S[:, np.newaxis]  # (M, frame_len)
    llr_n = -(diff * np.conj(diff)).real / N0
    return llr_n


# ====================================================================
# 2.  Soft demapper with a-priori information  (NumPy)  — softdem.m
# ====================================================================

from numba import njit
import os

@njit(nogil=True, cache=True)
def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))

@njit(nogil=True, cache=True)
def soft_demapper(llr: np.ndarray,
                  La: np.ndarray,
                  sym_matrix: np.ndarray) -> np.ndarray:
    """
    Compute extrinsic bit LLRs from channel metrics and a-priori LLRs.

    Uses the log-sum-exp (Jacobian logarithm) for numerical stability.

    Parameters
    ----------
    llr        : channel metrics, shape (M, num_symbols)  — from softdemod_qam
    La         : a-priori LLR for each bit, shape (num_channel_bits,)
    sym_matrix : binary symbol-to-bit matrix, shape (M, bps)

    Returns
    -------
    Lc : extrinsic LLRs, shape (num_channel_bits,)
    """
    M, num_sym = llr.shape
    bps = sym_matrix.shape[1]
    num_bits = num_sym * bps
    Lc = np.zeros(num_bits)
    INFTY = 1e5

    for n in range(num_sym):
        num_arr = np.full(bps, -INFTY)
        den_arr = np.full(bps, -INFTY)

        for i in range(M):
            # Channel metric for this symbol at this time slot
            metric = llr[i, n]  # MATLAB: llr(n*symnum + i) → llr[i, n]

            # Add a-priori contributions
            for k_bit in range(bps):
                if sym_matrix[i, k_bit] == 1:
                    metric += La[n * bps + k_bit]

            # Distribute to numerator (bit=1) or denominator (bit=0)
            for k_bit in range(bps):
                # Extrinsic metric: subtract a-priori for this bit
                if sym_matrix[i, k_bit] == 1:
                    delta1 = num_arr[k_bit]
                    delta2 = metric - La[n * bps + k_bit]
                    num_arr[k_bit] = _log_sum_exp(delta1, delta2)
                else:
                    delta1 = den_arr[k_bit]
                    delta2 = metric
                    den_arr[k_bit] = _log_sum_exp(delta1, delta2)

        for k_bit in range(bps):
            Lc[n * bps + k_bit] = num_arr[k_bit] - den_arr[k_bit]

    return Lc


# ====================================================================
# 3.  TensorFlow / Keras: Soft Demodulation Layer
# ====================================================================

class SoftDemodLayer(layers.Layer):
    """
    Keras layer: compute channel metrics for received IQ symbols.

    Input:  (batch, num_symbols, 2)  – received [I, Q]
    Params: N0 (noise variance), passed via constructor or call-time
    Output: (batch, M, num_symbols)  – log-likelihood per constellation point

    The constellation S is stored as a non-trainable constant; replace
    with a trainable lookup for neural demodulation.
    """

    def __init__(self, S_complex: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.M = len(S_complex)
        # Store constellation as (M, 2)
        S_iq = np.stack([S_complex.real, S_complex.imag],
                        axis=-1).astype(np.float32)
        self.S_iq = tf.constant(S_iq, dtype=tf.float32)  # (M, 2)

    def call(self, inputs):
        """
        inputs: list [rx_iq, N0]
            rx_iq : (batch, num_sym, 2)
            N0    : (batch, 1) or scalar
        Returns: (batch, M, num_sym)
        """
        rx_iq, N0 = inputs
        # rx_iq: (B, T, 2) → expand for broadcasting: (B, 1, T, 2)
        rx_exp = tf.expand_dims(rx_iq, axis=1)
        # S_iq: (M, 2) → (1, M, 1, 2)
        S_exp = tf.reshape(self.S_iq, (1, self.M, 1, 2))
        diff = rx_exp - S_exp  # (B, M, T, 2)
        dist_sq = tf.reduce_sum(diff ** 2, axis=-1)  # (B, M, T)

        N0_exp = tf.reshape(N0, (-1, 1, 1))
        return -dist_sq / N0_exp  # (B, M, T)


class SoftDemapperLayer(layers.Layer):
    """
    Keras layer: computes extrinsic bit LLRs from channel metrics
    and a-priori LLRs using the log-sum-exp algorithm.

    Inputs:
        llr     : (batch, M, num_sym)  – channel metrics
        La      : (batch, num_channel_bits) – a-priori LLRs
    Output:
        Lc      : (batch, num_channel_bits) – extrinsic LLRs

    sym_matrix is a non-trainable constant.

    NOTE: This layer is implemented in a differentiable manner using
    tf.reduce_logsumexp so gradients can flow for end-to-end training.
    """

    def __init__(self, sym_matrix: np.ndarray, **kwargs):
        """
        sym_matrix: (M, bps) int array — binary symbol labels
        """
        super().__init__(**kwargs)
        M, bps = sym_matrix.shape
        self.M = M
        self.bps = bps
        # Store as float for TF ops
        self.sym_matrix = tf.constant(
            sym_matrix.astype(np.float32), dtype=tf.float32)  # (M, bps)

    def call(self, inputs):
        """
        inputs: [llr, La]
            llr : (B, M, T)
            La  : (B, T*bps)
        Returns: Lc (B, T*bps)
        """
        llr, La = inputs
        B = tf.shape(llr)[0]
        T = tf.shape(llr)[2]
        bps = self.bps

        # Reshape La to (B, T, bps)
        La_reshaped = tf.reshape(La, (B, T, bps))  # (B, T, bps)

        # Compute symbol-level a-priori: sum of La for bits that are 1
        # sym_matrix: (M, bps), La_reshaped: (B, T, bps)
        # → apriori_sym: (B, T, M)
        sym_mat = tf.reshape(self.sym_matrix, (1, 1, self.M, bps))
        La_exp = tf.expand_dims(La_reshaped, axis=2)  # (B, T, 1, bps)
        apriori_sym = tf.reduce_sum(sym_mat * La_exp, axis=-1)  # (B, T, M)

        # Total metric: channel + a-priori
        llr_t = tf.transpose(llr, perm=[0, 2, 1])  # (B, T, M)
        metric = llr_t + apriori_sym  # (B, T, M)

        # For each bit position k, compute:
        #   num_k = logsumexp over symbols where bit k = 1 of (metric - La_k)
        #   den_k = logsumexp over symbols where bit k = 0 of metric
        Lc_list = []
        for k in range(bps):
            mask_1 = self.sym_matrix[:, k]  # (M,) — 0 or 1
            mask_0 = 1.0 - mask_1

            # Extrinsic metric for bit k: subtract La for bit k
            La_k = La_reshaped[:, :, k:k+1]  # (B, T, 1)
            metric_ext = metric - La_k * tf.reshape(mask_1, (1, 1, self.M))

            # logsumexp with masking
            large_neg = -1e10
            num_k = tf.reduce_logsumexp(
                metric_ext + tf.math.log(
                    tf.maximum(tf.reshape(mask_1, (1, 1, self.M)), 1e-30)
                ) + large_neg * tf.reshape(mask_0, (1, 1, self.M)),
                axis=-1)  # (B, T)

            # For denominator, use metric without subtracting La_k
            # but only on symbols where bit k = 0
            den_k = tf.reduce_logsumexp(
                metric + tf.math.log(
                    tf.maximum(tf.reshape(mask_0, (1, 1, self.M)), 1e-30)
                ) + large_neg * tf.reshape(mask_1, (1, 1, self.M)),
                axis=-1)  # (B, T)

            Lc_list.append(num_k - den_k)  # (B, T)

        # Stack and interleave: Lc_list[k] has shape (B, T)
        # Want output (B, T*bps) with ordering [t0_b0, t0_b1, ..., t0_b(bps-1), t1_b0, ...]
        Lc_stacked = tf.stack(Lc_list, axis=-1)  # (B, T, bps)
        Lc = tf.reshape(Lc_stacked, (B, T * bps))
        return Lc
