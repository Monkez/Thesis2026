"""
encoders.py  –  Channel Encoders for BIBCM-ID
==============================================
Converted from  rsc_encode.m,  encode_bit.m

Provides:
  1. RSCEncoder          – rate-1/2 recursive systematic convolutional encoder
  2. HammingEncoder      – extended Hamming (8,4) encoder
  3. RSCEncoderLayer     – Keras layer wrapping RSCEncoder (via tf.py_function)
  4. HammingEncoderLayer – Keras layer wrapping HammingEncoder
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ====================================================================
# 1.  encode_bit  (NumPy)  – port of encode_bit.m
# ====================================================================

def encode_bit(g: np.ndarray, input_bit: int,
               state: np.ndarray):
    """
    Encode a single input bit through the convolutional encoder.

    Parameters
    ----------
    g     : generator matrix, shape (n, k)
    input_bit : the bit to encode (0 or 1)
    state : current encoder state, shape (m,)

    Returns
    -------
    output : encoded output bits, shape (n,)
    new_state : updated state, shape (m,)
    """
    n, k = g.shape
    m = k - 1
    output = np.zeros(n, dtype=int)

    for i in range(n):
        output[i] = g[i, 0] * input_bit
        for j in range(1, k):
            output[i] = output[i] ^ (g[i, j] * state[j - 1])

    new_state = np.zeros(m, dtype=int)
    new_state[0] = input_bit
    new_state[1:] = state[:m - 1]
    return output, new_state


# ====================================================================
# 2.  RSC Encoder  (NumPy)  – port of rsc_encode.m
# ====================================================================

class RSCEncoder:
    """
    Rate-1/n Recursive Systematic Convolutional (RSC) encoder.

    Exactly reproduces the MATLAB rsc_encode.m function.
    """

    def __init__(self, g: np.ndarray):
        """
        Parameters
        ----------
        g : generator matrix, shape (n, K) where K = constraint length
            Row 0 is the feedback polynomial.
        """
        self.g = g
        self.n, self.K = g.shape
        self.m = self.K - 1  # memory

    def encode(self, u: np.ndarray) -> np.ndarray:
        """
        Encode information bits.

        Parameters
        ----------
        u : 1-D array of information bits {0, 1}, length data_len

        Returns
        -------
        v : 1-D array of coded bits, length n * data_len
        """
        data_len = len(u)
        state = np.zeros(self.m, dtype=int)
        v = np.zeros(self.n * data_len, dtype=int)

        for i in range(data_len):
            d_k = int(u[i])
            # Feedback: a_k = g[0,:] · [d_k, state]' mod 2
            a_k = int(np.dot(self.g[0, :],
                             np.concatenate(([d_k], state))) % 2)
            output_bits, state = encode_bit(self.g, a_k, state)
            # Systematic: first output = input bit
            output_bits[0] = d_k
            v[self.n * i: self.n * (i + 1)] = output_bits

        return v

    @property
    def code_rate(self) -> float:
        return 1.0 / self.n


# ====================================================================
# 3.  Extended Hamming Encoder  (NumPy)
# ====================================================================

class HammingEncoder:
    """
    Extended Hamming code encoder.
    Builds G and H matrices exactly as in sim_ham_bicmid.m.
    """

    def __init__(self, m: int = 3):
        """
        Parameters
        ----------
        m : Hamming parameter.
            Standard Hamming: (2^m - 1, 2^m - 1 - m)
            Extended Hamming: (2^m, 2^m - 1 - m) — adds an overall parity bit
        """
        self.m_param = m
        self._build_matrices()

    def _build_matrices(self):
        """Build G and H matrices following the MATLAB code."""
        m = self.m_param

        # P matrix: binary representations of 1..2^m-1, transposed
        P = np.array([[(i >> (m - 1 - j)) & 1
                       for j in range(m)]
                      for i in range(1, 2 ** m)],
                     dtype=int).T  # shape (m, 2^m - 1)

        # Remove columns corresponding to powers of 2
        powers_of_2_cols = [2 ** p - 1 for p in range(m)]  # 0-indexed column indices
        keep_cols = [c for c in range(2 ** m - 1) if c not in powers_of_2_cols]
        P = P[:, keep_cols]  # shape (m, 2^m - 1 - m)

        # Extended Hamming: add overall parity row
        parity_row = (np.sum(P, axis=0) + 1) % 2
        P = np.vstack([P, parity_row])  # shape (m+1, 2^m - 1 - m)

        k_info = P.shape[1]        # 2^m - 1 - m  (information bits)
        n_code = k_info + m + 1     # 2^m  (codeword length)

        self.G = np.hstack([np.eye(k_info, dtype=int), P.T])  # (k_info, n_code)
        self.H = np.hstack([P, np.eye(m + 1, dtype=int)])      # (m+1, n_code)
        self.k_info = k_info
        self.n_code = n_code
        self.n_checks = m + 1

    def encode(self, u: np.ndarray) -> np.ndarray:
        """
        Encode one block of k_info bits into n_code coded bits.

        Parameters
        ----------
        u : 1-D array of length k_info, bits {0, 1}

        Returns
        -------
        v : 1-D array of length n_code
        """
        return (u @ self.G) % 2

    def encode_frame(self, u: np.ndarray) -> np.ndarray:
        """
        Encode a full frame of multiple codewords.

        Parameters
        ----------
        u : 1-D array, length must be divisible by k_info

        Returns
        -------
        v : 1-D coded array, length = len(u) / k_info * n_code
        """
        num_cw = len(u) // self.k_info
        v = np.zeros(num_cw * self.n_code, dtype=int)
        for i in range(num_cw):
            info = u[i * self.k_info: (i + 1) * self.k_info]
            v[i * self.n_code: (i + 1) * self.n_code] = self.encode(info)
        return v

    @property
    def code_rate(self) -> float:
        return self.k_info / self.n_code


# ====================================================================
# 4.  TensorFlow / Keras Encoder Layers
# ====================================================================

class RSCEncoderLayer(layers.Layer):
    """
    Keras layer wrapping the RSC encoder via tf.py_function.

    This allows the encoder to be part of a Keras model graph while
    keeping the sequential logic in NumPy.  For end-to-end training,
    replace this with a neural encoder.

    Input:  (batch, info_len)  float32 {0,1}
    Output: (batch, info_len * n)  float32 {0,1}
    """

    def __init__(self, g: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.g = g
        self.encoder = RSCEncoder(g)
        self.n = g.shape[0]

    def call(self, bits):
        def _encode_batch(bits_np):
            batch = bits_np.numpy()
            results = []
            for b in batch:
                results.append(self.encoder.encode(b.astype(int)).astype(np.float32))
            return np.array(results, dtype=np.float32)

        output = tf.py_function(
            _encode_batch, [bits],
            tf.float32
        )
        # Set shape
        batch_size = tf.shape(bits)[0]
        info_len = tf.shape(bits)[1]
        output = tf.reshape(output, (batch_size, info_len * self.n))
        return output


class HammingEncoderLayer(layers.Layer):
    """
    Keras layer wrapping the Hamming encoder.

    Input:  (batch, num_info_bits)
    Output: (batch, num_coded_bits)
    """

    def __init__(self, m: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.ham = HammingEncoder(m)
        self.G_tf = tf.constant(self.ham.G.astype(np.float32))

    def call(self, bits):
        # bits: (B, num_info_bits)
        # Reshape to (B, num_cw, k_info), encode, reshape back
        B = tf.shape(bits)[0]
        k = self.ham.k_info
        n = self.ham.n_code
        num_cw = tf.shape(bits)[1] // k

        bits_r = tf.reshape(bits, (B, num_cw, k))
        # Matrix multiply with G mod 2
        coded = tf.cast(
            tf.math.mod(
                tf.cast(tf.matmul(bits_r, self.G_tf), tf.int32), 2
            ), tf.float32
        )  # (B, num_cw, n)
        return tf.reshape(coded, (B, num_cw * n))
