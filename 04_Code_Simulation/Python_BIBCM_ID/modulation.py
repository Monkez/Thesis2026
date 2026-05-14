"""
modulation.py  –  QAM Constellation & Modulation (TF/Keras layers)
==================================================================
Converted from  modulate16qam.m

Provides both a pure-NumPy function for classic simulation and a
TensorFlow/Keras layer that can be dropped into a deep-learning
autoencoder pipeline.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ====================================================================
# 1.  Constellation generation  (NumPy)
# ====================================================================

def qam_constellation(M: int = 16) -> np.ndarray:
    """
    Standard square-QAM constellation with unit average energy.

    Parameters
    ----------
    M : constellation size (must be a perfect square power of 2)

    Returns
    -------
    points : complex array of shape (M,), sorted in natural order.
    """
    k = int(np.sqrt(M))
    assert k * k == M, f"M={M} must be a perfect square for square QAM"
    coords = np.arange(k) - (k - 1) / 2.0
    # Real = column, Imag = row  (matches MATLAB modem.qammod natural order)
    points = np.array([complex(r, i) for i in reversed(coords) for r in coords])
    # Normalise to unit average energy
    Es = np.mean(np.abs(points) ** 2)
    return points / np.sqrt(Es)


def qam16_constellation_matlab() -> np.ndarray:
    """
    Reproduce the exact 16-QAM constellation order produced by
    MATLAB's  modem.qammod(16)  which uses Gray mapping internally.
    Points are in natural integer order 0..15 (not Gray-reordered).
    """
    # Standard square-QAM grid matching MATLAB's modem.qammod(16)
    # MATLAB orders: real outer product of [-3,-1,1,3] with itself
    coords = np.array([-3, -1, 1, 3], dtype=float)
    points = np.array([complex(r, i) for i in coords for r in coords])
    return points


def apply_mapping(qam_points: np.ndarray,
                  maprule: list,
                  Es: float = 10.0) -> np.ndarray:
    """
    Apply a mapping rule and normalise the constellation.

    Parameters
    ----------
    qam_points : complex array of shape (M,) – raw QAM points
    maprule    : list of length M, 1-indexed mapping (MATLAB convention)
    Es         : normalisation energy

    Returns
    -------
    S : complex array of shape (M,), mapped & normalised constellation
    """
    maprule_0idx = np.array(maprule, dtype=int) - 1  # convert to 0-indexed
    S = qam_points[maprule_0idx] / np.sqrt(Es)
    return S


def build_constellation(M: int = 16,
                        maprule: list = None,
                        Es: float = 10.0) -> np.ndarray:
    """
    Convenience: build the full mapped & normalised constellation.
    """
    if maprule is None:
        from .config import ModulationConfig
        cfg = ModulationConfig(M=M)
        maprule = cfg.maprule
        Es = cfg.Es

    qam_points = qam16_constellation_matlab()
    return apply_mapping(qam_points, maprule, Es)


def symbol_bit_matrix(M: int = 16) -> np.ndarray:
    """
    Binary representation of symbol indices 0..M-1 (MSB first).
    Shape: (M, bps).  Equivalent to MATLAB  de2bi((0:M-1), bps, 'left-msb').
    """
    bps = int(np.log2(M))
    return np.array([[(i >> (bps - 1 - j)) & 1 for j in range(bps)]
                     for i in range(M)], dtype=int)


# ====================================================================
# 2.  Modulation function  (NumPy)  –  direct port of modulate16qam.m
# ====================================================================

def modulate_bits(bits: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Map a binary sequence to complex symbols using constellation S.

    Parameters
    ----------
    bits : 1-D array of 0/1, length must be divisible by bps
    S    : complex constellation array of shape (M,)

    Returns
    -------
    symbols : complex array of shape (num_symbols,)
    """
    M = len(S)
    bps = int(np.log2(M))
    num_sym = len(bits) // bps
    assert len(bits) == num_sym * bps

    bipower = 2 ** np.arange(bps - 1, -1, -1)  # [2^(bps-1), ..., 1]
    bits_matrix = bits.reshape(bps, num_sym, order='F')  # column-major like MATLAB
    indices = bipower @ bits_matrix  # shape (num_sym,)
    return S[indices]


# ====================================================================
# 3.  TensorFlow / Keras Modulation Layer
# ====================================================================

class QAMModulationLayer(layers.Layer):
    """
    Keras layer: maps binary input tensor to complex IQ symbols.

    Input:  (batch, num_channel_bits)   float32  {0, 1}
    Output: (batch, num_symbols, 2)     float32  [I, Q]

    The constellation is stored as a non-trainable weight so that
    it can later be replaced by a trainable neural modulator.
    """

    def __init__(self, M: int = 16,
                 maprule: list = None,
                 Es: float = 10.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.bps = int(np.log2(M))

        S = build_constellation(M, maprule, Es)
        # Store as (M, 2) real tensor [I, Q]
        S_iq = np.stack([S.real, S.imag], axis=-1).astype(np.float32)
        self.S_iq = tf.constant(S_iq, dtype=tf.float32)

        bipower = 2 ** np.arange(self.bps - 1, -1, -1)
        self.bipower = tf.constant(bipower.astype(np.float32),
                                   dtype=tf.float32)

    def call(self, bits):
        """
        bits: (batch, num_channel_bits) float32 with values {0, 1}
        """
        batch_size = tf.shape(bits)[0]
        num_bits = tf.shape(bits)[1]
        num_sym = num_bits // self.bps

        # Reshape to (batch, num_sym, bps)
        bits_reshaped = tf.reshape(bits, (batch_size, num_sym, self.bps))
        # Compute symbol indices: dot with [2^(bps-1), ..., 1]
        indices = tf.cast(
            tf.reduce_sum(bits_reshaped * self.bipower, axis=-1),
            tf.int32
        )  # (batch, num_sym)

        # Gather constellation points
        symbols = tf.gather(self.S_iq, indices)  # (batch, num_sym, 2)
        return symbols

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'M': self.M})
        return cfg
