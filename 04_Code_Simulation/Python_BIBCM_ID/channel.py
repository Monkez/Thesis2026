"""
channel.py  –  Channel models for BIBCM-ID (TF/Keras)
======================================================
Provides both NumPy functions and TensorFlow/Keras layers.

Channel models:
  1. AWGN Channel         – additive white Gaussian noise
  2. AWGNChannelLayer     – Keras layer (differentiable, for training)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from numba import njit


# ====================================================================
# 1.  AWGN Channel  (NumPy)
# ====================================================================

def awgn_channel(symbols: np.ndarray,
                 N0: float) -> np.ndarray:
    """
    Pass complex symbols through an AWGN channel.

    Parameters
    ----------
    symbols : complex array of any shape
    N0      : noise power spectral density

    Returns
    -------
    received : complex array, same shape as symbols
    """
    noise = (np.random.randn(*symbols.shape)
             + 1j * np.random.randn(*symbols.shape))
    return symbols + np.sqrt(N0 / 2) * noise


@njit(nogil=True, cache=True)
def cfopa_channel(tx, N0, a_sat=1.2, p_rapp=3, alpha_pm=0.08, beta_pm=0.0, cfo_norm=0.03):
    """
    Numba-accelerated PA + CFO + AWGN Channel
    """
    rx = np.zeros_like(tx, dtype=np.complex128)
    
    # Numba doesn't support vector np.angle directly for complex arrays in some older versions,
    # but we can do it element-wise for maximum speed and compatibility
    for i in range(len(tx)):
        sym = tx[i]
        r = np.abs(sym)
        theta = np.angle(sym)
        
        ratio = r / a_sat
        g_r = r / ((1.0 + ratio**(2*p_rapp))**(1.0/(2*p_rapp)))
        
        phi_pa = alpha_pm * r**2 / (1.0 + beta_pm * r**2)
        pa_tx = g_r * np.exp(1j * (theta + phi_pa))
        
        phase = 2.0 * np.pi * cfo_norm * i
        cfo_tx = pa_tx * np.exp(1j * phase)
        
        noise_re = np.random.randn()
        noise_im = np.random.randn()
        rx[i] = cfo_tx + np.sqrt(N0 / 2) * (noise_re + 1j * noise_im)
        
    return rx


# ====================================================================
# 2.  AWGN Channel  (TF/Keras Layer)
# ====================================================================

class AWGNChannelLayer(layers.Layer):
    """
    Keras layer: AWGN channel for IQ symbols.

    Input:  [tx_iq, snr_db]
        tx_iq   : (batch, num_sym, 2) – transmitted [I, Q]
        snr_db  : (batch, 1) – SNR in dB (Eb/N0 or Es/N0 depending on usage)

    Output: (batch, num_sym, 2) – received [I, Q]

    During training, noise is added. During inference (training=False),
    noise is still added (channels are always stochastic).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        tx_iq, snr_db = inputs

        snr_lin = 10.0 ** (snr_db / 10.0)  # (B, 1)
        # sigma per real dimension: sigma = sqrt(1 / (2 * SNR_lin))
        sigma = tf.sqrt(1.0 / (2.0 * snr_lin + 1e-8))
        sigma = tf.reshape(sigma, (-1, 1, 1))  # (B, 1, 1)

        noise = tf.random.normal(tf.shape(tx_iq)) * sigma
        return tx_iq + noise


class AWGNChannelComplexLayer(layers.Layer):
    """
    AWGN channel operating on complex-valued tensors stored as
    (batch, num_sym, 2) where dim[-1] = [real, imag].

    Takes noise variance N0 directly (not SNR in dB).

    Input:  [tx_iq, N0]
        tx_iq : (batch, num_sym, 2)
        N0    : (batch, 1) – noise PSD

    Output: (batch, num_sym, 2)
    """

    def call(self, inputs, training=None):
        tx_iq, N0 = inputs
        N0_exp = tf.reshape(N0, (-1, 1, 1))
        sigma = tf.sqrt(N0_exp / 2.0 + 1e-10)
        noise = tf.random.normal(tf.shape(tx_iq)) * sigma
        return tx_iq + noise
