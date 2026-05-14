"""
bicm_id.py  –  BICM-ID Iterative Receiver Core
================================================
Integrates encoder, interleaver, modulator, channel, demodulator,
and decoder into the complete BICM-ID iterative processing loop.

This module provides:
  1. BICMIDSystem         – NumPy-based classic simulation
  2. BICMIDModel          – TF/Keras model for end-to-end DL experiments

The TF model stores all blocks as Keras layers, making it straightforward
to swap in neural-network-based modulator/demodulator/decoder later.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Optional, Tuple, Dict

from .modulation import (build_constellation, modulate_bits,
                         symbol_bit_matrix, QAMModulationLayer)
from .demodulation import (softdemod_qam, soft_demapper,
                           SoftDemodLayer, SoftDemapperLayer)
from .channel import awgn_channel, AWGNChannelLayer
from .encoders import RSCEncoder, HammingEncoder
from .decoders import SISODecoder, DualDecoder
from .interleavers import load_interleaver, random_interleaver


# ====================================================================
# 1.  Classic NumPy BICM-ID System
# ====================================================================

class BICMIDSystem:
    """
    Complete BICM-ID simulation system (NumPy implementation).

    Supports two decoder types:
      - 'conv'    : convolutional code + SISO BCJR decoder
      - 'hamming' : extended Hamming code + dual decoder

    Exactly reproduces the behaviour of sim_conv_bicmid.m and
    sim_ham_bicmid.m.
    """

    def __init__(self,
                 code_type: str = 'conv',
                 M: int = 16,
                 maprule: list = None,
                 Es: float = 10.0,
                 g: Optional[np.ndarray] = None,
                 hamming_m: int = 3,
                 interleaver: Optional[np.ndarray] = None,
                 interleaver_mat: Optional[str] = None,
                 max_iterations: int = 10,
                 scaling_factor: float = 0.85,
                 use_random_interleaver: bool = False):
        """
        Parameters
        ----------
        code_type   : 'conv' or 'hamming'
        M           : constellation size
        maprule     : mapping rule (1-indexed, MATLAB convention)
        Es          : normalisation energy
        g           : generator matrix for conv code, shape (n, K)
        hamming_m   : Hamming parameter
        interleaver : pre-loaded 0-indexed permutation, or None
        interleaver_mat : .mat filename to load interleaver from
        max_iterations  : number of BICM-ID iterations
        scaling_factor  : extrinsic information scaling
        use_random_interleaver : if True, generate random interleaver each frame
        """
        self.code_type = code_type
        self.M = M
        self.bps = int(np.log2(M))
        self.max_iterations = max_iterations
        self.scaling_factor = scaling_factor
        self.use_random_interleaver = use_random_interleaver

        # Constellation
        self.S = build_constellation(M, maprule, Es)
        self.sym_matrix = symbol_bit_matrix(M)

        # Encoder & Decoder
        if code_type == 'conv':
            if g is None:
                from .config import ConvCodeConfig
                cfg = ConvCodeConfig()
                g = cfg.g_matrix
            self.encoder = RSCEncoder(g)
            self.decoder = SISODecoder(g, code_type=0, dec_type=0)
            self.code_rate = self.encoder.code_rate
        elif code_type == 'hamming':
            self.ham_encoder = HammingEncoder(hamming_m)
            self.ham_decoder = DualDecoder(self.ham_encoder.H)
            self.code_rate = self.ham_encoder.code_rate
        else:
            raise ValueError(f"Unknown code_type: {code_type}")

        # Interleaver
        if interleaver is not None:
            self.alpha = interleaver
        elif interleaver_mat is not None:
            self.alpha = load_interleaver(interleaver_mat)
        else:
            self.alpha = None  # will be set per-frame

        if self.alpha is not None:
            self.num_channel_bits = len(self.alpha)
        else:
            self.num_channel_bits = None

    def _compute_noise_params(self, dB: float):
        """Compute noise parameters from Eb/N0 in dB."""
        SNR_lin = 10 ** (dB / 10) * self.bps
        N0_uncoded = 1.0 / SNR_lin
        N0 = N0_uncoded / self.code_rate
        return N0

    def simulate_one_frame(self, snr_dB: float):
        """
        Simulate one frame of BICM-ID.

        Returns
        -------
        v          : transmitted coded bits
        bit_errors : array of shape (max_iterations,) – errors per iteration
        """
        N0 = self._compute_noise_params(snr_dB)
        alpha = self.alpha

        if self.code_type == 'conv':
            return self._sim_conv_frame(N0, alpha)
        else:
            return self._sim_hamming_frame(N0, alpha)

    def _sim_conv_frame(self, N0: float, alpha: np.ndarray):
        """One frame of convolutional BICM-ID."""
        num_cb = len(alpha)
        info_len = int(num_cb * self.code_rate)
        bps = self.bps
        n_rate = self.encoder.n
        m = self.encoder.m

        # Generate random info bits
        u = np.random.randint(0, 2, info_len)
        v = self.encoder.encode(u)

        # Random interleaver (as in MATLAB: alpha = randperm each frame)
        if self.use_random_interleaver:
            alpha = random_interleaver(num_cb)

        # Interleave
        vv = v[alpha]

        # Modulate
        chan_in = modulate_bits(vv, self.S)

        # AWGN channel
        chan_out = awgn_channel(chan_in, N0)

        # Soft demodulation (channel metrics)
        demod = softdemod_qam(chan_out, N0, self.S)

        # Iterative decoding
        La = np.zeros(num_cb)
        Le = La[alpha].copy()
        b_llr = np.zeros(num_cb)
        SF = np.ones(num_cb) * self.scaling_factor

        bit_errors = np.zeros(self.max_iterations)

        for iteration in range(self.max_iterations):
            # Extrinsic from decoder, scaled
            Le = (La - b_llr) * SF
            Le_interleaved = Le[alpha]

            # Soft demapping with a-priori info
            b_llr_temp = soft_demapper(demod, Le_interleaved, self.sym_matrix)
            # De-interleave
            b_llr[alpha] = b_llr_temp

            # SISO decode
            input_u = np.zeros(info_len)
            input_c = np.concatenate([b_llr, np.zeros(m * n_rate)])
            output_u, output_c = self.decoder.decode(input_u, input_c)
            La = output_c[:num_cb]

            # Hard decision & error counting
            vhat = ((np.sign(La) + 1) / 2).astype(int)
            errors = np.sum(v != vhat)
            bit_errors[iteration] = errors

        return v, bit_errors

    def _sim_hamming_frame(self, N0: float, alpha: np.ndarray):
        """One frame of Hamming BICM-ID."""
        num_cb = len(alpha)
        n_code = self.ham_encoder.n_code
        k_info = self.ham_encoder.k_info
        frame_len = num_cb // n_code
        bps = self.bps

        # Generate random info bits
        u = np.zeros(frame_len * k_info, dtype=int)
        v = np.zeros(frame_len * n_code, dtype=int)
        for cnt in range(frame_len):
            temp = np.random.randint(0, 2, k_info)
            u[cnt * k_info: (cnt + 1) * k_info] = temp
            v[cnt * n_code: (cnt + 1) * n_code] = self.ham_encoder.encode(temp)

        # Interleave
        vv = v[alpha]

        # Modulate
        chan_in = modulate_bits(vv, self.S)

        # AWGN channel
        chan_out = awgn_channel(chan_in, N0)

        # Soft demodulation
        demod = softdemod_qam(chan_out, N0, self.S)

        # Iterative decoding
        La = np.zeros(num_cb)
        Le = La[alpha].copy()
        b_llr = np.zeros(num_cb)
        SF = np.ones(num_cb) * self.scaling_factor

        bit_errors = np.zeros(self.max_iterations)

        for iteration in range(self.max_iterations):
            Le = (La - b_llr) * SF
            Le_interleaved = Le[alpha]

            b_llr_temp = soft_demapper(demod, Le_interleaved, self.sym_matrix)
            b_llr[alpha] = b_llr_temp

            # Dual decode each codeword
            for cnt in range(frame_len):
                start = cnt * n_code
                end = (cnt + 1) * n_code
                La[start:end] = self.ham_decoder.decode(b_llr[start:end])

            # Hard decision
            vhat = ((np.sign(La) + 1) / 2).astype(int)
            errors = np.sum(v != vhat)
            bit_errors[iteration] = errors

        return v, bit_errors


# ====================================================================
# 2.  TensorFlow / Keras BICM-ID Model
# ====================================================================

class InterleaverLayer(layers.Layer):
    """
    Keras layer: applies a fixed permutation to bit sequences.

    Input:  (batch, num_bits)
    Output: (batch, num_bits)  — permuted
    """

    def __init__(self, permutation: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.perm = tf.constant(permutation.astype(np.int32))

    def call(self, x):
        return tf.gather(x, self.perm, axis=1)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class DeinterleaverLayer(layers.Layer):
    """
    Keras layer: applies the inverse permutation.

    Input:  (batch, num_bits)
    Output: (batch, num_bits)  — de-interleaved
    """

    def __init__(self, permutation: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        # Compute inverse permutation
        inv_perm = np.argsort(permutation)
        self.inv_perm = tf.constant(inv_perm.astype(np.int32))

    def call(self, x):
        return tf.gather(x, self.inv_perm, axis=1)


def build_bicmid_tx_model(S: np.ndarray,
                          alpha: np.ndarray,
                          num_channel_bits: int,
                          M: int = 16) -> Model:
    """
    Build the transmitter side of BICM-ID as a Keras model.

    coded_bits → interleave → modulate → tx_symbols

    This can be used standalone or combined with a channel + decoder
    for end-to-end training.

    Input:  (batch, num_channel_bits)
    Output: (batch, num_symbols, 2)
    """
    bps = int(np.log2(M))
    bits_in = layers.Input(shape=(num_channel_bits,), name='coded_bits')

    # Interleave
    interleaved = InterleaverLayer(alpha, name='interleaver')(bits_in)

    # Modulate
    symbols = QAMModulationLayer(M=M, name='qam_modulator')(interleaved)

    return Model(bits_in, symbols, name='BICM_Transmitter')


def build_bicmid_end2end_model(
        S: np.ndarray,
        sym_matrix: np.ndarray,
        alpha: np.ndarray,
        num_channel_bits: int,
        M: int = 16
) -> Model:
    """
    Build a full BICM-ID end-to-end Keras model:

        coded_bits → interleave → modulate → AWGN → demod → demapper → LLRs

    This is the foundation for replacing blocks with neural networks.

    Inputs:  [coded_bits (B, N), snr_db (B, 1)]
    Output:  LLRs (B, N)
    """
    bps = int(np.log2(M))

    bits_in = layers.Input(shape=(num_channel_bits,), name='coded_bits')
    snr_in = layers.Input(shape=(1,), name='snr_db')

    # Transmitter
    interleaved = InterleaverLayer(alpha, name='interleaver')(bits_in)
    symbols = QAMModulationLayer(M=M, name='qam_modulator')(interleaved)

    # Channel
    rx = AWGNChannelLayer(name='awgn_channel')([symbols, snr_in])

    # Convert SNR to N0 for demod
    # N0 = 1 / (SNR_lin * code_rate)  — simplified, user should set properly
    N0 = layers.Lambda(
        lambda s: 1.0 / (10.0 ** (s / 10.0) + 1e-8),
        name='snr_to_n0'
    )(snr_in)

    # Soft demodulation
    demod_metrics = SoftDemodLayer(S, name='soft_demod')([rx, N0])

    # Soft demapper (single iteration, no a-priori for first pass)
    La_zero = layers.Lambda(
        lambda x: tf.zeros_like(x), name='zero_apriori'
    )(bits_in)
    llr_out = SoftDemapperLayer(
        sym_matrix, name='soft_demapper'
    )([demod_metrics, La_zero])

    # De-interleave
    llr_deinterleaved = DeinterleaverLayer(
        alpha, name='deinterleaver'
    )(llr_out)

    model = Model([bits_in, snr_in], llr_deinterleaved,
                  name='BICM_ID_E2E')
    return model
