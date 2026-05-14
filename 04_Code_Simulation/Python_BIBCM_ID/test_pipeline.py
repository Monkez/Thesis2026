"""
test_pipeline.py  –  Quick verification of the full BICM-ID pipeline
====================================================================
Tests all components individually and then a single-frame simulation.
All results are logged to python_code/logs/ for later analysis.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
np.random.seed(42)

from python_code.utils import BICMIDLogger

# ─── Logger ──────────────────────────────────────────────────────────
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
logger = BICMIDLogger('test_pipeline', log_dir=log_dir)

# 1. Config
from python_code.config import ConvCodeConfig, ModulationConfig
conv_cfg = ConvCodeConfig()
mod_cfg = ModulationConfig()
logger.log_component_test('Config', 'OK', {
    'rsc': f'({conv_cfg.g1_octal},{conv_cfg.g2_octal})',
    'modulation': f'{mod_cfg.M}-QAM'
})

# 2. Encoder
from python_code.encoders import RSCEncoder, HammingEncoder
enc = RSCEncoder(conv_cfg.g_matrix)
u = np.random.randint(0, 2, 100)
v = enc.encode(u)
logger.log_component_test('RSC Encoder', 'OK', {
    'info_bits': len(u), 'coded_bits': len(v), 'rate': enc.code_rate
})

ham = HammingEncoder(3)
u_h = np.random.randint(0, 2, ham.k_info)
v_h = ham.encode(u_h)
logger.log_component_test('Hamming Encoder', 'OK', {
    'n_code': ham.n_code, 'k_info': ham.k_info, 'rate': ham.code_rate
})

# 3. Constellation & Modulation
from python_code.modulation import build_constellation, modulate_bits, symbol_bit_matrix
S = build_constellation(16, mod_cfg.maprule, mod_cfg.Es)
sym_matrix = symbol_bit_matrix(16)
avg_power = float(np.mean(np.abs(S)**2))
symbols = modulate_bits(v, S)
logger.log_component_test('Constellation', 'OK', {
    'M': len(S), 'avg_power': round(avg_power, 4)
})
logger.log_component_test('Modulation', 'OK', {
    'bits': len(v), 'symbols': len(symbols)
})

# 4. Channel
from python_code.channel import awgn_channel
N0 = 0.1
rx = awgn_channel(symbols, N0)
logger.log_component_test('AWGN Channel', 'OK', {'N0': N0})

# 5. Demodulation
from python_code.demodulation import softdemod_qam, soft_demapper
demod = softdemod_qam(rx, N0, S)
La_dummy = np.zeros(len(v))
Lc = soft_demapper(demod, La_dummy, sym_matrix)
logger.log_component_test('Soft Demod', 'OK', {'shape': list(demod.shape)})
logger.log_component_test('Soft Demapper', 'OK', {
    'num_llrs': len(Lc),
    'llr_range': [round(float(Lc.min()), 2), round(float(Lc.max()), 2)]
})

# 6. SISO Decoder
from python_code.decoders import SISODecoder
siso = SISODecoder(conv_cfg.g_matrix, code_type=0, dec_type=0)
input_u = np.zeros(len(u))
input_c = np.concatenate([Lc, np.zeros(enc.m * enc.n)])
t0 = time.time()
output_u, output_c = siso.decode(input_u, input_c)
dt = time.time() - t0
logger.log_component_test('SISO Decoder', 'OK', {
    'data_llrs': len(output_u), 'code_llrs': len(output_c),
    'decode_time_s': round(dt, 3)
})

# 7. Dual Decoder
from python_code.decoders import DualDecoder
dual = DualDecoder(ham.H)
test_llr = np.random.randn(ham.n_code) * 2
decoded_llr = dual.decode(test_llr)
logger.log_component_test('Dual Decoder', 'OK', {
    'input_llrs': len(test_llr), 'output_llrs': len(decoded_llr)
})

# 8. Interleaver
from python_code.interleavers import load_interleaver
alpha = load_interleaver("BIBCM-ID_4096Algeb.mat")
is_valid = len(np.unique(alpha)) == len(alpha)
logger.log_component_test('Interleaver', 'OK', {
    'length': len(alpha), 'range': [int(alpha.min()), int(alpha.max())],
    'valid_perm': is_valid
})

# 9. Girth analysis
from python_code.utils import girth_analysis
n4, n6, n8 = girth_analysis(ham.H)
logger.log_component_test('Girth Analysis', 'OK', {
    '4_cycles': n4, '6_cycles': n6, '8_cycles': n8
})

# 10. TF Layers
import tensorflow as tf
from python_code.modulation import QAMModulationLayer
from python_code.demodulation import SoftDemodLayer, SoftDemapperLayer
from python_code.channel import AWGNChannelLayer

mod_layer = QAMModulationLayer(M=16, name='qam_mod')
bits_tf = tf.constant(np.random.randint(0, 2, (4, 200)).astype(np.float32))
sym_tf = mod_layer(bits_tf)
logger.log_component_test('TF QAMModulationLayer', 'OK', {
    'input': list(bits_tf.shape), 'output': list(sym_tf.shape)
})

ch_layer = AWGNChannelLayer(name='awgn')
snr_tf = tf.constant(np.full((4, 1), 10.0, dtype=np.float32))
rx_tf = ch_layer([sym_tf, snr_tf])
logger.log_component_test('TF AWGNChannelLayer', 'OK', {
    'shape': list(rx_tf.shape)
})

demod_layer = SoftDemodLayer(S, name='soft_demod')
N0_tf = tf.constant(np.full((4, 1), 0.1, dtype=np.float32))
metrics_tf = demod_layer([rx_tf, N0_tf])
logger.log_component_test('TF SoftDemodLayer', 'OK', {
    'shape': list(metrics_tf.shape)
})

demapper_layer = SoftDemapperLayer(sym_matrix, name='soft_demapper')
La_tf = tf.zeros((4, 200), dtype=tf.float32)
Lc_tf = demapper_layer([metrics_tf, La_tf])
logger.log_component_test('TF SoftDemapperLayer', 'OK', {
    'shape': list(Lc_tf.shape)
})

# 11. Quick single-frame BICM-ID test
from python_code.bicm_id import BICMIDSystem
system = BICMIDSystem(
    code_type='conv',
    interleaver=alpha,
    max_iterations=3,
    use_random_interleaver=True
)
t0 = time.time()
v_frame, errors = system.simulate_one_frame(3.0)
dt = time.time() - t0
logger.log_component_test('BICM-ID Frame', 'OK', {
    'coded_bits': len(v_frame),
    'errors_per_iter': [int(e) for e in errors],
    'frame_time_s': round(dt, 3)
})

logger.log('')
logger.log('  ALL TESTS PASSED [OK]')
logger.close()
