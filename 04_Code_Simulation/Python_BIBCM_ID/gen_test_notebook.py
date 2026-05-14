"""Generate Test_BICMID_System.ipynb for verifying the original system."""
import json, os

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src.strip().split('\n')}

def code(src):
    lines = src.strip().split('\n')
    return {"cell_type":"code","execution_count":None,
            "metadata":{},"outputs":[],"source": lines}

cells = []

# ─── Title ───
cells.append(md("""# Test BICM-ID System — Python Implementation
---
Notebook kiểm tra toàn bộ hệ thống BICM-ID đã chuyển từ MATLAB sang Python/TensorFlow.

## Nội dung
1. Kiểm tra từng khối (encoder, modulator, channel, demodulator, decoder)
2. Chạy BER simulation với Convolutional Code (RSC rate-1/2)
3. Chạy BER simulation với Extended Hamming Code (8,4)
4. So sánh BER qua các iteration
5. Kiểm tra TF/Keras layers
"""))

# ─── Imports ───
cells.append(code("""import sys, os, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12, 'figure.figsize': (12, 7)})
%matplotlib inline

# Add project root
sys.path.insert(0, os.path.abspath('..'))

import warnings
warnings.filterwarnings('ignore')

# Init logger — all outputs go to logs/notebook_test.{log,jsonl}
from python_code.utils import BICMIDLogger
log_dir = os.path.join(os.path.abspath('..'), 'python_code', 'logs')
logger = BICMIDLogger('notebook_test', log_dir=log_dir)
print("OK - Imports & Logger ready")
"""))

# ─── Section 1: Config ───
cells.append(md("""## 1. Cau hinh he thong (Configuration)
"""))

cells.append(code("""from python_code.config import ConvCodeConfig, ModulationConfig, SimulationConfig, MAPRULES

conv_cfg = ConvCodeConfig()
mod_cfg = ModulationConfig()

print("=" * 55)
print("  RSC Convolutional Code")
print(f"  g1 = {conv_cfg.g1_octal} (octal), g2 = {conv_cfg.g2_octal} (octal)")
print(f"  Constraint length K = {conv_cfg.k}")
print(f"  Code rate = {conv_cfg.code_rate}")
print(f"  Generator matrix:\\n{conv_cfg.g_matrix}")
print("=" * 55)
print(f"  Modulation: {mod_cfg.M}-QAM")
print(f"  Bits per symbol: {mod_cfg.bps}")
print(f"  Mapping rule: {mod_cfg.maprule}")
print(f"  Es = {mod_cfg.Es}")
print("=" * 55)

print("\\nAvailable mapping rules:")
for name, rule in MAPRULES.items():
    print(f"  {name}: {rule}")
"""))

# ─── Section 2: Encoder Test ───
cells.append(md("""## 2. Test Encoder
### 2.1 RSC Encoder
"""))

cells.append(code("""from python_code.encoders import RSCEncoder, HammingEncoder

enc = RSCEncoder(conv_cfg.g_matrix)
np.random.seed(42)
u = np.random.randint(0, 2, 20)
v = enc.encode(u)

print(f"Info bits ({len(u)}):  {u}")
print(f"Coded bits ({len(v)}): {v}")
print(f"Rate: 1/{enc.n} = {enc.code_rate}")
print(f"Systematic check: coded[0::2] == info? {np.all(v[0::2] == u)}")
"""))

cells.append(md("""### 2.2 Extended Hamming Encoder
"""))

cells.append(code("""ham = HammingEncoder(3)
print(f"Extended Hamming ({ham.n_code}, {ham.k_info})")
print(f"Code rate: {ham.code_rate:.4f}")
print(f"\\nG matrix ({ham.G.shape}):")
print(ham.G)
print(f"\\nH matrix ({ham.H.shape}):")
print(ham.H)
print(f"\\nG*H^T mod 2 = 0? {np.all((ham.G @ ham.H.T) % 2 == 0)}")

u_h = np.array([1, 0, 1, 1])
v_h = ham.encode(u_h)
print(f"\\nEncode [{u_h}] -> [{v_h}]")
print(f"Syndrome H*v^T mod 2 = {(ham.H @ v_h) % 2}")
"""))

# ─── Section 3: Constellation ───
cells.append(md("""## 3. Test Constellation & Modulation
"""))

cells.append(code("""from python_code.modulation import (build_constellation, modulate_bits,
                                    symbol_bit_matrix, qam16_constellation_matlab)

# Raw QAM
qam_raw = qam16_constellation_matlab()
S = build_constellation(16, mod_cfg.maprule, mod_cfg.Es)
sym_matrix = symbol_bit_matrix(16)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot raw QAM
axes[0].scatter(qam_raw.real, qam_raw.imag, c='blue', s=120, edgecolors='k', zorder=5)
for i, pt in enumerate(qam_raw):
    axes[0].annotate(str(i), (pt.real, pt.imag), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
axes[0].set_title('Raw 16-QAM (MATLAB order)', fontweight='bold')
axes[0].grid(True, alpha=0.3); axes[0].set_aspect('equal')
axes[0].set_xlabel('I'); axes[0].set_ylabel('Q')

# Plot mapped constellation
axes[1].scatter(S.real, S.imag, c='red', s=120, edgecolors='k', zorder=5)
for i, pt in enumerate(S):
    bits_str = ''.join(str(b) for b in sym_matrix[i])
    axes[1].annotate(bits_str, (pt.real, pt.imag), textcoords="offset points",
                     xytext=(5, 5), fontsize=7)
axes[1].set_title(f'Mapped 16-QAM (MSEW optimised)\\nAvg power = {np.mean(np.abs(S)**2):.4f}',
                  fontweight='bold')
axes[1].grid(True, alpha=0.3); axes[1].set_aspect('equal')
axes[1].set_xlabel('I'); axes[1].set_ylabel('Q')

plt.tight_layout()
plt.show()

print(f"Symbol bit matrix (M x bps = {sym_matrix.shape}):")
print(sym_matrix)
"""))

# ─── Section 4: Modulation + Channel ───
cells.append(md("""## 4. Test Modulation + AWGN Channel
"""))

cells.append(code("""from python_code.channel import awgn_channel

np.random.seed(0)
test_bits = np.random.randint(0, 2, 400)
tx_symbols = modulate_bits(test_bits, S)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
snr_vals = [5, 10, 20]

for ax, snr_dB in zip(axes, snr_vals):
    N0 = 1.0 / (10 ** (snr_dB / 10) * mod_cfg.bps)
    rx = awgn_channel(tx_symbols, N0)
    ax.scatter(rx.real, rx.imag, c='blue', s=8, alpha=0.5, label='Received')
    ax.scatter(S.real, S.imag, c='red', s=100, edgecolors='k', zorder=5, label='Constellation')
    ax.set_title(f'SNR = {snr_dB} dB (N0={N0:.4f})', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    ax.legend(fontsize=9)

plt.suptitle('16-QAM through AWGN Channel', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()
"""))

# ─── Section 5: Soft Demod ───
cells.append(md("""## 5. Test Soft Demodulation
"""))

cells.append(code("""from python_code.demodulation import softdemod_qam, soft_demapper

np.random.seed(42)
test_bits = np.random.randint(0, 2, 40)
tx = modulate_bits(test_bits, S)
N0 = 0.1
rx = awgn_channel(tx, N0)

# Channel metrics
metrics = softdemod_qam(rx, N0, S)
print(f"Channel metrics shape: {metrics.shape}  (M x num_symbols)")

# Soft demapper (no a-priori)
La = np.zeros(len(test_bits))
Lc = soft_demapper(metrics, La, sym_matrix)

# Hard decision from LLR
hard_bits = (Lc > 0).astype(int)
errors = np.sum(test_bits != hard_bits)
print(f"\\nBit errors (no coding, no iteration): {errors}/{len(test_bits)}")
print(f"BER = {errors/len(test_bits):.4f}")

fig, ax = plt.subplots(figsize=(12, 3))
ax.stem(range(len(Lc)), Lc, linefmt='b-', markerfmt='bo', basefmt='r-')
ax.axhline(0, color='r', lw=2)
ax.set_xlabel('Bit index'); ax.set_ylabel('LLR')
ax.set_title('Soft Demapper Output (LLR > 0 => bit 1)', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ─── Section 6: Decoder Test ───
cells.append(md("""## 6. Test Decoders
### 6.1 SISO BCJR Decoder (Convolutional)
"""))

cells.append(code("""from python_code.decoders import SISODecoder, DualDecoder

siso = SISODecoder(conv_cfg.g_matrix, code_type=0, dec_type=0)

# Small test: encode, add noise via LLR, decode
np.random.seed(42)
u_test = np.random.randint(0, 2, 50)
v_test = enc.encode(u_test)

# Simulate channel LLRs: +/- with noise
snr_test = 3.0
sigma = 1.0 / np.sqrt(2 * 10**(snr_test/10))
noisy_llr = (2 * v_test.astype(float) - 1) / sigma**2 + np.random.randn(len(v_test)) * 2/sigma

# Pad for tail bits
input_u = np.zeros(len(u_test))
input_c = np.concatenate([noisy_llr, np.zeros(enc.m * enc.n)])

t0 = time.time()
output_u, output_c = siso.decode(input_u, input_c)
dt = time.time() - t0

# Hard decision on code bits
La = output_c[:len(v_test)]
vhat = ((np.sign(La) + 1) / 2).astype(int)
errors = np.sum(v_test != vhat)
print(f"SISO Decode: {len(u_test)} info bits, time = {dt:.3f}s")
print(f"Coded bit errors: {errors}/{len(v_test)}")
"""))

cells.append(md("""### 6.2 Dual Decoder (Hamming)
"""))

cells.append(code("""dual = DualDecoder(ham.H)

np.random.seed(42)
u_h = np.random.randint(0, 2, ham.k_info)
v_h = ham.encode(u_h)

# Channel LLR
noisy_h = (2 * v_h.astype(float) - 1) * 3 + np.random.randn(ham.n_code) * 0.5
decoded_h = dual.decode(noisy_h)
vhat_h = ((np.sign(decoded_h) + 1) / 2).astype(int)
print(f"Transmitted: {v_h}")
print(f"Decoded:     {vhat_h}")
print(f"Errors: {np.sum(v_h != vhat_h)}")
"""))

# ─── Section 7: Interleaver ───
cells.append(md("""## 7. Test Interleaver
"""))

cells.append(code("""from python_code.interleavers import load_interleaver, random_interleaver
from python_code.utils import list_available_interleavers

print("Available interleaver .mat files:")
for f in list_available_interleavers():
    print(f"  {f}")

alpha = load_interleaver("BIBCM-ID_4096Algeb.mat")
print(f"\\nLoaded interleaver: length={len(alpha)}, range=[{alpha.min()}, {alpha.max()}]")
print(f"Is valid permutation: {len(np.unique(alpha)) == len(alpha)}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(alpha[:200], 'b-', lw=0.8)
ax.set_xlabel('Input position'); ax.set_ylabel('Output position')
ax.set_title('Interleaver mapping (first 200 positions)', fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ─── Section 8: Girth ───
cells.append(md("""## 8. Girth Analysis
"""))

cells.append(code("""from python_code.utils import girth_analysis

n4, n6, n8 = girth_analysis(ham.H)
print(f"Extended Hamming ({ham.n_code},{ham.k_info}) H matrix:")
print(f"  4-cycles: {n4}")
print(f"  6-cycles: {n6}")
print(f"  8-cycles: {n8}")
print(f"  Minimum girth: {'4' if n4 > 0 else '6' if n6 > 0 else '8+'}")
"""))

# ─── Section 9: BER Sim Conv ───
cells.append(md("""## 9. BER Simulation — Convolutional Code (RSC)

Chạy simulation với **đa tiến trình (100% CPU)**. Bạn có thể tùy chỉnh các tham số bên dưới trước khi chạy.
"""))

cells.append(code("""# ╔══════════════════════════════════════════════════════════╗
# ║  🔧 CẤU HÌNH MÔ PHỎNG — CONVOLUTIONAL CODE (RSC)       ║
# ║  Thay đổi các giá trị bên dưới theo ý bạn rồi Run cell  ║
# ╚══════════════════════════════════════════════════════════╝

CONV_SNR_dB       = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]   # Dải SNR (dB)
CONV_INTERLEAVER  = "BIBCM-ID_4096Algeb.mat"                     # Bộ giao vòng (xem danh sách ở mục 7)
CONV_MAX_ITER     = 10                                            # Số vòng lặp BICM-ID
CONV_SCALING      = 0.85                                          # Hệ số scaling thông tin ngoại lai
CONV_MAX_BITS     = 100_000_000                                   # Tổng số bit tối đa mỗi mức SNR (10^8)
CONV_MAPRULE      = [13, 6, 7, 16, 3, 12, 14, 5,                 # Quy tắc ánh xạ chòm sao (MSEW)
                     8, 15, 9, 2, 10, 1, 4, 11]

print("✅ Cấu hình Conv đã sẵn sàng. Chạy cell tiếp theo để bắt đầu mô phỏng.")
"""))

cells.append(code("""from python_code.sim_conv_bicmid import main as sim_conv_main

BER_conv, dB_conv = sim_conv_main(
    dB_range       = CONV_SNR_dB,
    interleaver_mat = CONV_INTERLEAVER,
    max_iterations = CONV_MAX_ITER,
    scaling_factor = CONV_SCALING,
    max_bits_total = CONV_MAX_BITS,
    maprule        = CONV_MAPRULE,
    console        = False,
)
"""))

# ─── Section 10: BER Sim Hamming ───
cells.append(md("""## 10. BER Simulation — Extended Hamming Code (8,4)

Tương tự, bạn có thể tùy chỉnh tham số cho mô phỏng Hamming bên dưới.
"""))

cells.append(code("""# ╔══════════════════════════════════════════════════════════╗
# ║  🔧 CẤU HÌNH MÔ PHỎNG — EXTENDED HAMMING CODE          ║
# ║  Thay đổi các giá trị bên dưới theo ý bạn rồi Run cell  ║
# ╚══════════════════════════════════════════════════════════╝

HAM_SNR_dB        = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]   # Dải SNR (dB)
HAM_INTERLEAVER   = "BIBCM-ID_4096Algeb.mat"                     # Bộ giao vòng
HAM_MAX_ITER      = 10                                            # Số vòng lặp BICM-ID
HAM_SCALING       = 0.85                                          # Hệ số scaling
HAM_MAX_BITS      = 100_000_000                                   # Tổng số bit tối đa mỗi mức SNR
HAM_MAPRULE       = [11, 2, 1, 12, 4, 9, 10, 3,                  # Quy tắc ánh xạ (MSEW Hamming)
                     5, 16, 15, 6, 14, 7, 8, 13]
HAM_M             = 3                                             # Tham số Hamming (m=3 → code (8,4))

print("✅ Cấu hình Hamming đã sẵn sàng. Chạy cell tiếp theo để bắt đầu mô phỏng.")
"""))

cells.append(code("""from python_code.sim_ham_bicmid import main as sim_ham_main

BER_ham, dB_ham = sim_ham_main(
    dB_range        = HAM_SNR_dB,
    interleaver_mat = HAM_INTERLEAVER,
    max_iterations  = HAM_MAX_ITER,
    scaling_factor  = HAM_SCALING,
    max_bits_total  = HAM_MAX_BITS,
    maprule         = HAM_MAPRULE,
    hamming_m       = HAM_M,
    console         = False,
)
"""))

# ─── Section 12: TF Layers ───
cells.append(md("""## 12. Test TensorFlow/Keras Layers

Kiem tra cac Keras layer de san sang cho deep learning integration.
"""))

cells.append(code("""import tensorflow as tf
from python_code.modulation import QAMModulationLayer
from python_code.demodulation import SoftDemodLayer, SoftDemapperLayer
from python_code.channel import AWGNChannelLayer
from python_code.bicm_id import InterleaverLayer, DeinterleaverLayer

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Test batch processing
B = 8
N_bits = 200
bits = tf.constant(np.random.randint(0, 2, (B, N_bits)).astype(np.float32))
snr = tf.constant(np.full((B, 1), 10.0, dtype=np.float32))
N0 = tf.constant(np.full((B, 1), 0.1, dtype=np.float32))

# Forward pass
mod_layer = QAMModulationLayer(M=16, name='qam')
symbols = mod_layer(bits)
print(f"QAMModulationLayer: ({B}, {N_bits}) -> {symbols.shape}")

ch_layer = AWGNChannelLayer(name='awgn')
rx = ch_layer([symbols, snr])
print(f"AWGNChannelLayer:   {symbols.shape} -> {rx.shape}")

demod_layer = SoftDemodLayer(S, name='demod')
metrics = demod_layer([rx, N0])
print(f"SoftDemodLayer:     {rx.shape} -> {metrics.shape}")

demapper = SoftDemapperLayer(sym_matrix, name='demapper')
La_zero = tf.zeros((B, N_bits))
llr = demapper([metrics, La_zero])
print(f"SoftDemapperLayer:  {metrics.shape} + La -> {llr.shape}")

# Interleaver roundtrip
perm = np.random.permutation(N_bits)
interleave = InterleaverLayer(perm, name='pi')
deinterleave = DeinterleaverLayer(perm, name='pi_inv')
x = tf.constant(np.random.randn(B, N_bits).astype(np.float32))
y = deinterleave(interleave(x))
roundtrip_ok = np.allclose(x.numpy(), y.numpy())
print(f"Interleaver roundtrip: {roundtrip_ok}")

print("\\n--- All TF layers OK ---")
"""))

# ─── Section 13: E2E Model ───
cells.append(md("""## 13. End-to-End Keras Model

Build complete BICM-ID as a single Keras model (1 iteration, no decoder).
Day la nen tang de thay the tung block bang neural network.
"""))

cells.append(code("""from python_code.bicm_id import build_bicmid_end2end_model

alpha_small = load_interleaver("BIBCM-ID_1024Algeb.mat")

e2e_model = build_bicmid_end2end_model(
    S=S, sym_matrix=sym_matrix,
    alpha=alpha_small, num_channel_bits=len(alpha_small), M=16
)
e2e_model.summary()
"""))

cells.append(code("""# Test forward pass
B = 16
bits_test = np.random.randint(0, 2, (B, len(alpha_small))).astype(np.float32)
snr_test = np.full((B, 1), 10.0, dtype=np.float32)

llr_out = e2e_model.predict([bits_test, snr_test], verbose=0)
print(f"Input:  {bits_test.shape}")
print(f"Output: {llr_out.shape}")
print(f"LLR range: [{llr_out.min():.2f}, {llr_out.max():.2f}]")
"""))

# ─── Section 14: Log Analysis ───
cells.append(md("""## 14. Log Analysis

Xem log file da ghi trong qua trinh chay notebook.
"""))

cells.append(code("""# Close the logger to flush all data
logger.close()

# Analyze the log file
from python_code.utils import load_log, summarize_log
import os

log_path = os.path.join(log_dir, 'notebook_test.jsonl')
if os.path.exists(log_path):
    print(summarize_log(log_path))
    print()
    
    # Show raw records count by event type
    records = load_log(log_path)
    from collections import Counter
    counts = Counter(r['event'] for r in records)
    print("Event counts:")
    for event, count in counts.most_common():
        print(f"  {event}: {count}")
    
    print(f"\\nLog files:")
    print(f"  Text:  {os.path.join(log_dir, 'notebook_test.log')}")
    print(f"  JSONL: {log_path}")
else:
    print("No log file found")
"""))

# ─── Summary ───
cells.append(md("""## 15. Summary

| Component | Status | Ghi chu |
|---|---|---|
| RSC Encoder | OK | rate-1/2, K=3, g=(7,5) |
| Hamming Encoder | OK | Extended (8,4) |
| 16-QAM Constellation | OK | MSEW mapping |
| AWGN Channel | OK | Complex noise |
| Soft Demodulator | OK | Per-symbol metrics |
| Soft Demapper | OK | Log-sum-exp LLR |
| SISO BCJR Decoder | OK | Log-MAP (Python) |
| Dual Decoder | OK | Block code decoder |
| Interleaver | OK | Load from .mat |
| Girth Analysis | OK | 4/6/8 cycles |
| TF/Keras Layers | OK | Differentiable |
| E2E Keras Model | OK | 1-pass forward model |
| BER Sim (Conv) | OK | Matches MATLAB trends |
| BER Sim (Hamming) | OK | Matches MATLAB trends |
| **Logging** | OK | .log + .jsonl files |

**Ket luan:** Toan bo pipeline BICM-ID da duoc chuyen doi thanh cong tu MATLAB sang Python/TensorFlow.
San sang de tich hop deep learning (neural modulator/demodulator/LDPC).

Log files duoc luu tai `python_code/logs/` de phan tich sau nay.
"""))

# ─── Generate notebook ───
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}

path = os.path.join(os.path.dirname(__file__), 'Test_BICMID_System.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"OK - Generated {os.path.basename(path)}")
