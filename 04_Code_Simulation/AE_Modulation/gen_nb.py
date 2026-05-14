"""Generate Demo3_Adaptive_Rate.ipynb — k=4, configs 4-2/4-4/4-6"""
import json, os

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src.strip().split('\n')}

def code(src):
    lines = src.strip().split('\n')
    return {"cell_type":"code","execution_count":None,
            "metadata":{},"outputs":[],"source": lines}

cells = []

# ─── Title ───
cells.append(md("""# Demo 3: ANN Channel Coding — Adaptive Rate Modulation
---
## Neural Network as Channel Encoder (FEC-like)

**Key Innovation**: The ANN encoder outputs **MORE symbols** than needed,
learning *redundancy* similar to Forward Error Correction (FEC).

**k = 4 bits** → 2⁴ = 16 codewords

| Config | k bits | N symbols | Rate R | Analogous to |
|--------|--------|-----------|--------|-------------|
| 4→1 (R=2.0) | 4 | 1 | 2.00 | Max compression |
| 4→2 (R=1.0) | 4 | 2 | 1.00 | Uncoded |
| 4→4 (R=0.5) | 4 | 4 | 0.50 | Rate-1/2 FEC |
| 4→6 (R=0.33) | 4 | 6 | 0.33 | Rate-1/3 FEC |

**Channel**: Rapp PA + CFO + AWGN
"""))

# ─── Imports ───
cells.append(code("""import importlib
import adaptive_rate as ar
importlib.reload(ar)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})
import warnings
warnings.filterwarnings('ignore')
print("✓ Module loaded")
"""))

# ─── Section 1: Config ───
cells.append(md("""## 1. System Parameters
"""))

cells.append(code("""K_BITS = 4
A_SAT = 1.2
P_RAPP = 3
ALPHA_PM = 0.08
BETA_PM = 0.0
CFO_MAX = 0.03
BATCH_SIZE = 4096

CONFIGS = [
    (1, "4→1 R=2.00"),
    (2, "4→2 R=1.00"),
    (4, "4→4 R=0.50"),
    (6, "4→6 R=0.33"),
]

print("=" * 55)
print(f"  k={K_BITS} bits, 2^{K_BITS}={2**K_BITS} codewords")
print("=" * 55)
for n_sym, label in CONFIGS:
    rate = K_BITS / (n_sym * 2)
    print(f"  {label}: {n_sym} IQ sym, Rate={rate:.2f}")
print("=" * 55)
"""))

# ─── Train 4→6 (R≈0.33) ───
cells.append(md("""## 2. Train — 4 bits → 6 symbols (Rate ≈ 1/3)
Most redundancy: 12 real dims for 16 codewords.
"""))

cells.append(code("""logger_r03 = ar.init_logger('train_4to6.txt')
n_sym_r03 = 6

encoder_r03, ae_r03 = ar.build_coded_autoencoder(
    k_bits=K_BITS, n_sym=n_sym_r03,
    a_sat=A_SAT, p_rapp=P_RAPP,
    alpha_pm=ALPHA_PM, beta_pm=BETA_PM,
    cfo_max=CFO_MAX)

logger_r03.log_params(K_BITS, n_sym_r03, A_SAT, P_RAPP,
                       ALPHA_PM, BETA_PM, CFO_MAX, BATCH_SIZE)
logger_r03.log_model(encoder_r03)
logger_r03.log_model(ae_r03)
ae_r03.summary()
"""))

cells.append(code("""hist_r03 = ar.train_curriculum_coded(
    ae_r03, encoder_r03, k_bits=K_BITS, n_sym=n_sym_r03,
    a_sat=A_SAT, batch_size=BATCH_SIZE,
    logger=logger_r03, verbose=0, use_custom_loss=False)
print("\\n✓ 4→6 done")
"""))

# ─── Train 4→4 (R=0.5) ───
cells.append(md("""## 3. Train — 4 bits → 4 symbols (Rate = 1/2)
"""))

cells.append(code("""logger_r05 = ar.init_logger('train_4to4.txt')
n_sym_r05 = 4

encoder_r05, ae_r05 = ar.build_coded_autoencoder(
    k_bits=K_BITS, n_sym=n_sym_r05,
    a_sat=A_SAT, p_rapp=P_RAPP,
    alpha_pm=ALPHA_PM, beta_pm=BETA_PM,
    cfo_max=CFO_MAX)

logger_r05.log_params(K_BITS, n_sym_r05, A_SAT, P_RAPP,
                       ALPHA_PM, BETA_PM, CFO_MAX, BATCH_SIZE)
logger_r05.log_model(encoder_r05)
logger_r05.log_model(ae_r05)
"""))

cells.append(code("""hist_r05 = ar.train_curriculum_coded(
    ae_r05, encoder_r05, k_bits=K_BITS, n_sym=n_sym_r05,
    a_sat=A_SAT, batch_size=BATCH_SIZE,
    logger=logger_r05, verbose=0, use_custom_loss=False)
print("\\n✓ 4→4 done")
"""))

# ─── Train 4→2 (R=1.0, baseline) ───
cells.append(md("""## 4. Train — 4 bits → 2 symbols (Rate = 1, no redundancy)
"""))

cells.append(code("""logger_r1 = ar.init_logger('train_4to2.txt')
n_sym_r1 = 2

encoder_r1, ae_r1 = ar.build_coded_autoencoder(
    k_bits=K_BITS, n_sym=n_sym_r1,
    a_sat=A_SAT, p_rapp=P_RAPP,
    alpha_pm=ALPHA_PM, beta_pm=BETA_PM,
    cfo_max=CFO_MAX)

logger_r1.log_params(K_BITS, n_sym_r1, A_SAT, P_RAPP,
                      ALPHA_PM, BETA_PM, CFO_MAX, BATCH_SIZE)
logger_r1.log_model(encoder_r1)
logger_r1.log_model(ae_r1)
"""))

cells.append(code("""hist_r1 = ar.train_curriculum_coded(
    ae_r1, encoder_r1, k_bits=K_BITS, n_sym=n_sym_r1,
    a_sat=A_SAT, batch_size=BATCH_SIZE,
    logger=logger_r1, verbose=0, use_custom_loss=False)
print("\\n✓ 4→2 done")
"""))

# ─── Train 4→1 (R=2.0, max compression) ───
cells.append(md("""## 5. Train — 4 bits → 1 symbol (Rate = 2, max compression)
No redundancy at all — 4 bits compressed into 1 IQ symbol (2 real dims).
Similar to standard 16-QAM mapping.
"""))

cells.append(code("""logger_r2 = ar.init_logger('train_4to1.txt')
n_sym_r2 = 1

encoder_r2, ae_r2 = ar.build_coded_autoencoder(
    k_bits=K_BITS, n_sym=n_sym_r2,
    a_sat=A_SAT, p_rapp=P_RAPP,
    alpha_pm=ALPHA_PM, beta_pm=BETA_PM,
    cfo_max=CFO_MAX)

logger_r2.log_params(K_BITS, n_sym_r2, A_SAT, P_RAPP,
                      ALPHA_PM, BETA_PM, CFO_MAX, BATCH_SIZE)
logger_r2.log_model(encoder_r2)
logger_r2.log_model(ae_r2)
"""))

cells.append(code("""hist_r2 = ar.train_curriculum_coded(
    ae_r2, encoder_r2, k_bits=K_BITS, n_sym=n_sym_r2,
    a_sat=A_SAT, batch_size=BATCH_SIZE,
    logger=logger_r2, verbose=0, use_custom_loss=False)
print("\\n✓ 4→1 done")
"""))

# ─── Training history ───
cells.append(md("""## 6. Training History
"""))

cells.append(code("""fig = ar.plot_history(hist_r03)
fig.axes[0].set_title('Loss — 4→6 (R≈1/3)', fontweight='bold')
fig.axes[1].set_title('Bit Acc — 4→6', fontweight='bold')
plt.show()

fig = ar.plot_history(hist_r05)
fig.axes[0].set_title('Loss — 4→4 (R=1/2)', fontweight='bold')
fig.axes[1].set_title('Bit Acc — 4→4', fontweight='bold')
plt.show()

fig = ar.plot_history(hist_r1)
fig.axes[0].set_title('Loss — 4→2 (R=1)', fontweight='bold')
fig.axes[1].set_title('Bit Acc — 4→2', fontweight='bold')
plt.show()

fig = ar.plot_history(hist_r2)
fig.axes[0].set_title('Loss — 4→1 (R=2)', fontweight='bold')
fig.axes[1].set_title('Bit Acc — 4→1', fontweight='bold')
plt.show()
"""))

# ─── Constellation ───
cells.append(md("""## 7. Learned Coded Constellations
16 codewords mapped to N IQ symbols each.
"""))

cells.append(code("""fig = ar.plot_coded_constellation(
    encoder_r03, K_BITS, n_sym_r03, A_SAT, P_RAPP, ALPHA_PM, BETA_PM)
plt.show()

fig = ar.plot_coded_constellation(
    encoder_r05, K_BITS, n_sym_r05, A_SAT, P_RAPP, ALPHA_PM, BETA_PM)
plt.show()

fig = ar.plot_coded_constellation(
    encoder_r1, K_BITS, n_sym_r1, A_SAT, P_RAPP, ALPHA_PM, BETA_PM)
plt.show()

fig = ar.plot_coded_constellation(
    encoder_r2, K_BITS, n_sym_r2, A_SAT, P_RAPP, ALPHA_PM, BETA_PM)
plt.show()
"""))

# ─── Codeword distances ───
cells.append(md("""## 8. Codeword Distance Analysis
"""))

cells.append(code("""fig = ar.plot_codeword_distances(encoder_r03, K_BITS, n_sym_r03)
plt.show()

fig = ar.plot_codeword_distances(encoder_r05, K_BITS, n_sym_r05)
plt.show()

fig = ar.plot_codeword_distances(encoder_r1, K_BITS, n_sym_r1)
plt.show()

fig = ar.plot_codeword_distances(encoder_r2, K_BITS, n_sym_r2)
plt.show()
"""))

# ─── BER vs SNR ───
cells.append(md("""## 9. BER vs SNR — All Rates vs Baselines
CFO = 0.02
"""))

cells.append(code("""snr_arr = np.arange(0, 26, 2)
CFO_TEST = 0.02

print("Eval 4→6...")
ber_r03 = ar.eval_coded_ber(encoder_r03, ae_r03, snr_arr, CFO_TEST,
                             K_BITS, N=200_000, trials=3)
print("\\nEval 4→4...")
ber_r05 = ar.eval_coded_ber(encoder_r05, ae_r05, snr_arr, CFO_TEST,
                             K_BITS, N=200_000, trials=3)
print("\\nEval 4→2...")
ber_r1 = ar.eval_coded_ber(encoder_r1, ae_r1, snr_arr, CFO_TEST,
                            K_BITS, N=200_000, trials=3)
print("\\nEval 4→1...")
ber_r2 = ar.eval_coded_ber(encoder_r2, ae_r2, snr_arr, CFO_TEST,
                            K_BITS, N=200_000, trials=3)

print("\\nBaselines...")
ber_qam = ar.qam16_ber_uncoded(
    snr_arr, 0.0, a_sat=A_SAT, p_rapp=P_RAPP,
    alpha_pm=ALPHA_PM, beta_pm=BETA_PM)
ber_nocomp = ar.qam16_ber_cfo_nocomp(
    snr_arr, CFO_TEST, n_sym=4,
    a_sat=A_SAT, p_rapp=P_RAPP, alpha_pm=ALPHA_PM, beta_pm=BETA_PM)
print("✓")
"""))

cells.append(code("""results = {
    'ANN 4→6 (R≈1/3)': ber_r03,
    'ANN 4→4 (R=1/2)': ber_r05,
    'ANN 4→2 (R=1)':   ber_r1,
    'ANN 4→1 (R=2)':   ber_r2,
    '16 QAM perfect CFO comp': ber_qam,
    '16QAM+CFO NoComp':  ber_nocomp,
}
fig = ar.plot_ber_comparison(snr_arr, results,
    title=f'\\n(k=4, PA+CFO={CFO_TEST}+AWGN)')
plt.show()
"""))

# ─── Coding gain ───
cells.append(md("""## 10. Coding Gain Analysis
"""))

cells.append(code("""coded = {
    'ANN 4→6 (R≈1/3)': ber_r03,
    'ANN 4→4 (R=1/2)': ber_r05,
    'ANN 4→2 (R=1)':   ber_r1,
    'ANN 4→1 (R=2)':   ber_r2,
}
fig = ar.plot_coding_gain_analysis(snr_arr, ber_qam, coded)
plt.show()
"""))

# ─── BER vs CFO ───
cells.append(md("""## 11. BER vs CFO
"""))

cells.append(code("""cfo_arr = np.linspace(0, 0.05, 11)
SNR_TEST = 15

print(f"BER vs CFO @ SNR={SNR_TEST} dB")
ber_cfo_r03 = ar.eval_coded_ber_vs_cfo(
    encoder_r03, ae_r03, SNR_TEST, cfo_arr, K_BITS, N=200_000, trials=3)
ber_cfo_r05 = ar.eval_coded_ber_vs_cfo(
    encoder_r05, ae_r05, SNR_TEST, cfo_arr, K_BITS, N=200_000, trials=3)
ber_cfo_r1 = ar.eval_coded_ber_vs_cfo(
    encoder_r1, ae_r1, SNR_TEST, cfo_arr, K_BITS, N=200_000, trials=3)
ber_cfo_r2 = ar.eval_coded_ber_vs_cfo(
    encoder_r2, ae_r2, SNR_TEST, cfo_arr, K_BITS, N=200_000, trials=3)

ber_cfo_nc = np.array([max(ar.qam16_ber_cfo_nocomp(
    np.array([SNR_TEST]), c, n_sym=4,
    a_sat=A_SAT, p_rapp=P_RAPP,
    alpha_pm=ALPHA_PM, beta_pm=BETA_PM)[0], 1e-6) for c in cfo_arr])
print("✓")
"""))

cells.append(code("""results_cfo = {
    'ANN 4→6 (R≈1/3)': ber_cfo_r03,
    'ANN 4→4 (R=1/2)': ber_cfo_r05,
    'ANN 4→2 (R=1)':   ber_cfo_r1,
    'ANN 4→1 (R=2)':   ber_cfo_r2,
    '16QAM+CFO NoComp':  ber_cfo_nc,
}
fig = ar.plot_ber_vs_cfo(cfo_arr, results_cfo, snr_val=SNR_TEST)
plt.show()
"""))

# ─── Rate comparison ───
cells.append(md("""## 12. Rate Comparison
"""))

cells.append(code("""rate_results = {
    f'ANN 4→{n_sym_r03} (R≈1/3)': ber_r03,
    f'ANN 4→{n_sym_r05} (R=1/2)': ber_r05,
    f'ANN 4→{n_sym_r1} (R=1)':    ber_r1,
    f'ANN 4→{n_sym_r2} (R=2)':    ber_r2,
}
fig = ar.plot_rate_comparison(snr_arr, rate_results, cfo_val=CFO_TEST)
plt.show()
"""))

# ─── Summary ───
cells.append(md("""## 13. Summary

| Config | Symbols | Rate | Noise Resilience |
|--------|---------|------|-----------------|
| 4→1 | 1 | 2.00 | Worst (max compression) |
| 4→2 | 2 | 1.00 | Baseline |
| 4→4 | 4 | 0.50 | Good |
| 4→6 | 6 | 0.33 | Best |

- **Lower rate** = more symbols = more redundancy = better BER
- **4→1**: Nén tối đa, 4 bit vào 1 symbol IQ — tương đương 16-QAM
- ANN learns to spread info across extra symbols (neural FEC)
- Jointly handles PA + CFO — no separate estimation
"""))

cells.append(code("""try:
    logger_r03.log_finish()
    logger_r05.log_finish()
    logger_r1.log_finish()
    logger_r2.log_finish()
except: pass
print("Demo3 Complete!")
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

path = os.path.join(os.path.dirname(__file__), 'Demo3_Adaptive_Rate.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"OK Generated {path}")
