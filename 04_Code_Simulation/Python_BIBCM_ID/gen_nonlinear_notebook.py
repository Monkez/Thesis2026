import json, os

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src.strip().split('\n')}

def code(src):
    lines = src.strip().split('\n')
    return {"cell_type":"code","execution_count":None,
            "metadata":{},"outputs":[],"source": lines}

cells = []

# ─── Title ───
cells.append(md("""# Thử nghiệm Kênh truyền Phi tuyến (PA + CFO) trên hệ thống BICM-ID
---
Notebook này thử nghiệm đưa kênh truyền có nhiễu phi tuyến (tương tự như trong `Demo3_Adaptive_Rate.ipynb`) vào hệ thống BICM-ID truyền thống.
"""))

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
"""))

# ─── Section: Channel Definition ───
cells.append(md("""## 1. Định nghĩa kênh truyền PA + CFO + AWGN (Numpy)
"""))

cells.append(code("""def cfopa_channel(tx, N0, a_sat=1.2, p_rapp=3, alpha_pm=0.08, beta_pm=0.0, cfo_norm=0.03):
    \"\"\"
    Mô phỏng kênh truyền PA (Rapp model) + CFO (Carrier Frequency Offset) + AWGN
    Dựa trên mô hình CFOPAChannelCoded trong Demo3_Adaptive_Rate.ipynb.
    \"\"\"
    # PA per symbol (Rapp model)
    r = np.abs(tx)
    theta = np.angle(tx)
    
    ratio = r / a_sat
    g_r = r / ((1.0 + ratio**(2*p_rapp))**(1.0/(2*p_rapp)))
    
    phi_pa = alpha_pm * r**2 / (1.0 + beta_pm * r**2)
    
    pa_tx = g_r * np.exp(1j * (theta + phi_pa))
    
    # CFO rotation
    n_idx = np.arange(len(pa_tx))
    phase = 2.0 * np.pi * cfo_norm * n_idx
    
    cfo_tx = pa_tx * np.exp(1j * phase)
    
    # AWGN
    noise = (np.random.randn(len(cfo_tx)) + 1j * np.random.randn(len(cfo_tx)))
    rx = cfo_tx + np.sqrt(N0 / 2) * noise
    
    return rx

print("Defined cfopa_channel.")
"""))

cells.append(md("""## 2. Visualize Kênh Phi Tuyến
Thử truyền tín hiệu 16-QAM qua kênh để xem chòm sao bị biến dạng thế nào.
"""))

cells.append(code("""from python_code.modulation import build_constellation, modulate_bits, qam16_constellation_matlab

# Build 16-QAM
S = build_constellation(16, [11, 2, 1, 12, 4, 9, 10, 3, 5, 16, 15, 6, 14, 7, 8, 13], 10.0)

np.random.seed(0)
test_bits = np.random.randint(0, 2, 4000)
tx_symbols = modulate_bits(test_bits, S)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cfo_vals = [0.0, 0.005, 0.03]  # CFO values to test
snr_dB = 20
N0 = 1.0 / (10 ** (snr_dB / 10) * 4)

for ax, cfo in zip(axes, cfo_vals):
    rx = cfopa_channel(tx_symbols, N0, a_sat=3.5, p_rapp=3, alpha_pm=0.0, cfo_norm=cfo)
    ax.scatter(rx.real, rx.imag, c='blue', s=8, alpha=0.5, label='Received')
    ax.scatter(S.real, S.imag, c='red', s=100, edgecolors='k', zorder=5, label='Constellation')
    ax.set_title(f'SNR = {snr_dB} dB, CFO = {cfo}', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    ax.legend(fontsize=9)

plt.suptitle('16-QAM qua kênh PA + CFO + AWGN (a_sat=3.5)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()
"""))

# ─── Section: Modified BICM-ID System ───
cells.append(md("""## 3. Override hệ thống BICM-ID để sử dụng Kênh Mới
Ta sẽ kế thừa lớp `BICMIDSystem` và ghi đè hàm channel thành `cfopa_channel`.
"""))

cells.append(code("""from python_code.bicm_id import BICMIDSystem
from python_code.demodulation import softdemod_qam, soft_demapper

class NonlinearBICMIDSystem(BICMIDSystem):
    def __init__(self, cfo_norm=0.005, a_sat=3.5, p_rapp=3, alpha_pm=0.0, **kwargs):
        super().__init__(**kwargs)
        self.cfo_norm = cfo_norm
        self.a_sat = a_sat
        self.p_rapp = p_rapp
        self.alpha_pm = alpha_pm
        
    def _sim_conv_frame(self, N0: float, alpha: np.ndarray):
        num_cb = len(alpha)
        info_len = int(num_cb * self.code_rate)
        n_rate = self.encoder.n
        m = self.encoder.m

        # Generate random info bits
        u = np.random.randint(0, 2, info_len)
        v = self.encoder.encode(u)

        # Interleave
        vv = v[alpha]

        # Modulate
        chan_in = modulate_bits(vv, self.S)

        # === NONLINEAR CHANNEL ===
        chan_out = cfopa_channel(chan_in, N0, 
                                 a_sat=self.a_sat, p_rapp=self.p_rapp, 
                                 alpha_pm=self.alpha_pm, cfo_norm=self.cfo_norm)

        # Soft demodulation (channel metrics)
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

print("Overridden NonlinearBICMIDSystem.")
"""))

# ─── Section: Run Simulation ───
cells.append(md("""## 4. Chạy Simulation với Kênh Phi Tuyến
So sánh kết quả BER khi có CFO và khi không có CFO.
"""))

cells.append(code("""# ╔══════════════════════════════════════════════════════════╗
# ║  🔧 CẤU HÌNH MÔ PHỎNG — KÊNH PHI TUYẾN                ║
# ║  Thay đổi các giá trị bên dưới theo ý bạn rồi Run cell  ║
# ╚══════════════════════════════════════════════════════════╝

NL_SNR_dB         = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]   # Dải SNR (dB)
NL_INTERLEAVER    = "BIBCM-ID_1024Algeb.mat"                    # Bộ giao vòng
NL_MAX_ITER       = 10                                           # Số vòng lặp BICM-ID
NL_SCALING        = 0.85                                         # Hệ số scaling
NL_MAX_BLOCKS     = 10000                                        # Số block tối đa mỗi mức SNR
NL_MAPRULE        = [13, 6, 7, 16, 3, 12, 14, 5,                # Quy tắc ánh xạ chòm sao
                     8, 15, 9, 2, 10, 1, 4, 11]

# ─── Tham số kênh phi tuyến ────────────
NL_CFO            = 0.001                                        # CFO (nhỏ = ít méo pha)
NL_A_SAT          = 5.0                                          # Ngưỡng bão hòa PA
NL_P_RAPP         = 3                                            # Bậc Rapp model
NL_ALPHA_PM       = 0.0                                          # Hệ số AM/PM conversion

print("✅ Cấu hình kênh phi tuyến đã sẵn sàng. Chạy cell tiếp theo để bắt đầu mô phỏng.")
"""))

cells.append(code("""from python_code.interleavers import load_interleaver
from python_code.config import ConvCodeConfig, ModulationConfig
from python_code.modulation import build_constellation, symbol_bit_matrix
from python_code.workers import simulate_chunk_nonlinear_conv
import concurrent.futures
import os
import time
from tqdm.auto import tqdm

mod_cfg = ModulationConfig(M=16, maprule=NL_MAPRULE, Es=10.0)
alpha_conv = load_interleaver(NL_INTERLEAVER) 

def run_simulation(cfo_val, a_sat_val, p_rapp=3, alpha_pm=0.0):
    conv_cfg = ConvCodeConfig()
    g_matrix = conv_cfg.g_matrix
    code_rate = conv_cfg.code_rate
    m_encoder = conv_cfg.k - 1
    n_rate = 2
    
    S = build_constellation(16, mod_cfg.maprule, mod_cfg.Es)
    sym_matrix = symbol_bit_matrix(16)
    
    num_channel_bits = len(alpha_conv)
    info_len = int(num_channel_bits * code_rate)
    
    BER = np.zeros((NL_MAX_ITER, len(NL_SNR_dB)))
    num_workers = os.cpu_count() or 4
    
    print(f"Running Simulation: CFO={cfo_val}, A_sat={a_sat_val}, p_rapp={p_rapp}")
    for z, snr_dB in enumerate(NL_SNR_dB):
        N0_z = 1.0 / (10 ** (snr_dB / 10) * 4 * code_rate)
        max_block_errors = 30 if snr_dB >= 4 else 100
        
        total_bit_errors = np.zeros(NL_MAX_ITER)
        total_block_errors = 0
        total_blocks = 0
        chunk_size = 20 if snr_dB >= 4 else 5
        
        pbar = tqdm(total=NL_MAX_BLOCKS, desc=f"SNR={snr_dB:.0f}dB", leave=True, 
                    bar_format='{desc} | {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            def submit_chunk():
                return executor.submit(
                    simulate_chunk_nonlinear_conv, chunk_size, N0_z, NL_MAX_ITER,
                    NL_SCALING, num_channel_bits, info_len, m_encoder, n_rate,
                    g_matrix, S, sym_matrix, cfo_val, a_sat_val, p_rapp, alpha_pm
                )
            for _ in range(num_workers * 2):
                futures.add(submit_chunk())
            while total_block_errors < max_block_errors and total_blocks < NL_MAX_BLOCKS:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    blocks_done, b_errs, bit_errs = future.result()
                    total_blocks += blocks_done
                    total_block_errors += b_errs
                    total_bit_errors += bit_errs
                    pbar.update(blocks_done)
                    ber_now = total_bit_errors[-1] / (total_blocks * num_channel_bits)
                    pbar.set_postfix({'Err': f'{total_block_errors}/{max_block_errors}', 'BER': f'{ber_now:.2e}'})
                    if total_block_errors < max_block_errors and (total_blocks + len(futures)*chunk_size) < NL_MAX_BLOCKS:
                        futures.add(submit_chunk())
            for f in futures: f.cancel()
        pbar.close()
        for it in range(NL_MAX_ITER):
            BER[it, z] = total_bit_errors[it] / (total_blocks * num_channel_bits)
        
    return NL_SNR_dB, BER

dB_nonlin, BER_nonlin = run_simulation(cfo_val=NL_CFO, a_sat_val=NL_A_SAT, p_rapp=NL_P_RAPP, alpha_pm=NL_ALPHA_PM)
"""))

cells.append(code("""# Plot BER
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(dB_nonlin, BER_nonlin[-1, :], 's-', lw=2, ms=8, label=f'Nonlinear (Iter {NL_MAX_ITER}, CFO={NL_CFO}, Asat={NL_A_SAT})')
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('BER')
ax.set_title('Tác động của kênh PA+CFO lên hiệu năng BICM-ID', fontweight='bold')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim(1e-6, 1)
plt.tight_layout()
plt.show()
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

path = os.path.join(os.path.dirname(__file__), 'Test_Nonlinear_Channel.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"OK - Generated {os.path.basename(path)}")
