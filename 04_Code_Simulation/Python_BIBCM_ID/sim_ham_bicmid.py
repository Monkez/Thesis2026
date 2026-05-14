"""
sim_ham_bicmid.py  –  BICM-ID with Extended Hamming Code
========================================================
Direct port of sim_ham_bicmid.m

Usage:
    python -m python_code.sim_ham_bicmid

Copyright (c) 2020 by Dinh The Cuong (original MATLAB)
Python conversion 2026
"""

import numpy as np
import time
import sys
import os
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_code.config import ModulationConfig, SimulationConfig
from python_code.encoders import HammingEncoder
from python_code.decoders import DualDecoder
from python_code.modulation import (build_constellation, modulate_bits,
                                    symbol_bit_matrix)
from python_code.demodulation import softdemod_qam, soft_demapper
from python_code.channel import awgn_channel
from python_code.interleavers import load_interleaver
from python_code.utils import plot_ber, BICMIDLogger
import concurrent.futures
from python_code.workers import simulate_chunk_ham


def main(dB_range=None, interleaver_mat=None, max_iterations=10,
         scaling_factor=0.85, max_bits_total=100_000_000,
         maprule=None, hamming_m=3, console=True):
    """
    Run BICM-ID simulation with Extended Hamming Code.
    
    Parameters
    ----------
    dB_range : list of float, optional
        SNR values to simulate. Default: [0, 1, 2, 3, 3.5, 4, 4.5, 5]
    interleaver_mat : str, optional
        Interleaver .mat filename. Default: "BIBCM-ID_4096Algeb.mat"
    max_iterations : int, optional
        Number of BICM-ID iterations. Default: 10
    scaling_factor : float, optional
        Extrinsic info scaling factor. Default: 0.85
    max_bits_total : int, optional
        Maximum total bits per SNR point. Default: 10^8
    maprule : list of int, optional
        Constellation mapping rule. Default: MSEW-optimised for Hamming
    hamming_m : int, optional
        Hamming parameter m. Default: 3 -> Extended Hamming (8,4)
    console : bool, optional
        Print results to console. Default: True
    
    Returns
    -------
    BER : ndarray, shape (max_iterations, len(dB_range))
    dB  : ndarray of SNR values
    """
    # ─── Defaults ────────────────────────────────────────────────
    if dB_range is None:
        dB_range = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]
    if interleaver_mat is None:
        interleaver_mat = "BIBCM-ID_4096Algeb.mat"
    if maprule is None:
        maprule = [11, 2, 1, 12, 4, 9, 10, 3,
                   5, 16, 15, 6, 14, 7, 8, 13]

    start_time = time.time()

    # ─── Logger ─────────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    logger = BICMIDLogger('sim_ham_bicmid', log_dir=log_dir, console=console)

    # ─── Configuration ──────────────────────────────────────────
    ham = HammingEncoder(hamming_m)
    mod_cfg = ModulationConfig(M=16, maprule=maprule, Es=10.0)

    H = ham.H
    G = ham.G
    n_code = ham.n_code       # codeword length (cols of H)
    k_info = ham.k_info       # info bits
    n_checks = ham.n_checks   # parity checks (rows of H)
    code_rate = ham.code_rate
    bps = mod_cfg.bps

    # ─── Constellation ──────────────────────────────────────────
    S = build_constellation(mod_cfg.M, mod_cfg.maprule, mod_cfg.Es)
    sym_matrix = symbol_bit_matrix(mod_cfg.M)

    # ─── Interleaver ────────────────────────────────────────────
    alpha = load_interleaver(interleaver_mat)
    num_channel_bits = len(alpha)
    frame_len = num_channel_bits // n_code

    # ─── Decoder ────────────────────────────────────────────────
    dual_dec = DualDecoder(H)

    # ─── Simulation parameters ──────────────────────────────────
    max_iter = max_iterations
    max_blocks_total = int(np.ceil(max_bits_total / num_channel_bits))

    logger.log_config(
        code_type='extended_hamming',
        n_code=n_code,
        k_info=k_info,
        code_rate=code_rate,
        M=mod_cfg.M,
        bps=bps,
        maprule=mod_cfg.maprule,
        Es=mod_cfg.Es,
        interleaver_mat=interleaver_mat,
        interleaver_length=num_channel_bits,
        frame_len=frame_len,
        max_iterations=max_iter,
        scaling_factor=scaling_factor,
        dB_range=[float(d) for d in dB_range],
    )

    # ─── SNR Setup ──────────────────────────────────────────────
    dB = np.array(dB_range, dtype=float)
    SNR_lin = 10 ** (dB / 10) * bps
    N0_uncoded = 1.0 / SNR_lin
    N0 = N0_uncoded / code_rate

    BER = np.zeros((max_iter, len(dB)))

    # ─── Main Simulation Loop (Chunk-based Parallel) ──────────────
    num_workers = os.cpu_count() or 4
    
    for z in range(len(dB)):
        snr_start_time = time.time()
        
        # Adaptive block error budget
        if dB[z] <= 2:
            max_block_errors = 150
        elif dB[z] <= 3:
            max_block_errors = 100
        elif dB[z] <= 4:
            max_block_errors = 70
        elif dB[z] <= 4.5:
            max_block_errors = 50
        else:
            max_block_errors = 30
            
        logger.log_snr_start(dB[z], max_block_errors)
        
        total_bit_errors = np.zeros(max_iter)
        total_block_errors = 0
        total_blocks = 0
        
        chunk_size = 20 if dB[z] >= 4.0 else 5
        
        pbar = tqdm(total=max_blocks_total, desc=f"SNR={dB[z]:.1f}dB", leave=True, 
                    bar_format='{desc} | {percentage:3.0f}%|{bar}| Block: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            
            def submit_chunk():
                return executor.submit(
                    simulate_chunk_ham, chunk_size, N0[z], max_iter,
                    scaling_factor, num_channel_bits, frame_len, k_info, n_code,
                    H, G, S, sym_matrix, alpha
                )
            
            for _ in range(num_workers * 2):
                futures.add(submit_chunk())
                
            while total_block_errors < max_block_errors and total_blocks < max_blocks_total:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done:
                    blocks_done, b_errs, bit_errs = future.result()
                    total_blocks += blocks_done
                    total_block_errors += b_errs
                    total_bit_errors += bit_errs
                    
                    pbar.update(blocks_done)
                    current_ber = total_bit_errors[max_iter - 1] / (total_blocks * num_channel_bits)
                    pbar.set_postfix({'Err': f'{total_block_errors}/{max_block_errors}', 'BER': f'{current_ber:.2e}'})
                    
                    if total_block_errors < max_block_errors and (total_blocks + len(futures)*chunk_size) < max_blocks_total:
                        futures.add(submit_chunk())
                        
            for future in futures:
                future.cancel()
                
        pbar.close()
        
        snr_elapsed = time.time() - snr_start_time
        ber_per_iter = []
        for iteration in range(max_iter):
            BER[iteration, z] = total_bit_errors[iteration] / (total_blocks * num_channel_bits)
            ber_per_iter.append(BER[iteration, z])

        logger.log_snr_result(dB[z], total_blocks, total_block_errors,
                              ber_per_iter, snr_elapsed)

    # ─── Results ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.log_simulation_done(dB, BER, elapsed)
    logger.close()

    # Plot only the final iteration
    ber_dict = {f'Iter {max_iter}': BER[max_iter - 1, :]}
    fig = plot_ber(dB, ber_dict,
                   title=f'BICM-ID BER -- Extended Hamming ({n_code},{k_info})',
                   save_path='ber_ham_bicmid.png')
    import matplotlib.pyplot as plt
    plt.show()

    return BER, dB


if __name__ == '__main__':
    main()
