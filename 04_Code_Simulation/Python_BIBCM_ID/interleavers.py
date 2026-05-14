"""
interleavers.py  –  Algebraic & random interleavers for BIBCM-ID
================================================================
Converted from  algebInterleaver.m  and  mat2perm.m

All functions return **0-indexed** permutation arrays (Python convention).
The MATLAB originals use 1-indexed arrays.
"""

import numpy as np
import os
from typing import Optional


def algebInterleaver(pr: np.ndarray, add: np.ndarray,
                     Np: int, m: int, n: int,
                     interleaver_type: int = 0) -> np.ndarray:
    """
    Generalised algebraic block interleaver.

    Parameters
    ----------
    pr   : array of primitive-root-like multipliers, length J
    add  : array of additive offsets, length J
    Np   : total interleaver length
    n    : codeword length  (= codelen)
    m    : modulation degree (= bps)
    interleaver_type : 0 = overall, 1 = inline

    Returns
    -------
    alp : 0-indexed permutation array of length Np
    """
    M_val = Np // n
    N_val = Np // m
    alp = np.arange(Np, dtype=int)

    if interleaver_type == 0:
        J, I = n, M_val
    else:  # type == 1
        J, I = m, N_val

    for j in range(J):
        for i in range(I):
            if interleaver_type == 0:
                src = i * n + j                                  # 0-indexed
                dst = ((pr[j] * i + add[j]) % I) + j * I       # 0-indexed
                alp[dst] = src
            else:  # inline
                if j + (i + 1) * m <= Np:  # guard bounds
                    dst_idx = j + (i + 1) * m                  # MATLAB: j + i*m (0-idx)
                    src_idx = j * I + ((pr[j] * (i + 1) + add[j]) % I)
                    if dst_idx < Np and src_idx < Np:
                        alp[dst_idx] = src_idx
    return alp


def mat2perm(H: np.ndarray, bps: int) -> np.ndarray:
    """
    Convert a parity-check (or connection) matrix to a permutation.

    Parameters
    ----------
    H   : parity-check matrix, shape (rows, cols)
    bps : bits per symbol

    Returns
    -------
    alp : 0-indexed permutation array
    """
    matrix = H.T  # MATLAB: matrix = H'
    Ns, Ncw = matrix.shape
    m = bps
    Ncb = Ns * m
    n = Ncb // Ncw
    alp = np.arange(Ncb, dtype=int)
    T = np.zeros((Ns, Ncw), dtype=int)

    for i in range(Ncw):
        for j in range(Ns):
            if matrix[j, i] > 0:
                T[j, i] += matrix[j, i]
                # 0-indexed conversion of MATLAB:
                #   alp((j-1)*m + sum(T(j,:))) = (i-1)*n + sum(T(:,i))
                src = j * m + int(np.sum(T[j, :])) - 1
                dst = i * n + int(np.sum(T[:, i])) - 1
                if 0 <= src < Ncb and 0 <= dst < Ncb:
                    alp[src] = dst
    return alp


def load_interleaver(mat_filename: str,
                     data_dir: Optional[str] = None) -> np.ndarray:
    """
    Load a pre-computed interleaver permutation from a MATLAB .mat file.

    Returns a **0-indexed** permutation array.
    """
    import scipy.io

    if data_dir is None:
        from .config import MATLAB_DATA_DIR
        data_dir = MATLAB_DATA_DIR

    path = os.path.join(data_dir, mat_filename)
    data = scipy.io.loadmat(path)
    alpha = data['alpha'].flatten().astype(int)
    # Convert from 1-indexed (MATLAB) to 0-indexed (Python)
    alpha = alpha - 1
    return alpha


def random_interleaver(length: int) -> np.ndarray:
    """Generate a random permutation of given length (0-indexed)."""
    return np.random.permutation(length)
