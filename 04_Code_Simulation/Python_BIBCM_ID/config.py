"""
config.py  –  System configuration for BIBCM-ID simulations
============================================================
Centralises all parameters so that both classic simulation scripts
and future DL-based replacements share the same settings.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os

# ---------------------------------------------------------------------------
# Path to the original MATLAB .mat interleaver files
# ---------------------------------------------------------------------------
MATLAB_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "BIBCM-ID"
)


# ---------------------------------------------------------------------------
# Convolutional code parameters  (RSC rate-1/2, constraint length 3)
# ---------------------------------------------------------------------------
@dataclass
class ConvCodeConfig:
    """RSC convolutional code configuration."""
    k: int = 3              # constraint length
    g1_octal: int = 7       # feedback polynomial (octal)
    g2_octal: int = 5       # feedforward polynomial (octal)
    code_rate: float = 1 / 2

    @property
    def g_matrix(self) -> np.ndarray:
        """Generator matrix in binary, shape (n, k). Row 0 = feedback."""
        g = np.zeros((2, self.k), dtype=int)
        for j in range(self.k):
            g[0, self.k - 1 - j] = (self.g1_octal >> j) & 1
            g[1, self.k - 1 - j] = (self.g2_octal >> j) & 1
        return g


# ---------------------------------------------------------------------------
# Hamming / Extended-Hamming code parameters
# ---------------------------------------------------------------------------
@dataclass
class HammingCodeConfig:
    """Extended Hamming code configuration."""
    m: int = 3              # Hamming parameter → (2^m - 1, 2^m - 1 - m)

    @property
    def n(self) -> int:
        """Codeword length (extended Hamming → 2^m)."""
        return 2 ** self.m

    @property
    def k_info(self) -> int:
        """Information bits per codeword."""
        return 2 ** self.m - self.m - 1

    @property
    def code_rate(self) -> float:
        return self.k_info / self.n


# ---------------------------------------------------------------------------
# Modulation parameters
# ---------------------------------------------------------------------------
@dataclass
class ModulationConfig:
    """QAM modulation settings."""
    M: int = 16             # constellation size
    # BIBCM-ID optimised mapping rule (1-indexed, converted to 0-indexed in code)
    maprule: Optional[List[int]] = None
    Es: float = 10.0        # reference energy for normalisation

    def __post_init__(self):
        if self.maprule is None:
            # Default: MSEW-optimised mapping from sim_conv_bicmid.m
            self.maprule = [13, 6, 7, 16, 3, 12, 14, 5,
                            8, 15, 9, 2, 10, 1, 4, 11]

    @property
    def bps(self) -> int:
        """Bits per symbol."""
        return int(np.log2(self.M))


# ---------------------------------------------------------------------------
# Alternative mapping rules (for switching)
# ---------------------------------------------------------------------------
MAPRULES = {
    "MSEW_conv": [13, 6, 7, 16, 3, 12, 14, 5, 8, 15, 9, 2, 10, 1, 4, 11],
    "MSEW_ham":  [11, 2, 1, 12, 4, 9, 10, 3, 5, 16, 15, 6, 14, 7, 8, 13],
    "Gray_LTE":  [7, 8, 3, 4, 6, 5, 2, 1, 11, 12, 15, 16, 10, 9, 14, 13],
    "SP":        list(range(1, 17)),  # set partitioning (natural)
    "Alt1":      [11, 2, 5, 16, 1, 12, 15, 6, 13, 8, 3, 10, 7, 14, 9, 4],
}


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Monte-Carlo simulation settings."""
    max_iterations: int = 10           # BICM-ID iterations
    scaling_factor: float = 0.85       # extrinsic scaling factor
    dB_range: Optional[List[float]] = None
    max_block_errors_default: int = 50
    max_blocks: int = 800_000_000      # ceiling

    # Interleaver .mat file to load
    interleaver_mat: str = "BIBCM-ID_4096Algeb.mat"

    def __post_init__(self):
        if self.dB_range is None:
            self.dB_range = list(range(0, 4)) + list(range(4, 6))

    def max_block_errors_for_snr(self, snr_dB: float) -> int:
        """Adaptive block-error budget per SNR point."""
        if snr_dB <= 2:
            return 150
        elif snr_dB <= 3:
            return 100
        elif snr_dB <= 4:
            return 70
        elif snr_dB <= 5:
            return 50
        else:
            return 20
