"""
utils.py  –  Utility functions for BIBCM-ID
============================================
- Girth analysis (girth8.m)
- BER plotting
- MAT file loading helpers
- Logging
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from datetime import datetime


# ====================================================================
# 1.  Girth Analysis  (port of girth8.m)
# ====================================================================

def girth_analysis(H: np.ndarray):
    """
    Compute the number of cycles of length 4, 6, and 8 in the
    Tanner graph defined by the parity-check matrix H.

    Parameters
    ----------
    H : binary parity-check matrix, shape (n_checks, n_code)

    Returns
    -------
    num_4 : number of 4-cycles
    num_6 : number of 6-cycles
    num_8 : number of 8-cycles
    """
    E1 = H @ H.T
    np.fill_diagonal(E1, 0)

    E2 = E1 @ E1
    np.fill_diagonal(E2, 0)

    num_4 = np.sum(E1 * (E1 - 1)) // 4
    num_6 = np.sum(E2 * E1) // 6
    num_8 = np.sum(E2 * (E2 - 1)) // 8

    return int(num_4), int(num_6), int(num_8)


# ====================================================================
# 2.  BER Plotting
# ====================================================================

def plot_ber(dB_values: np.ndarray,
             ber_dict: Dict[str, np.ndarray],
             title: str = 'BER vs Eb/N0',
             xlabel: str = 'Eb/N0 (dB)',
             ylabel: str = 'BER',
             save_path: Optional[str] = None,
             figsize: tuple = (10, 7)):
    """
    Plot BER curves (semilogy) for multiple configurations.

    Parameters
    ----------
    dB_values : array of SNR values
    ber_dict  : dict mapping label → BER array
    title     : plot title
    save_path : if given, save figure to this path
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })

    fig, ax = plt.subplots(figsize=figsize)

    styles = [
        ('o-', '#E63946', 2.5), ('s-', '#2A9D8F', 2.5),
        ('^--', '#457B9D', 2.0), ('D--', '#E9C46A', 2.0),
        ('v-', '#264653', 2.0), ('P:', '#9B59B6', 2.0),
        ('*-', '#E67E22', 2.0), ('h--', '#1ABC9C', 2.0),
    ]

    for idx, (name, ber) in enumerate(ber_dict.items()):
        st = styles[idx % len(styles)]
        valid = np.where(ber > 0)[0]
        if len(valid) > 0:
            ax.semilogy(dB_values[valid], ber[valid],
                        st[0], color=st[1], lw=st[2], ms=8,
                        label=name, mfc='white', mew=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-8, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_ber_iterations(dB_values: np.ndarray,
                        ber_matrix: np.ndarray,
                        iterations_to_plot: Optional[List[int]] = None,
                        title: str = 'BER vs Eb/N0 per Iteration',
                        save_path: Optional[str] = None):
    """
    Plot BER for different iteration counts.

    Parameters
    ----------
    dB_values  : array of SNR values
    ber_matrix : shape (max_iter, num_snr)
    iterations_to_plot : list of iteration indices to plot (0-indexed)
    """
    if iterations_to_plot is None:
        iterations_to_plot = list(range(ber_matrix.shape[0]))

    ber_dict = {}
    for it in iterations_to_plot:
        if it < ber_matrix.shape[0]:
            ber_dict[f'Iteration {it + 1}'] = ber_matrix[it, :]

    return plot_ber(dB_values, ber_dict, title=title, save_path=save_path)


# ====================================================================
# 3.  MAT File Loading
# ====================================================================

def load_mat_variable(filename: str, varname: str = 'alpha',
                      data_dir: Optional[str] = None):
    """
    Load a variable from a MATLAB .mat file.

    Returns the variable as a numpy array.
    """
    import scipy.io
    import os

    if data_dir is None:
        from .config import MATLAB_DATA_DIR
        data_dir = MATLAB_DATA_DIR

    path = os.path.join(data_dir, filename)
    data = scipy.io.loadmat(path)
    return data[varname]


def list_available_interleavers(data_dir: Optional[str] = None) -> list:
    """List all .mat files in the BIBCM-ID data directory."""
    import os

    if data_dir is None:
        from .config import MATLAB_DATA_DIR
        data_dir = MATLAB_DATA_DIR

    files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    return sorted(files)


# ====================================================================
# 4.  Simulation Logger  (text + structured JSONL)
# ====================================================================

import json as _json
import platform as _platform
import time as _time


class BICMIDLogger:
    """
    Comprehensive logger for BICM-ID simulations.

    Writes two files side-by-side:
      1. ``<name>.log``   – human-readable text log
      2. ``<name>.jsonl``  – one JSON object per line (machine-parseable)

    The JSONL file is designed so that an AI assistant (or any script)
    can later load it with::

        import json
        records = [json.loads(l) for l in open('sim.jsonl')]

    and reconstruct the full simulation history.
    """

    def __init__(self, name: str = 'bicmid_sim',
                 log_dir: Optional[str] = None,
                 console: bool = True):
        """
        Parameters
        ----------
        name     : base name for log files (without extension)
        log_dir  : directory to place log files (default: cwd)
        console  : if True, also print to stdout
        """
        import os
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            base = os.path.join(log_dir, name)
        else:
            base = name

        self.log_path = base + '.log'
        self.jsonl_path = base + '.jsonl'
        self.console = console
        self._start_time = _time.time()

        self._f_log = open(self.log_path, 'w', encoding='utf-8')
        self._f_json = open(self.jsonl_path, 'w', encoding='utf-8')

        # Opening record
        self._write_json({
            'event': 'session_start',
            'name': name,
            'python': _platform.python_version(),
            'platform': _platform.platform(),
            'timestamp': datetime.now().isoformat(),
        })
        self.log('=' * 65)
        self.log(f'  BICM-ID Logger  |  {name}')
        self.log(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.log('=' * 65)

    # ── core I/O ────────────────────────────────────────────────

    def log(self, msg: str = '', to_console: bool = True):
        """Write a human-readable line to .log (+ console)."""
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        if self.console and to_console:
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode('ascii', 'replace').decode())
        self._f_log.write(line + '\n')
        self._f_log.flush()

    def _write_json(self, record: dict):
        """Append one JSON record to .jsonl."""
        record['_ts'] = datetime.now().isoformat()
        self._f_json.write(_json.dumps(record, ensure_ascii=False) + '\n')
        self._f_json.flush()

    # ── structured events ───────────────────────────────────────

    def log_config(self, **kwargs):
        """Log simulation configuration parameters."""
        self._write_json({'event': 'config', **kwargs})
        self.log('  Configuration:')
        for k, v in kwargs.items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:77] + '...'
            self.log(f'    {k} = {val_str}')

    def log_component_test(self, component: str, status: str = 'OK',
                           details: Optional[Dict] = None):
        """Log a component verification result."""
        rec = {'event': 'component_test', 'component': component,
               'status': status}
        if details:
            rec['details'] = details
        self._write_json(rec)
        det_str = f'  ({details})' if details else ''
        self.log(f'  [{status}] {component}{det_str}')

    def log_snr_start(self, snr_dB: float, max_block_errors: int):
        """Log the start of simulation for a new SNR point."""
        self._write_json({
            'event': 'snr_start',
            'snr_dB': snr_dB,
            'max_block_errors': max_block_errors,
        })
        self.log(f'  --- SNR = {snr_dB:.1f} dB  '
                 f'(max_block_errors={max_block_errors}) ---')

    def log_block_progress(self, snr_dB: float, block: int,
                           block_errors: int, max_block_errors: int,
                           current_ber: float):
        """Log progress within an SNR point (called every N blocks)."""
        self._write_json({
            'event': 'block_progress',
            'snr_dB': snr_dB,
            'block': block,
            'block_errors': block_errors,
            'max_block_errors': max_block_errors,
            'current_ber': float(current_ber),
        })
        self.log(f'    Block {block:5d} | '
                 f'BlockErr={block_errors}/{max_block_errors} | '
                 f'BER~{current_ber:.2e}', to_console=False)

    def log_snr_result(self, snr_dB: float, blocks: int,
                       block_errors: int,
                       ber_per_iteration: list,
                       elapsed_s: float):
        """Log final result for one SNR point."""
        self._write_json({
            'event': 'snr_result',
            'snr_dB': snr_dB,
            'blocks': blocks,
            'block_errors': block_errors,
            'ber_per_iteration': [float(b) for b in ber_per_iteration],
            'elapsed_s': round(elapsed_s, 3),
        })
        self.log(f'  SNR={snr_dB:.1f}dB done: {blocks} blocks, '
                 f'{block_errors} block errors, '
                 f'BER(last)={ber_per_iteration[-1]:.4e}, '
                 f'time={elapsed_s:.1f}s')

    def log_simulation_done(self, dB_values, ber_matrix,
                            total_elapsed_s: float):
        """Log the final BER table for all SNR points."""
        max_iter = ber_matrix.shape[0]
        table = {}
        for z, snr in enumerate(dB_values):
            table[f'{snr:.1f}'] = [float(ber_matrix[it, z])
                                   for it in range(max_iter)]
        self._write_json({
            'event': 'simulation_done',
            'dB_values': [float(d) for d in dB_values],
            'ber_table': table,
            'max_iterations': max_iter,
            'total_elapsed_s': round(total_elapsed_s, 3),
        })
        self.log('')
        self.log('=' * 65)
        self.log(f'  Simulation complete  |  {total_elapsed_s:.1f}s total')
        self.log('  BER (last iteration):')
        for z, snr in enumerate(dB_values):
            self.log(f'    {snr:5.1f} dB  ->  {ber_matrix[-1, z]:.4e}')
        self.log('=' * 65)

    def log_custom(self, event_name: str, **kwargs):
        """Log an arbitrary named event with key-value data."""
        self._write_json({'event': event_name, **kwargs})
        parts = '  '.join(f'{k}={v}' for k, v in kwargs.items())
        self.log(f'  [{event_name}] {parts}')

    # ── lifecycle ───────────────────────────────────────────────

    def close(self):
        elapsed = _time.time() - self._start_time
        self._write_json({
            'event': 'session_end',
            'total_elapsed_s': round(elapsed, 3),
        })
        self.log(f'  Session closed ({elapsed:.1f}s)')
        self._f_log.close()
        self._f_json.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Keep the old name as an alias for backward compatibility
SimLogger = BICMIDLogger


def load_log(jsonl_path: str) -> List[dict]:
    """
    Load a .jsonl log file and return a list of event dicts.

    Usage::

        records = load_log('bicmid_sim.jsonl')
        configs = [r for r in records if r['event'] == 'config']
        results = [r for r in records if r['event'] == 'snr_result']
    """
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_json.loads(line))
    return records


def summarize_log(jsonl_path: str) -> str:
    """
    Generate a human-readable summary from a .jsonl log file.
    Useful for quick inspection or feeding to an AI for analysis.
    """
    records = load_log(jsonl_path)
    lines = []
    lines.append('=== BICM-ID Log Summary ===')

    for r in records:
        ev = r.get('event', '')
        if ev == 'session_start':
            lines.append(f"Session: {r.get('name')} @ {r.get('timestamp')}")
            lines.append(f"  Python {r.get('python')}, {r.get('platform')}")
        elif ev == 'config':
            items = {k: v for k, v in r.items()
                     if k not in ('event', '_ts')}
            lines.append(f"Config: {items}")
        elif ev == 'snr_result':
            ber_list = r.get('ber_per_iteration', [])
            lines.append(
                f"  SNR={r['snr_dB']:.1f}dB: "
                f"blocks={r['blocks']}, "
                f"BER(1)={ber_list[0]:.2e}, "
                f"BER({len(ber_list)})={ber_list[-1]:.2e}, "
                f"time={r['elapsed_s']:.1f}s"
            )
        elif ev == 'simulation_done':
            lines.append(f"Total time: {r['total_elapsed_s']:.1f}s")
        elif ev == 'component_test':
            lines.append(f"  [{r['status']}] {r['component']}")

    return '\n'.join(lines)
