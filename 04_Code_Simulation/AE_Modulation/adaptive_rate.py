"""
Demo3: ANN Channel Coding — Adaptive Rate Modulation
=====================================================
ANN autoencoder that outputs MORE symbols than input bits,
analogous to Forward Error Correction (FEC) channel coding.

Key concept:
  Traditional system: k bits → M-QAM → 1 symbol
  This system:        k bits → Encoder → N symbols (N > k/log2(M))

  Rate R = k/(N·log2(M_out)) < 1  (redundancy!)

  The encoder *learns* to spread information across extra symbols
  in a way that maximizes noise resilience — this is neural FEC.

Architecture:
  Encoder: (B, k_bits) → Dense layers → (B, N_sym, 2)   [real IQ]
           where N_sym > k_bits/log2(M), creating coded redundancy
  Channel: PA → CFO → AWGN
  Decoder: (B, N_sym, 2) + SNR → (B, k_bits)  [bit probabilities]

  We directly encode/decode raw bits (not one-hot symbols).
  Each output neuron is sigmoid → P(bit_i = 1).

Rates supported:
  k=4 bits → N=4 symbols → R=0.50 (2× redundancy, like rate-1/2 code)
  k=4 bits → N=3 symbols → R=0.67 (1.5× redundancy)
  k=4 bits → N=2 symbols → R=1.00 (no redundancy, baseline)
  k=8 bits → N=6 symbols → R=0.67
  k=8 bits → N=8 symbols → R=0.50
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Global Matplotlib styling for papers
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})
import matplotlib
matplotlib.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})
from datetime import datetime


# ============================================================
# 0. Logger
# ============================================================

class TrainLogger:
    def __init__(self, log_file='train_rate.txt', append=False):
        self.log_file = log_file
        self.f = open(log_file, 'a' if append else 'w', encoding='utf-8')
        self.log('=' * 70)
        self.log(f'  Demo3: ANN Channel Coding — Adaptive Rate')
        self.log(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.log('=' * 70)

    def log(self, msg=''):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        self.f.write(line + '\n')
        self.f.flush()

    def log_params(self, k_bits, n_sym, a_sat, p_rapp, alpha_pm, beta_pm,
                    cfo_max, batch_size):
        rate = k_bits / (n_sym * 2)  # 2 = dimensions per symbol (IQ)
        self.log(f'  k_bits={k_bits}, N_sym={n_sym}')
        self.log(f'  Code rate R = {k_bits}/{n_sym}×2 = {rate:.3f}')
        self.log(f'  Spectral efficiency = {rate*2:.3f} bits/symbol/dim')
        self.log(f'  A_sat={a_sat}, p_rapp={p_rapp}')
        self.log(f'  alpha_pm={alpha_pm}, beta_pm={beta_pm}')
        self.log(f'  CFO_max={cfo_max} (normalized Δf·Ts)')
        self.log(f'  batch_size={batch_size}')

    def log_model(self, model):
        self.log(f'  {model.name}: {model.count_params():,} params')

    def log_finish(self):
        self.log('')
        self.log('=' * 70)
        self.log(f'  Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.log('=' * 70)
        self.f.close()


def init_logger(log_file='train_rate.txt', append=False):
    return TrainLogger(log_file, append)


class _EpochLogCB(keras.callbacks.Callback):
    def __init__(self, logger): super().__init__(); self.logger = logger
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.logger.log(
                f'  Ep {epoch+1:3d} | '
                f'loss={logs.get("loss",0):.4f} '
                f'acc={logs.get("bit_accuracy",0):.4f} | '
                f'v_loss={logs.get("val_loss",0):.4f} '
                f'v_acc={logs.get("val_bit_accuracy",0):.4f} | '
                f'lr={float(self.model.optimizer.learning_rate):.2e}')


# ============================================================
# 1. Custom Layers
# ============================================================

class PowerNormCoded(layers.Layer):
    """Normalize average power across all N_sym coded symbols.
    Input: (batch, N_sym, 2). Output: same shape, avg power = 1."""
    def call(self, x):
        mean_power = tf.reduce_mean(tf.reduce_sum(x**2, axis=-1))
        return x / tf.sqrt(mean_power + 1e-8)


class CFOPAChannelCoded(layers.Layer):
    """Rapp PA + CFO + AWGN for coded multi-symbol blocks.

    Signal: x[n] → PA(x[n]) → ·exp(j·2π·Δf·n) → +noise → y[n]
    Inputs: [tx(B,N,2), snr_db(B,1), cfo_norm(B,1)]
    Output: (B, N, 2)
    """
    def __init__(self, a_sat=1.2, p_rapp=3,
                 alpha_pm=0.08, beta_pm=0.0, n_sym=4, **kwargs):
        super().__init__(**kwargs)
        self.a_sat = a_sat
        self.p_rapp = p_rapp
        self.alpha_pm = alpha_pm
        self.beta_pm = beta_pm
        self.n_sym = n_sym

    def call(self, inputs, training=None):
        tx, snr_db, cfo_norm = inputs
        B = tf.shape(tx)[0]

        # PA per symbol (Rapp model)
        x_I, x_Q = tx[:, :, 0:1], tx[:, :, 1:2]
        r = tf.sqrt(x_I**2 + x_Q**2 + 1e-8)
        theta = tf.atan2(x_Q, x_I)
        ratio = r / self.a_sat
        g_r = r / tf.pow(1.0 + tf.pow(ratio, 2*self.p_rapp),
                          1.0 / (2*self.p_rapp))
        phi_pa = self.alpha_pm * r**2 / (1.0 + self.beta_pm * r**2)
        pa_I = g_r * tf.cos(theta + phi_pa)
        pa_Q = g_r * tf.sin(theta + phi_pa)

        # CFO rotation: phase = 2π·Δf·n
        n_idx = tf.cast(tf.range(self.n_sym), tf.float32)
        n_idx = tf.reshape(n_idx, (1, self.n_sym, 1))
        cfo_exp = tf.reshape(cfo_norm, (-1, 1, 1))
        phase = 2.0 * 3.141592653589793 * cfo_exp * n_idx

        y_I = pa_I * tf.cos(phase) - pa_Q * tf.sin(phase)
        y_Q = pa_I * tf.sin(phase) + pa_Q * tf.cos(phase)

        # AWGN
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigma = tf.sqrt(1.0 / (2.0 * snr_lin + 1e-8))
        sigma = tf.reshape(sigma, (-1, 1, 1))
        y_I += tf.random.normal(tf.shape(y_I)) * sigma
        y_Q += tf.random.normal(tf.shape(y_Q)) * sigma

        return tf.concat([y_I, y_Q], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'a_sat': self.a_sat, 'p_rapp': self.p_rapp,
                    'alpha_pm': self.alpha_pm, 'beta_pm': self.beta_pm,
                    'n_sym': self.n_sym})
        return cfg


# ============================================================
# 2. Build ANN Channel Coding Autoencoder
# ============================================================

def bit_accuracy(y_true, y_pred):
    """Per-bit accuracy. Both (batch, k_bits), pred is sigmoid prob."""
    pred_bits = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(
        tf.equal(y_true, pred_bits), tf.float32))


def build_coded_autoencoder(k_bits=4, n_sym=4, n_channel=2,
                             a_sat=1.2, p_rapp=3,
                             alpha_pm=0.08, beta_pm=0.0,
                             cfo_max=0.03, dropout_rate=0.05):
    """
    ANN Channel Coding Autoencoder.

    Encoder: (B, k_bits) → MLP → (B, N_sym, 2)
      Maps k raw bits into N_sym IQ symbols.
      N_sym > ceil(k_bits / log2(M)) → redundancy!

    Channel: PA → CFO → AWGN

    Decoder: (B, N_sym, 2) + SNR → (B, k_bits)  [sigmoid]
      Jointly decodes all coded symbols back to k bits.

    Rate R = k_bits / (N_sym × 2) [bits per real dimension]
    """
    rate = k_bits / (n_sym * n_channel)

    # --- Inputs ---
    bits_in = layers.Input(shape=(k_bits,), name='bits_input')
    snr_in  = layers.Input(shape=(1,), name='snr_input')
    cfo_in  = layers.Input(shape=(1,), name='cfo_input')

    # ---- Encoder: bits → coded IQ symbols ----
    #   Light model with BN for gradient stability through PA channel
    enc_dim = 64
    e = layers.Dense(enc_dim, activation='relu', name='enc_fc1')(bits_in)
    e = layers.BatchNormalization(name='enc_bn1')(e)
    e = layers.Dense(enc_dim, activation='relu', name='enc_fc2')(e)
    e = layers.BatchNormalization(name='enc_bn2')(e)
    # tanh bounds output before power norm, prevents gradient explosion in PA
    e = layers.Dense(n_sym * n_channel, activation='tanh', name='enc_iq')(e)
    e = layers.Reshape((n_sym, n_channel), name='enc_reshape')(e)
    tx = PowerNormCoded(name='power_norm')(e)

    encoder = Model(bits_in, tx, name='Encoder')

    # ---- Channel ----
    rx = CFOPAChannelCoded(
        a_sat=a_sat, p_rapp=p_rapp,
        alpha_pm=alpha_pm, beta_pm=beta_pm, n_sym=n_sym,
        name='cfo_pa_channel'
    )([tx, snr_in, cfo_in])

    # ---- Decoder: received IQ → k bits ----
    rx_flat = layers.Flatten(name='rx_flatten')(rx)

    # SNR conditioning
    snr_norm = layers.Lambda(
        lambda s: tf.clip_by_value((s - 12.5) / 12.5, -1.5, 1.5),
        name='snr_norm')(snr_in)
    dec_in = layers.Concatenate(name='dec_concat1')([rx_flat, snr_norm])

    d = layers.Dense(128, activation='relu', name='dec_fc1')(dec_in)
    d = layers.BatchNormalization(name='dec_bn1')(d)
    d = layers.Dense(128, activation='relu', name='dec_fc2')(d)
    d = layers.BatchNormalization(name='dec_bn2')(d)
    d = layers.Dense(64, activation='relu', name='dec_fc3')(d)

    # Output: k_bits probabilities via sigmoid
    bits_out = layers.Dense(k_bits, activation='sigmoid',
                            name='bits_output')(d)

    autoencoder = Model([bits_in, snr_in, cfo_in], bits_out,
                        name='CodedAutoencoder')

    return encoder, autoencoder


# ============================================================
# 3. Custom Loss
# ============================================================

def make_coded_loss(encoder, k_bits=4, n_sym=4, a_sat=1.0,
                    lambda_power=0.005):
    """
    Binary crossentropy + light power penalty.
    Spread penalty removed — it causes NaN with small models.
    For k=4 (16 codewords) the optimizer naturally separates codewords.
    """
    n_codes = 2 ** k_bits
    all_bits = np.array(
        [[(i >> b) & 1 for b in range(k_bits)]
         for i in range(n_codes)],
        dtype=np.float32
    )
    all_bits_tf = tf.constant(all_bits)

    def _loss(y_true, y_pred):
        # Clipped BCE
        y_pred_clip = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -tf.reduce_mean(
            y_true * tf.math.log(y_pred_clip) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred_clip),
            axis=-1)

        # Light power penalty
        codewords = encoder(all_bits_tf, training=False)
        amps = tf.sqrt(tf.reduce_sum(codewords**2, axis=-1) + 1e-8)
        power_pen = tf.reduce_mean(
            tf.square(tf.maximum(amps - a_sat * 0.95, 0.0)))

        return bce + lambda_power * power_pen

    return _loss


# ============================================================
# 4. Data Generation
# ============================================================

def gen_bits(N, k_bits=4, snr_range=(0, 25), cfo_range=(-0.03, 0.03)):
    """Generate random bit vectors + channel params."""
    bits = np.random.randint(0, 2, (N, k_bits)).astype(np.float32)
    snr = np.random.uniform(*snr_range, (N, 1)).astype(np.float32)
    cfo = np.random.uniform(*cfo_range, (N, 1)).astype(np.float32)
    return bits, snr, cfo


def make_tf_dataset_coded(k_bits, batch_size, snr_range, cfo_range, steps):
    """tf.data pipeline for coded AE training."""
    def _gen():
        while True:
            bits = np.random.randint(
                0, 2, (batch_size, k_bits)).astype(np.float32)
            snr = np.random.uniform(
                *snr_range, (batch_size, 1)).astype(np.float32)
            cfo = np.random.uniform(
                *cfo_range, (batch_size, 1)).astype(np.float32)
            yield (bits, snr, cfo), bits

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            (tf.TensorSpec((batch_size, k_bits), tf.float32),
             tf.TensorSpec((batch_size, 1), tf.float32),
             tf.TensorSpec((batch_size, 1), tf.float32)),
            tf.TensorSpec((batch_size, k_bits), tf.float32),
        )
    )
    return ds.repeat().prefetch(tf.data.AUTOTUNE)


# ============================================================
# 5. Curriculum Training
# ============================================================

def train_curriculum_coded(ae, encoder=None, k_bits=4, n_sym=4,
                            a_sat=1.0, batch_size=4096,
                            logger=None, verbose=1,
                            use_custom_loss=True):
    """
    4-phase curriculum:
      Phase 1: High SNR, no CFO  → learn basic coding
      Phase 2: Medium SNR, small CFO → begin noise adaptation
      Phase 3: Full range → learn robust coding
      Phase 4: Polish → fine-tune
    """
    phases = [
        # (snr_lo, snr_hi, cfo_max, steps, epochs, lr, desc)
        (12, 25, 0.005, 256, 25, 2e-3, "Phase 1: High SNR, tiny CFO"),
        ( 5, 25, 0.020, 256, 25, 5e-4, "Phase 2: Mid SNR, small CFO"),
        ( 0, 30, 0.030, 256, 20, 1e-4, "Phase 3: Full range + polish"),
    ]

    if logger:
        rate = k_bits / (n_sym * 2)
        total_ep = sum(p[4] for p in phases)
        total_s = sum(p[3] * batch_size * p[4] for p in phases)
        logger.log(f'  Code rate R = {rate:.3f}')
        logger.log(f'  Total phases: {len(phases)}, epochs: {total_ep}')
        logger.log(f'  Total samples: {total_s:,}')
        logger.log(f'  Custom loss: {use_custom_loss}')

    hist_all = {'loss': [], 'bit_accuracy': [],
                'val_loss': [], 'val_bit_accuracy': []}

    if use_custom_loss and encoder is not None:
        loss_fn = make_coded_loss(encoder, k_bits, n_sym, a_sat)
    else:
        loss_fn = 'binary_crossentropy'

    ae.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3,
                                                clipnorm=1.0),
               loss=loss_fn, metrics=[bit_accuracy])

    for snr_lo, snr_hi, cfo_max, steps, epochs, lr, desc in phases:
        if logger:
            logger.log('')
            logger.log('=' * 60)
            logger.log(f'  {desc}')
            logger.log(f'  SNR: {snr_lo}-{snr_hi} dB | CFO_max: {cfo_max}')
            logger.log(f'  Steps: {steps} | Epochs: {epochs} | LR: {lr}')
            logger.log('=' * 60)

        cfo_range = (-cfo_max, cfo_max)
        train_ds = make_tf_dataset_coded(
            k_bits, batch_size, (snr_lo, snr_hi), cfo_range, steps)
        val = gen_bits(50_000, k_bits, (snr_lo, snr_hi), cfo_range)
        val_data = ([val[0], val[1], val[2]], val[0])

        keras.backend.set_value(ae.optimizer.learning_rate, lr)

        cbs = [
            keras.callbacks.ReduceLROnPlateau(
                'val_loss', factor=0.5, patience=5,
                min_lr=lr / 50, verbose=verbose),
            keras.callbacks.EarlyStopping(
                'val_loss', patience=8,
                restore_best_weights=True, verbose=verbose),
        ]
        if logger:
            cbs.append(_EpochLogCB(logger))

        h = ae.fit(
            train_ds, steps_per_epoch=steps, epochs=epochs,
            validation_data=val_data, callbacks=cbs, verbose=verbose
        )

        for k in hist_all:
            hist_all[k].extend(h.history.get(k, []))

        if logger:
            best_vl = min(h.history.get('val_loss', [999]))
            best_va = max(h.history.get('val_bit_accuracy', [0]))
            logger.log(f'  >> {desc} done: val_loss={best_vl:.4f}, '
                        f'val_bit_acc={best_va:.4f}')

    return hist_all


# ============================================================
# 6. Baselines: 16-QAM uncoded + coded (repetition)
# ============================================================

def qam16_constellation():
    pts = np.array([complex(i, q)
                    for i in [-3,-1,1,3] for q in [-3,-1,1,3]])
    return pts / np.sqrt(np.mean(np.abs(pts)**2))


def apply_pa_numpy(sig, a_sat=1.2, p_rapp=3,
                    alpha_pm=0.08, beta_pm=0.0):
    r = np.abs(sig)
    theta = np.angle(sig)
    ratio = r / a_sat
    g_r = r / (1.0 + ratio**(2*p_rapp))**(1.0/(2*p_rapp))
    phi = alpha_pm * r**2 / (1.0 + beta_pm * r**2)
    return g_r * np.exp(1j * (theta + phi))


def qam16_ber_uncoded(snr_db_arr, cfo_val=0.0, N=200_000,
                       a_sat=1.2, p_rapp=3,
                       alpha_pm=0.08, beta_pm=0.0):
    """16-QAM uncoded BER: 4 bits → 1 symbol, PA + optional CFO + AWGN.
    Single symbol (no repetition). Rate = 1.0"""
    const = qam16_constellation()
    const_pa = apply_pa_numpy(const, a_sat, p_rapp, alpha_pm, beta_pm)
    M, k = 16, 4
    ber_list = []
    for snr_db in snr_db_arr:
        sigma = np.sqrt(1.0 / (2.0 * 10**(snr_db/10.0)))
        idx = np.random.randint(0, M, N)
        tx = const[idx]
        tx_pa = apply_pa_numpy(tx, a_sat, p_rapp, alpha_pm, beta_pm)
        # Apply CFO (single symbol = n=0, so no effect for single)
        rx = tx_pa + (np.random.randn(N) + 1j*np.random.randn(N)) * sigma
        det = np.argmin(np.abs(rx[:,None] - const_pa[None,:]) ** 2, axis=-1)
        tx_b = (idx[:,None] >> np.arange(k)) & 1
        rx_b = (det[:,None] >> np.arange(k)) & 1
        ber_list.append(max(np.mean(tx_b != rx_b), 1e-6))
    return np.array(ber_list)


def qam16_ber_repetition(snr_db_arr, n_repeat=4, cfo_val=0.0, N=200_000,
                          a_sat=1.2, p_rapp=3,
                          alpha_pm=0.08, beta_pm=0.0):
    """16-QAM with simple repetition coding.
    Each symbol is repeated n_repeat times, MRC combining at receiver.
    Rate = 1/n_repeat. This is the simplest channel code baseline."""
    const = qam16_constellation()
    const_pa = apply_pa_numpy(const, a_sat, p_rapp, alpha_pm, beta_pm)
    M, k = 16, 4
    ber_list = []
    for snr_db in snr_db_arr:
        sigma = np.sqrt(1.0 / (2.0 * 10**(snr_db/10.0)))
        idx = np.random.randint(0, M, N)
        tx = const[idx]
        tx_pa = apply_pa_numpy(tx, a_sat, p_rapp, alpha_pm, beta_pm)
        # Repeat and add independent noise to each copy
        errs_total, bits_total = 0, 0
        rx_sum = np.zeros(N, dtype=complex)
        for rep in range(n_repeat):
            phase = 2.0 * np.pi * cfo_val * rep
            rx_rep = tx_pa * np.exp(1j * phase) + \
                     (np.random.randn(N) + 1j*np.random.randn(N)) * sigma
            # De-rotate if CFO known perfectly
            rx_rep_comp = rx_rep * np.exp(-1j * phase)
            rx_sum += rx_rep_comp
        # MRC (equal weighting since equal SNR per copy)
        rx_avg = rx_sum / n_repeat
        det = np.argmin(np.abs(rx_avg[:,None] - const_pa[None,:]) ** 2, axis=-1)
        tx_b = (idx[:,None] >> np.arange(k)) & 1
        rx_b = (det[:,None] >> np.arange(k)) & 1
        ber_list.append(max(np.mean(tx_b != rx_b), 1e-6))
    return np.array(ber_list)


def qam16_ber_cfo_nocomp(snr_db_arr, cfo_val, n_sym=4, N=200_000,
                          a_sat=1.2, p_rapp=3,
                          alpha_pm=0.08, beta_pm=0.0):
    """16-QAM, PA + CFO, NO compensation."""
    const = qam16_constellation()
    const_pa = apply_pa_numpy(const, a_sat, p_rapp, alpha_pm, beta_pm)
    M, k = 16, 4
    ber_list = []
    for snr_db in snr_db_arr:
        sigma = np.sqrt(1.0 / (2.0 * 10**(snr_db/10.0)))
        idx = np.random.randint(0, M, (N, n_sym))
        tx_pa = apply_pa_numpy(
            const[idx.flatten()], a_sat, p_rapp,
            alpha_pm, beta_pm).reshape(N, n_sym)
        n_idx = np.arange(n_sym).reshape(1, n_sym)
        phase = 2.0 * np.pi * cfo_val * n_idx
        rx = tx_pa * np.exp(1j * phase) + \
             (np.random.randn(N, n_sym) + 1j*np.random.randn(N, n_sym)) * sigma
        det = np.argmin(
            np.abs(rx[:,:,None] - const_pa[None,None,:]) ** 2, axis=-1)
        tx_b = (idx[:,:,None] >> np.arange(k)) & 1
        rx_b = (det[:,:,None] >> np.arange(k)) & 1
        ber_list.append(max(np.mean(tx_b != rx_b), 1e-6))
    return np.array(ber_list)


# ============================================================
# 7. Evaluate ANN Autoencoder
# ============================================================

def eval_coded_ber(encoder, ae, snr_arr, cfo_val,
                    k_bits=4, N=200_000, trials=3):
    """Evaluate ANN Channel Coding AE BER at fixed CFO across SNR range."""
    ber_list = []
    for snr_db in snr_arr:
        errs, bits = 0, 0
        for _ in range(trials):
            b, _, _ = gen_bits(N, k_bits, (snr_db, snr_db),
                               (cfo_val, cfo_val))
            snr = np.full((N, 1), snr_db, dtype=np.float32)
            cfo = np.full((N, 1), cfo_val, dtype=np.float32)
            pred = ae.predict([b, snr, cfo], batch_size=4096, verbose=0)
            pred_bits = (pred > 0.5).astype(np.float32)
            errs += np.sum(b != pred_bits)
            bits += b.size
        ber_list.append(max(errs / bits, 1e-6))
        print(f'  SNR {snr_db:2.0f} dB, CFO={cfo_val:.3f} -> '
              f'BER={ber_list[-1]:.2e}')
    return np.array(ber_list)


def eval_coded_ber_vs_cfo(encoder, ae, snr_val, cfo_arr,
                           k_bits=4, N=200_000, trials=3):
    """Evaluate ANN AE BER at fixed SNR across CFO range."""
    ber_list = []
    for cfo_val in cfo_arr:
        errs, bits = 0, 0
        for _ in range(trials):
            b, _, _ = gen_bits(N, k_bits, (snr_val, snr_val),
                               (cfo_val, cfo_val))
            snr = np.full((N, 1), snr_val, dtype=np.float32)
            cfo = np.full((N, 1), cfo_val, dtype=np.float32)
            pred = ae.predict([b, snr, cfo], batch_size=4096, verbose=0)
            pred_bits = (pred > 0.5).astype(np.float32)
            errs += np.sum(b != pred_bits)
            bits += b.size
        ber_list.append(max(errs / bits, 1e-6))
        print(f'  CFO={cfo_val:.4f}, SNR={snr_val:.0f} dB -> '
              f'BER={ber_list[-1]:.2e}')
    return np.array(ber_list)


# ============================================================
# 8. Visualization
# ============================================================

def plot_coded_constellation(encoder, k_bits=4, n_sym=4, a_sat=1.2,
                              p_rapp=3, alpha_pm=0.08, beta_pm=0.0,
                              max_show=64):
    """Visualize the learned coded constellation for 2^k codewords.
    Each codeword maps to N_sym IQ points → show as trajectory.
    For k>6, randomly samples max_show codewords to keep plot readable."""
    n_codes = 2 ** k_bits
    all_bits = np.array(
        [[(i >> b) & 1 for b in range(k_bits)]
         for i in range(n_codes)], dtype=np.float32)

    tx = encoder.predict(all_bits, verbose=0)  # (2^k, N_sym, 2)

    # Sample codewords for visualization if too many
    if n_codes > max_show:
        show_idx = np.random.choice(n_codes, max_show, replace=False)
        show_idx.sort()
    else:
        show_idx = np.arange(n_codes)
    n_show = len(show_idx)

    colors = plt.cm.hsv(np.linspace(0, 0.95, n_show))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: All codeword symbols (before PA)
    for i, cw in enumerate(show_idx):
        axes[0].scatter(tx[cw, :, 0], tx[cw, :, 1],
                       c=[colors[i]],
                       s=30, alpha=0.6, edgecolors='none')
        # Connect symbols within each codeword
        axes[0].plot(tx[cw, :, 0], tx[cw, :, 1],
                    color=colors[i], alpha=0.2, lw=0.8)
    c0 = plt.Circle((0,0), a_sat, fill=False, color='red', ls='--', lw=2)
    axes[0].add_patch(c0)
    shown_txt = f'(showing {n_show}/{n_codes})' if n_codes > max_show else ''
    axes[0].set_title(f'Coded Constellation (before PA)\n'
                      f'{n_codes} codewords × {n_sym} symbols {shown_txt}',
                      fontweight='bold', color='#2A9D8F')
    axes[0].set_xlim(-2,2); axes[0].set_ylim(-2,2)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('I'); axes[0].set_ylabel('Q')

    # Plot 2: After PA
    tx_flat = tx.reshape(-1, 2)
    tx_complex = tx_flat[:, 0] + 1j * tx_flat[:, 1]
    tx_pa = apply_pa_numpy(tx_complex, a_sat, p_rapp, alpha_pm, beta_pm)
    tx_pa_2d = np.stack([tx_pa.real, tx_pa.imag], axis=-1)
    tx_pa_reshape = tx_pa_2d.reshape(n_codes, n_sym, 2)

    for i, cw in enumerate(show_idx):
        axes[1].scatter(tx_pa_reshape[cw, :, 0], tx_pa_reshape[cw, :, 1],
                       c=[colors[i]],
                       s=30, alpha=0.6, edgecolors='none')
    axes[1].set_title(f'Coded Constellation (after PA)',
                      fontweight='bold', color='#E63946')
    axes[1].set_xlim(-2,2); axes[1].set_ylim(-2,2)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('I'); axes[1].set_ylabel('Q')

    # Plot 3: 16-QAM for reference
    qam = qam16_constellation()
    qam_pa = apply_pa_numpy(qam, a_sat, p_rapp, alpha_pm, beta_pm)
    axes[2].scatter(qam.real, qam.imag, c='blue', s=150,
                   edgecolors='k', zorder=5, label='Before PA', marker='o')
    axes[2].scatter(qam_pa.real, qam_pa.imag, c='red', s=150,
                   edgecolors='k', zorder=5, label='After PA', marker='^')
    axes[2].set_title('16-QAM Reference', fontweight='bold', color='#457B9D')
    axes[2].set_xlim(-2,2); axes[2].set_ylim(-2,2)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('I'); axes[2].set_ylabel('Q')
    axes[2].legend()

    rate = k_bits / (n_sym * 2)
    pass # suptitle removed
    plt.tight_layout()
    return fig


def plot_codeword_distances(encoder, k_bits=4, n_sym=4, max_codes=128):
    """Visualize pairwise distances between codewords in IQ space.
    For k>7, samples max_codes codewords to keep matrix manageable."""
    n_codes = 2 ** k_bits
    all_bits = np.array(
        [[(i >> b) & 1 for b in range(k_bits)]
         for i in range(n_codes)], dtype=np.float32)

    tx = encoder.predict(all_bits, verbose=0)  # (2^k, N_sym, 2)
    flat = tx.reshape(n_codes, -1)  # (2^k, N_sym*2)

    # Sample if too many
    if n_codes > max_codes:
        idx = np.random.choice(n_codes, max_codes, replace=False)
        idx.sort()
        flat_show = flat[idx]
        n_show = max_codes
        sampled = True
    else:
        flat_show = flat
        n_show = n_codes
        sampled = False

    # Pairwise Euclidean distance
    dist = np.sqrt(np.sum(
        (flat_show[:, None, :] - flat_show[None, :, :]) ** 2, axis=-1))

    # Also compute full min distance using all codewords
    # (do in chunks to avoid OOM for 256×256)
    all_min_dist = np.inf
    chunk = 32
    for i in range(0, n_codes, chunk):
        for j in range(i+1, n_codes, chunk):
            d = np.sqrt(np.sum(
                (flat[i:i+chunk, None, :] - flat[None, j:j+chunk, :]) ** 2,
                axis=-1))
            all_min_dist = min(all_min_dist, np.min(d))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    im = axes[0].imshow(dist, cmap='viridis', aspect='equal')
    title0 = 'Pairwise Codeword Distance'
    if sampled:
        title0 += f'\n(showing {n_show}/{n_codes})'
    axes[0].set_title(title0, fontweight='bold')
    axes[0].set_xlabel('Codeword index')
    axes[0].set_ylabel('Codeword index')
    plt.colorbar(im, ax=axes[0], label='Euclidean distance')

    # Histogram of distances
    upper_tri = dist[np.triu_indices(n_show, k=1)]
    # Guard against NaN
    upper_tri = upper_tri[np.isfinite(upper_tri)]
    if len(upper_tri) == 0:
        axes[1].text(0.5, 0.5, 'No valid distances\n(model may have NaN weights)',
                    ha='center', va='center', transform=axes[1].transAxes, color='red')
    else:
        axes[1].hist(upper_tri, bins=50, color='#2A9D8F', edgecolor='k', alpha=0.8)
    if len(upper_tri) > 0 and np.isfinite(all_min_dist):
        axes[1].axvline(all_min_dist, color='red', ls='--', lw=2,
                       label=f'Global min dist: {all_min_dist:.3f}')
        axes[1].axvline(np.mean(upper_tri), color='orange', ls='--', lw=2,
                       label=f'Mean dist: {np.mean(upper_tri):.3f}')
    axes[1].set_title('Distance Distribution', fontweight='bold')
    axes[1].set_xlabel('Euclidean distance')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    rate = k_bits / (n_sym * 2)
    pass # suptitle removed
    plt.tight_layout()
    return fig


def plot_ber_comparison(snr_arr, results_dict, title='',
                         rate_labels=None):
    """Plot BER vs SNR for multiple configurations."""
    fig, ax = plt.subplots(figsize=(11, 7))
    styles = [
        ('o-',  '#E63946', 2.5), ('s-',  '#2A9D8F', 2.5),
        ('^--', '#457B9D', 2.0), ('D--', '#E9C46A', 2.0),
        ('v-',  '#264653', 2.0), ('P:',  '#9B59B6', 2.0),
        ('*-',  '#E67E22', 2.0), ('h--', '#1ABC9C', 2.0),
    ]
    for idx, (name, ber) in enumerate(results_dict.items()):
        st = styles[idx % len(styles)]
        ax.semilogy(snr_arr, ber, st[0], color=st[1], lw=st[2], ms=8,
                    label=name, mfc='white', mew=2)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title(f'BER vs SNR — ANN Channel Coding{title}',
                  fontweight='bold')
    ax.legend( loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(snr_arr[0], snr_arr[-1])
    ax.set_ylim(1e-5, 1)
    plt.tight_layout()
    return fig


def plot_ber_vs_cfo(cfo_arr, results_dict, snr_val=15):
    """Plot BER vs CFO at fixed SNR."""
    fig, ax = plt.subplots(figsize=(11, 7))
    styles = [
        ('o-',  '#E63946', 2.5), ('s-',  '#2A9D8F', 2.5),
        ('^--', '#457B9D', 2.0), ('D--', '#E9C46A', 2.0),
        ('v-',  '#264653', 2.0),
    ]
    for idx, (name, ber) in enumerate(results_dict.items()):
        st = styles[idx % len(styles)]
        ax.semilogy(cfo_arr, ber, st[0], color=st[1], lw=st[2], ms=8,
                    label=name, mfc='white', mew=2)

    ax.set_xlabel('Normalized CFO (Δf·Ts)')
    ax.set_ylabel('BER')
    ax.set_title(f'BER vs CFO @ SNR={snr_val} dB\n'
                 f'ANN Channel Coding vs Baselines',
                  fontweight='bold')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-5, 1)
    plt.tight_layout()
    return fig


def plot_rate_comparison(snr_arr, rate_results, cfo_val=0.0):
    """Compare BER across different code rates at same conditions.
    rate_results: dict of {rate_label: ber_array}"""
    fig, ax = plt.subplots(figsize=(11, 7))

    cmap = plt.cm.coolwarm
    n = len(rate_results)
    colors = [cmap(i / max(n-1, 1)) for i in range(n)]
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'h']

    for idx, (name, ber) in enumerate(rate_results.items()):
        ax.semilogy(snr_arr, ber,
                    f'{markers[idx%len(markers)]}-',
                    color=colors[idx], lw=2.5, ms=9,
                    label=name, mfc='white', mew=2)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title(f'Code Rate Comparison (CFO={cfo_val:.3f})\n'
                 f'Higher rate = more efficient, Lower rate = more robust',
                  fontweight='bold')
    ax.legend( loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(snr_arr[0], snr_arr[-1])
    ax.set_ylim(1e-5, 1)
    plt.tight_layout()
    return fig


def plot_history(hist):
    """Plot training history for coded AE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist['loss'], lw=2, color='#E63946', label='Train')
    axes[0].plot(hist['val_loss'], lw=2, color='#457B9D', label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist['bit_accuracy'], lw=2, color='#E63946', label='Train')
    axes[1].plot(hist['val_bit_accuracy'], lw=2, color='#457B9D', label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Bit Accuracy')
    axes[1].set_title('Per-Bit Accuracy', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    pass # suptitle removed
    plt.tight_layout()
    return fig


def plot_coding_gain_analysis(snr_arr, ber_uncoded, ber_coded_dict):
    """Analyze coding gain: how much SNR is saved by the coded system
    compared to uncoded at same BER target."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # BER curves
    axes[0].semilogy(snr_arr, ber_uncoded, 'k--', lw=2.5, ms=8,
                    label='Uncoded 16-QAM', mfc='white', mew=2)
    styles = [
        ('o-', '#E63946', 2.5), ('s-', '#2A9D8F', 2.5),
        ('^-', '#457B9D', 2.0), ('D-', '#9B59B6', 2.0),
    ]
    for idx, (name, ber) in enumerate(ber_coded_dict.items()):
        st = styles[idx % len(styles)]
        axes[0].semilogy(snr_arr, ber, st[0], color=st[1], lw=st[2], ms=8,
                        label=name, mfc='white', mew=2)
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('BER')
    axes[0].set_title('BER vs SNR', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].set_ylim(1e-5, 1)

    # Coding gain at target BERs
    targets = [1e-2, 1e-3, 1e-4]
    bar_data = {}
    for name, ber in ber_coded_dict.items():
        gains = []
        for t in targets:
            # Find SNR where BER crosses target (interpolate)
            snr_uncoded = _interp_snr(snr_arr, ber_uncoded, t)
            snr_coded = _interp_snr(snr_arr, ber, t)
            if snr_uncoded is not None and snr_coded is not None:
                gains.append(snr_uncoded - snr_coded)
            else:
                gains.append(0)
        bar_data[name] = gains

    x = np.arange(len(targets))
    width = 0.8 / max(len(bar_data), 1)
    colors = ['#E63946', '#2A9D8F', '#457B9D', '#9B59B6']
    for idx, (name, gains) in enumerate(bar_data.items()):
        axes[1].bar(x + idx * width, gains, width,
                   label=name, color=colors[idx % len(colors)],
                   edgecolor='k', alpha=0.85)
    axes[1].set_xlabel('Target BER')
    axes[1].set_ylabel('Coding Gain (dB)')
    axes[1].set_title('Coding Gain vs Uncoded', fontweight='bold')
    axes[1].set_xticks(x + width * (len(bar_data)-1) / 2)
    axes[1].set_xticklabels([f'{t:.0e}' for t in targets])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    pass # removed suptitle
    plt.tight_layout()
    return fig


def _interp_snr(snr_arr, ber_arr, target_ber):
    """Linearly interpolate SNR at which BER crosses target."""
    log_ber = np.log10(np.clip(ber_arr, 1e-10, 1))
    log_target = np.log10(target_ber)
    for i in range(len(log_ber) - 1):
        if log_ber[i] >= log_target >= log_ber[i+1]:
            frac = (log_target - log_ber[i]) / (log_ber[i+1] - log_ber[i])
            return snr_arr[i] + frac * (snr_arr[i+1] - snr_arr[i])
    return None
