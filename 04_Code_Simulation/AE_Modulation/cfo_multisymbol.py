"""
Demo2: Multi-Symbol CFO + PA + AWGN
====================================
Autoencoder jointly decodes L symbols under:
  - Rapp PA nonlinearity (AM/AM + AM/PM)
  - Carrier Frequency Offset (CFO) → progressive phase rotation
  - AWGN noise

Key insight:
  CFO causes linear phase ramp: y[n] = PA(x[n])·exp(j·2π·Δf·n) + noise
  Traditional QAM must estimate+compensate CFO separately.
  ANN autoencoder learns joint decode, implicitly estimating CFO.
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
    'legend.fontsize': 14,
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
    def __init__(self, log_file='train_cfo.txt', append=False):
        self.log_file = log_file
        self.f = open(log_file, 'a' if append else 'w', encoding='utf-8')
        self.log('=' * 70)
        self.log(f'  Demo2: Multi-Symbol CFO+PA+AWGN — Training Log')
        self.log(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.log('=' * 70)

    def log(self, msg=''):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        self.f.write(line + '\n')
        self.f.flush()

    def log_params(self, M, L, a_sat, p_rapp, alpha_pm, beta_pm,
                    cfo_max, batch_size):
        self.log(f'  M={M}, L={L} symbols/block')
        self.log(f'  A_sat={a_sat}, p_rapp={p_rapp}')
        self.log(f'  alpha_pm={alpha_pm}, beta_pm={beta_pm}')
        self.log(f'  CFO_max={cfo_max} (normalized Δf·Ts)')
        self.log(f'  batch_size={batch_size}')

    def log_model(self, model):
        self.log(f'  {model.name}: {model.count_params():,} params')

    def log_constellation(self, encoder, M=16, L=4):
        test_in = np.zeros((M, L, M), dtype=np.float32)
        for i in range(M):
            test_in[i, 0, i] = 1.0
        pts = encoder.predict(test_in, verbose=0)[:, 0, :]
        amps = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        min_dist = np.inf
        for i in range(M):
            for j in range(i+1, M):
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                min_dist = min(min_dist, d)
        papr = 10 * np.log10(np.max(amps**2) / (np.mean(amps**2) + 1e-12))
        self.log(f'  Constellation:')
        self.log(f'    Avg amp: {np.mean(amps):.4f}')
        self.log(f'    Max amp: {np.max(amps):.4f}')
        self.log(f'    Min dist: {min_dist:.4f}')
        self.log(f'    PAPR: {papr:.2f} dB')

    def log_ber_table(self, snr_arr, results_dict):
        self.log('')
        header = f'  {"SNR":>4s}'
        for name in results_dict:
            header += f'  {name:>14s}'
        self.log(header)
        self.log('  ' + '-' * len(header))
        for i, s in enumerate(snr_arr):
            line = f'  {s:4.0f}'
            for name, ber_arr in results_dict.items():
                line += f'  {ber_arr[i]:14.2e}'
            self.log(line)

    def log_finish(self):
        self.log('')
        self.log('=' * 70)
        self.log(f'  Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.log('=' * 70)
        self.f.close()


def init_logger(log_file='train_cfo.txt', append=False):
    return TrainLogger(log_file, append)


class _EpochLogCB(keras.callbacks.Callback):
    def __init__(self, logger): super().__init__(); self.logger = logger
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.logger.log(
                f'  Ep {epoch+1:3d} | '
                f'loss={logs.get("loss",0):.4f} '
                f'acc={logs.get("symbol_accuracy",0):.4f} | '
                f'v_loss={logs.get("val_loss",0):.4f} '
                f'v_acc={logs.get("val_symbol_accuracy",0):.4f} | '
                f'lr={float(self.model.optimizer.learning_rate):.2e}')


# ============================================================
# 1. Custom Layers
# ============================================================

class PowerNormBlock(layers.Layer):
    """Normalize average power per symbol. Input: (batch, L, 2)."""
    def call(self, x):
        mean_power = tf.reduce_mean(tf.reduce_sum(x**2, axis=-1))
        return x / tf.sqrt(mean_power + 1e-8)


class CFOPAChannel(layers.Layer):
    """Rapp PA + CFO + AWGN for multi-symbol blocks.
    
    Signal: x[n] → PA(x[n]) → ·exp(j·2π·Δf·n) → +noise → y[n]
    Inputs: [tx(B,L,2), snr_db(B,1), cfo_norm(B,1)]
    Output: (B, L, 2)
    """
    def __init__(self, a_sat=1.2, p_rapp=3,
                 alpha_pm=0.08, beta_pm=0.0, L=4, **kwargs):
        super().__init__(**kwargs)
        self.a_sat = a_sat
        self.p_rapp = p_rapp
        self.alpha_pm = alpha_pm
        self.beta_pm = beta_pm
        self.L = L

    def call(self, inputs, training=None):
        tx, snr_db, cfo_norm = inputs
        B = tf.shape(tx)[0]

        # 1. PA per symbol
        x_I, x_Q = tx[:, :, 0:1], tx[:, :, 1:2]
        r = tf.sqrt(x_I**2 + x_Q**2 + 1e-8)
        theta = tf.atan2(x_Q, x_I)
        ratio = r / self.a_sat
        g_r = r / tf.pow(1.0 + tf.pow(ratio, 2*self.p_rapp),
                          1.0 / (2*self.p_rapp))
        phi_pa = self.alpha_pm * r**2 / (1.0 + self.beta_pm * r**2)
        pa_I = g_r * tf.cos(theta + phi_pa)
        pa_Q = g_r * tf.sin(theta + phi_pa)

        # 2. Rayleigh fading (block fading: same h for all L symbols)
        h_r = tf.random.normal([B, 1, 1])
        h_i = tf.random.normal([B, 1, 1])
        y_r = h_r * pa_I - h_i * pa_Q
        y_i = h_r * pa_Q + h_i * pa_I

        # 3. CFO rotation: phase = 2π·Δf·n
        n_idx = tf.cast(tf.range(self.L), tf.float32)
        n_idx = tf.reshape(n_idx, (1, self.L, 1))
        cfo_exp = tf.reshape(cfo_norm, (-1, 1, 1))
        phase = 2.0 * 3.141592653589793 * cfo_exp * n_idx

        yr2 = y_r * tf.cos(phase) - y_i * tf.sin(phase)
        yi2 = y_r * tf.sin(phase) + y_i * tf.cos(phase)

        # 4. AWGN
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigma = tf.sqrt(1.0 / (2.0 * snr_lin + 1e-8))
        sigma = tf.reshape(sigma, (-1, 1, 1))
        yr2 += tf.random.normal(tf.shape(yr2)) * sigma
        yi2 += tf.random.normal(tf.shape(yi2)) * sigma

        # Output flat: [y_0_I, y_0_Q, ..., y_{L-1}_I, y_{L-1}_Q, h_r, h_i]
        rx = tf.concat([yr2, yi2], axis=-1)  # (B, L, 2)
        rx_flat = tf.reshape(rx, [B, self.L * 2])  # (B, 2L)
        h_out = tf.concat([tf.reshape(h_r, [B, 1]),
                           tf.reshape(h_i, [B, 1])], axis=-1)  # (B, 2)
        return tf.concat([rx_flat, h_out], axis=-1)  # (B, 2L+2)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'a_sat': self.a_sat, 'p_rapp': self.p_rapp,
                    'alpha_pm': self.alpha_pm, 'beta_pm': self.beta_pm,
                    'L': self.L})
        return cfg


# ============================================================
# 2. Build Multi-Symbol Autoencoder
# ============================================================

def symbol_accuracy(y_true, y_pred):
    """Per-symbol accuracy for (batch, L, M) tensors."""
    true_idx = tf.argmax(y_true, axis=-1)
    pred_idx = tf.argmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(true_idx, pred_idx), tf.float32))


def build_multisymbol_autoencoder(M=16, L=4, n_channel=2,
                                   a_sat=1.2, p_rapp=3,
                                   alpha_pm=0.08, beta_pm=0.0,
                                   dropout_rate=0.05):
    """
    Encoder: shared weights, per-symbol via TimeDistributed
      (B, L, M) → (B, L, 2)
    Channel: PA → CFO → AWGN
    Decoder: joint blind decode of all L symbols
      (B, L, 2) + SNR → (B, L, M)
    """
    msg_in = layers.Input(shape=(L, M), name='msg_input')
    snr_in = layers.Input(shape=(1,), name='snr_input')
    cfo_in = layers.Input(shape=(1,), name='cfo_input')

    # ---- Encoder (shared weights) ----
    e = layers.TimeDistributed(
        layers.Dense(256, activation='selu'), name='enc_td1')(msg_in)
    e = layers.TimeDistributed(
        layers.BatchNormalization(), name='enc_bn1')(e)
    e = layers.TimeDistributed(
        layers.Dense(128, activation='selu'), name='enc_td2')(e)
    e = layers.TimeDistributed(
        layers.Dense(n_channel, activation=None), name='enc_out')(e)
    tx = PowerNormBlock(name='power_norm')(e)

    encoder = Model(msg_in, tx, name='Encoder')

    # ---- Channel: PA → Rayleigh → CFO → AWGN ----
    # Output: flat (B, 2L+2) = [rx_flat, h_r, h_i]
    ch_out = CFOPAChannel(
        a_sat=a_sat, p_rapp=p_rapp,
        alpha_pm=alpha_pm, beta_pm=beta_pm, L=L,
        name='cfo_pa_channel'
    )([tx, snr_in, cfo_in])

    # ---- Decoder: [rx_flat, h, snr_norm] → L output heads ----
    snr_norm = layers.Lambda(
        lambda s: tf.clip_by_value((s - 12.5) / 12.5, -1.5, 1.5),
        name='snr_norm')(snr_in)
    dec_in = layers.Concatenate(name='dec_concat1')([ch_out, snr_norm])

    d = layers.Dense(512, activation='selu', name='dec_fc1')(dec_in)
    d = layers.BatchNormalization(name='dec_bn1')(d)
    d = layers.Dropout(dropout_rate, name='dec_drop1')(d)

    d = layers.Dense(512, activation='selu', name='dec_fc2')(d)
    d = layers.BatchNormalization(name='dec_bn2')(d)

    # Re-inject SNR
    d = layers.Concatenate(name='dec_concat2')([d, snr_norm])

    d = layers.Dense(256, activation='selu', name='dec_fc3')(d)
    d = layers.BatchNormalization(name='dec_bn3')(d)
    d = layers.Dropout(dropout_rate, name='dec_drop2')(d)
    d = layers.Dense(128, activation='selu', name='dec_fc4')(d)

    # L separate output heads
    outputs = []
    for i in range(L):
        out_i = layers.Dense(M, activation='softmax', name=f'dec_sym{i}')(d)
        outputs.append(out_i)

    output = layers.Lambda(
        lambda x: tf.stack(x, axis=1), name='stack_outputs')(outputs)

    autoencoder = Model([msg_in, snr_in, cfo_in], output, name='Autoencoder')
    return encoder, autoencoder


def build_multisymbol_autoencoder_light(M=16, L=4, n_channel=2,
                                        a_sat=1.2, p_rapp=3,
                                        alpha_pm=0.08, beta_pm=0.0):
    """
    Light model with Rayleigh fading.
    Channel: PA → Rayleigh (block) → CFO → AWGN
    Channel output: flat [rx(2L), h(2)] → decoder gets [rx, h, snr]
    """
    msg_in = layers.Input(shape=(L, M), name='msg_input')
    snr_in = layers.Input(shape=(1,), name='snr_input')
    cfo_in = layers.Input(shape=(1,), name='cfo_input')

    # ---- Encoder (shared weights, matching Demo4 depth: 64→32→2) ----
    e = layers.TimeDistributed(
        layers.Dense(64, activation='selu'), name='enc_td1')(msg_in)
    e = layers.TimeDistributed(
        layers.Dense(32, activation='selu'), name='enc_td2')(e)
    e = layers.TimeDistributed(
        layers.Dense(n_channel, activation=None), name='enc_out')(e)
    tx = PowerNormBlock(name='power_norm')(e)

    encoder = Model(msg_in, tx, name='Encoder')

    # ---- Channel: PA → Rayleigh → CFO → AWGN ----
    # Output: flat (B, 2L+2) = [rx_flat, h_r, h_i]
    ch_out = CFOPAChannel(
        a_sat=a_sat, p_rapp=p_rapp,
        alpha_pm=alpha_pm, beta_pm=beta_pm, L=L,
        name='cfo_pa_channel'
    )([tx, snr_in, cfo_in])

    # ---- Decoder: [rx_flat, h_r, h_i, snr_norm] → L output heads ----
    snr_norm = layers.Lambda(
        lambda s: tf.clip_by_value((s - 17.5) / 17.5, -1.5, 1.5),
        name='snr_norm')(snr_in)
    dec_in = layers.Concatenate(name='dec_concat1')([ch_out, snr_norm])

    d = layers.Dense(512, activation='selu', name='dec_fc1')(dec_in)
    d = layers.BatchNormalization(name='dec_bn1')(d)
    d = layers.Dense(256, activation='selu', name='dec_fc2')(d)
    d = layers.BatchNormalization(name='dec_bn2')(d)
    d = layers.Dense(256, activation='selu', name='dec_fc3')(d)
    d = layers.BatchNormalization(name='dec_bn3')(d)
    d = layers.Dense(128, activation='selu', name='dec_fc4')(d)

    outputs = []
    for i in range(L):
        out_i = layers.Dense(M, activation='softmax', name=f'dec_sym{i}')(d)
        outputs.append(out_i)

    output = layers.Lambda(
        lambda x: tf.stack(x, axis=1), name='stack_outputs')(outputs)

    autoencoder = Model([msg_in, snr_in, cfo_in], output, name='Autoencoder')
    return encoder, autoencoder


# ============================================================
# 3. Custom Loss
# ============================================================

def make_custom_loss(encoder, M=16, L=4, a_sat=1.0,
                     lambda_power=0.01, lambda_dist=0.005):
    test_in = np.zeros((M, L, M), dtype=np.float32)
    for i in range(M):
        test_in[i, 0, i] = 1.0

    def _loss(y_true, y_pred):
        cce = keras.losses.categorical_crossentropy(y_true, y_pred)
        cce = tf.reduce_mean(cce, axis=-1)

        pts = encoder(tf.constant(test_in), training=False)[:, 0, :]
        amps = tf.sqrt(tf.reduce_sum(pts**2, axis=-1))
        power_pen = tf.reduce_mean(
            tf.square(tf.maximum(amps - a_sat * 0.92, 0.0)))

        pts_a = tf.expand_dims(pts, 0)
        pts_b = tf.expand_dims(pts, 1)
        sq_d = tf.reduce_sum(tf.square(pts_a - pts_b), axis=-1)
        mask = 1.0 - tf.eye(M)
        masked = sq_d + (1.0 - mask) * 1e6
        min_d_target = 4.0 / M
        dist_pen = tf.maximum(min_d_target**2 - tf.reduce_min(masked), 0.0)

        return cce + lambda_power * power_pen + lambda_dist * dist_pen
    return _loss


# ============================================================
# 4. Data Generation
# ============================================================

def gen_train_cfo(N, M=16, L=4, snr_range=(0, 35), cfo_range=(-0.03, 0.03)):
    labels = np.random.randint(0, M, (N, L))
    oh = np.eye(M, dtype=np.float32)[labels]
    snr = np.random.uniform(*snr_range, (N, 1)).astype(np.float32)
    cfo = np.random.uniform(*cfo_range, (N, 1)).astype(np.float32)
    return oh, snr, cfo, labels


def make_tf_dataset_cfo(M, L, batch_size, snr_range, cfo_range, steps):
    def _gen():
        while True:
            lbl = np.random.randint(0, M, (batch_size, L)).astype(np.int32)
            oh = np.eye(M, dtype=np.float32)[lbl]
            snr = np.random.uniform(*snr_range,
                                     (batch_size, 1)).astype(np.float32)
            cfo = np.random.uniform(*cfo_range,
                                     (batch_size, 1)).astype(np.float32)
            yield (oh, snr, cfo), oh

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            (tf.TensorSpec((batch_size, L, M), tf.float32),
             tf.TensorSpec((batch_size, 1), tf.float32),
             tf.TensorSpec((batch_size, 1), tf.float32)),
            tf.TensorSpec((batch_size, L, M), tf.float32),
        )
    )
    return ds.repeat().prefetch(tf.data.AUTOTUNE)


# ============================================================
# 5. Curriculum Training
# ============================================================

def train_curriculum_cfo(ae, encoder=None, M=16, L=4, a_sat=1.0,
                          batch_size=4096, logger=None, verbose=1,
                          use_custom_loss=True):
    """
    4-phase curriculum with increasing CFO range.
    """
    phases = [
        # (snr_lo, snr_hi, cfo_max, steps, epochs, lr, desc)
        (14, 35, 0.005, 512, 60, 2e-3, "Phase 1: High SNR, tiny CFO"),
        ( 7, 35, 0.015, 512, 50, 5e-4, "Phase 2: Mid SNR, small CFO"),
        ( 0, 35, 0.030, 768, 60, 2e-4, "Phase 3: Full range"),
        ( 0, 35, 0.030, 512, 40, 5e-5, "Phase 4: Polish"),
    ]

    if logger:
        total_ep = sum(p[4] for p in phases)
        total_s = sum(p[3] * batch_size * p[4] for p in phases)
        logger.log(f'  Total phases: {len(phases)}, epochs: {total_ep}')
        logger.log(f'  Total samples: {total_s:,}')
        logger.log(f'  Custom loss: {use_custom_loss}')

    hist_all = {'loss': [], 'symbol_accuracy': [],
                'val_loss': [], 'val_symbol_accuracy': []}

    if use_custom_loss and encoder is not None:
        loss_fn = make_custom_loss(encoder, M, L, a_sat)
    else:
        loss_fn = 'categorical_crossentropy'

    ae.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3),
               loss=loss_fn, metrics=[symbol_accuracy])

    for snr_lo, snr_hi, cfo_max, steps, epochs, lr, desc in phases:
        if logger:
            logger.log('')
            logger.log('=' * 60)
            logger.log(f'  {desc}')
            logger.log(f'  SNR: {snr_lo}-{snr_hi} dB | CFO_max: {cfo_max}')
            logger.log(f'  Steps: {steps} | Epochs: {epochs} | LR: {lr}')
            logger.log('=' * 60)

        cfo_range = (-cfo_max, cfo_max)
        train_ds = make_tf_dataset_cfo(
            M, L, batch_size, (snr_lo, snr_hi), cfo_range, steps)
        val = gen_train_cfo(100_000, M, L, (snr_lo, snr_hi), cfo_range)
        val_data = ([val[0], val[1], val[2]], val[0])

        keras.backend.set_value(ae.optimizer.learning_rate, lr)

        cbs = [
            keras.callbacks.ReduceLROnPlateau(
                'val_loss', factor=0.5, patience=7,
                min_lr=lr / 50, verbose=verbose),
            keras.callbacks.EarlyStopping(
                'val_loss', patience=15,
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
            best_va = max(h.history.get('val_symbol_accuracy', [0]))
            logger.log(f'  >> {desc} done: val_loss={best_vl:.4f}, '
                        f'val_acc={best_va:.4f}')

    return hist_all


def train_curriculum_cfo_light(ae, encoder=None, M=16, L=4, a_sat=1.0,
                               batch_size=4096, logger=None, verbose=1,
                               use_custom_loss=True):
    """
    Light 4-phase curriculum: half epochs + half steps vs full version.
    ~45% training time.
    """
    phases = [
        # (snr_lo, snr_hi, cfo_max, steps, epochs, lr, desc)
        (14, 35, 0.005, 256, 30, 2e-3, "[LIGHT] Phase 1: High SNR, tiny CFO"),
        ( 7, 35, 0.015, 256, 25, 5e-4, "[LIGHT] Phase 2: Mid SNR, small CFO"),
        ( 0, 35, 0.030, 512, 30, 2e-4, "[LIGHT] Phase 3: Full range"),
        ( 0, 35, 0.030, 256, 20, 5e-5, "[LIGHT] Phase 4: Polish"),
    ]

    if logger:
        total_ep = sum(p[4] for p in phases)
        total_s = sum(p[3] * batch_size * p[4] for p in phases)
        logger.log(f'  [LIGHT] Total phases: {len(phases)}, epochs: {total_ep}')
        logger.log(f'  [LIGHT] Total samples: {total_s:,}')
        logger.log(f'  Custom loss: {use_custom_loss}')

    hist_all = {'loss': [], 'symbol_accuracy': [],
                'val_loss': [], 'val_symbol_accuracy': []}

    if use_custom_loss and encoder is not None:
        loss_fn = make_custom_loss(encoder, M, L, a_sat)
    else:
        loss_fn = 'categorical_crossentropy'

    ae.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3),
               loss=loss_fn, metrics=[symbol_accuracy])

    for snr_lo, snr_hi, cfo_max, steps, epochs, lr, desc in phases:
        if logger:
            logger.log('')
            logger.log('=' * 60)
            logger.log(f'  {desc}')
            logger.log(f'  SNR: {snr_lo}-{snr_hi} dB | CFO_max: {cfo_max}')
            logger.log(f'  Steps: {steps} | Epochs: {epochs} | LR: {lr}')
            logger.log('=' * 60)

        cfo_range = (-cfo_max, cfo_max)
        train_ds = make_tf_dataset_cfo(
            M, L, batch_size, (snr_lo, snr_hi), cfo_range, steps)
        val = gen_train_cfo(50_000, M, L, (snr_lo, snr_hi), cfo_range)
        val_data = ([val[0], val[1], val[2]], val[0])

        keras.backend.set_value(ae.optimizer.learning_rate, lr)

        cbs = [
            keras.callbacks.ReduceLROnPlateau(
                'val_loss', factor=0.5, patience=5,
                min_lr=lr / 50, verbose=verbose),
            keras.callbacks.EarlyStopping(
                'val_loss', patience=10,
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
            best_va = max(h.history.get('val_symbol_accuracy', [0]))
            logger.log(f'  >> {desc} done: val_loss={best_vl:.4f}, '
                        f'val_acc={best_va:.4f}')

    return hist_all


# ============================================================
# 6. 16-QAM Baselines
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


def _qam16_grid_slicer(rx_eq, scale):
    """16-QAM grid slicer (same as Demo4). rx_eq: (N,L) complex → idx (N,L).
    Uses standard QAM decision boundaries — does NOT know PA distortion."""
    thresh = np.array([-2, 0, 2], dtype=np.float64) * scale
    i_lev = np.searchsorted(thresh, rx_eq.real)
    q_lev = np.searchsorted(thresh, rx_eq.imag)
    return i_lev * 4 + q_lev


def qam16_ber_cfo_nocomp(snr_db_arr, cfo_val, L=4, N=200_000,
                          a_sat=1.2, p_rapp=3,
                          alpha_pm=0.08, beta_pm=0.0):
    """16-QAM + PA + Rayleigh + CFO → ZF equalize + grid slicer (no CFO comp).
    Matches Demo4's QAM receiver: ZF + standard grid slicer."""
    const = qam16_constellation()
    M, k = 16, 4
    scale = 1.0 / np.sqrt(10.0)
    ber_list = []
    for snr_db in snr_db_arr:
        sigma = np.sqrt(1.0 / (2.0 * 10**(snr_db/10.0)))
        idx = np.random.randint(0, M, (N, L))
        tx_pa = apply_pa_numpy(
            const[idx.flatten()], a_sat, p_rapp,
            alpha_pm, beta_pm).reshape(N, L)
        # Rayleigh (block fading: same h for all L symbols)
        h = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
        rx = h * tx_pa  # fading
        # CFO
        n_idx = np.arange(L).reshape(1, L)
        phase = 2.0 * np.pi * cfo_val * n_idx
        rx = rx * np.exp(1j * phase)
        # AWGN
        rx += (np.random.randn(N, L) + 1j*np.random.randn(N, L)) * sigma
        # ZF equalize (no CFO compensation)
        rx_eq = rx / (h + 1e-10)
        # Grid slicer (standard QAM boundaries — doesn't know PA)
        det = _qam16_grid_slicer(rx_eq, scale)
        tx_b = (idx[:,:,None] >> np.arange(k)) & 1
        rx_b = (det[:,:,None] >> np.arange(k)) & 1
        ber_list.append(max(np.mean(tx_b != rx_b), 1e-6))
    return np.array(ber_list)


def qam16_ber_cfo_perfect(snr_db_arr, cfo_val, L=4, N=200_000,
                           a_sat=1.2, p_rapp=3,
                           alpha_pm=0.08, beta_pm=0.0):
    """16-QAM + PA + Rayleigh + CFO, PERFECT CFO compensation + ZF."""
    const = qam16_constellation()
    M, k = 16, 4
    scale = 1.0 / np.sqrt(10.0)
    ber_list = []
    for snr_db in snr_db_arr:
        sigma = np.sqrt(1.0 / (2.0 * 10**(snr_db/10.0)))
        idx = np.random.randint(0, M, (N, L))
        tx_pa = apply_pa_numpy(
            const[idx.flatten()], a_sat, p_rapp,
            alpha_pm, beta_pm).reshape(N, L)
        h = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
        rx = h * tx_pa
        n_idx = np.arange(L).reshape(1, L)
        phase = 2.0 * np.pi * cfo_val * n_idx
        rx = rx * np.exp(1j * phase)
        rx += (np.random.randn(N, L) + 1j*np.random.randn(N, L)) * sigma
        # Perfect CFO de-rotation + ZF
        rx_comp = rx * np.exp(-1j * phase)
        rx_eq = rx_comp / (h + 1e-10)
        det = _qam16_grid_slicer(rx_eq, scale)
        tx_b = (idx[:,:,None] >> np.arange(k)) & 1
        rx_b = (det[:,:,None] >> np.arange(k)) & 1
        ber_list.append(max(np.mean(tx_b != rx_b), 1e-6))
    return np.array(ber_list)


def qam16_ber_no_cfo(snr_db_arr, L=4, N=200_000,
                      a_sat=1.2, p_rapp=3,
                      alpha_pm=0.08, beta_pm=0.0):
    """16-QAM + PA + Rayleigh, no CFO → ZF + grid slicer."""
    return qam16_ber_cfo_nocomp(snr_db_arr, 0.0, L, N,
                                 a_sat, p_rapp, alpha_pm, beta_pm)


# ============================================================
# 7. Evaluate Autoencoder
# ============================================================

def _apply_channel_numpy(tx_complex, snr_db, cfo_val, L,
                         a_sat=1.2, p_rapp=3,
                         alpha_pm=0.08, beta_pm=0.0):
    """Apply PA + Rayleigh + CFO + AWGN channel in NumPy (for BER eval).

    Returns:
        rx_IQ: (N, L, 2) real-valued received signal [I, Q]
        h_complex: (N, 1) complex Rayleigh fading coefficient
    """
    N = tx_complex.shape[0]

    # 1. PA
    tx_pa = apply_pa_numpy(tx_complex, a_sat, p_rapp, alpha_pm, beta_pm)

    # 2. Rayleigh (block fading)
    h_complex = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
    y = h_complex * tx_pa

    # 3. CFO
    n_idx = np.arange(L).reshape(1, L)
    phase = 2.0 * np.pi * cfo_val * n_idx
    y = y * np.exp(1j * phase)

    # 4. AWGN
    sigma = np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))
    y += (np.random.randn(N, L) + 1j * np.random.randn(N, L)) * sigma

    # Convert to real (N, L, 2): [I, Q]
    rx_IQ = np.stack([y.real, y.imag], axis=-1).astype(np.float32)
    return rx_IQ, h_complex


def _run_decoder(ae, rx_IQ, h_complex, snr_db, M=16, L=4, batch_size=4096):
    """Run received signal through AE decoder layers only.

    Decoder input format matches training: [rx_flat(2L), h_r(1), h_i(1), snr_norm(1)]

    Args:
        ae: trained autoencoder model
        rx_IQ: (N, L, 2) received signal in real format
        h_complex: (N, 1) complex Rayleigh fading coefficient
        snr_db: scalar SNR in dB
    Returns:
        det: (N, L) detected symbol indices
    """
    N = rx_IQ.shape[0]
    all_det = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        rx_batch = rx_IQ[start:end]
        h_batch = h_complex[start:end]
        n_batch = rx_batch.shape[0]

        # Flatten rx: (n, L, 2) → (n, 2L)
        rx_flat = rx_batch.reshape(n_batch, -1).astype(np.float32)

        # h: (n, 1) complex → (n, 2) [h_r, h_i]
        h_ri = np.concatenate([h_batch.real, h_batch.imag],
                               axis=-1).astype(np.float32)

        # SNR normalization
        snr_in = np.full((n_batch, 1), snr_db, dtype=np.float32)
        snr_normed = np.clip((snr_in - 17.5) / 17.5, -1.5, 1.5)

        # Concatenate [rx_flat, h_r, h_i, snr_norm] — matches channel output + snr
        dec_input = np.concatenate([rx_flat, h_ri, snr_normed],
                                    axis=-1).astype(np.float32)
        d = tf.constant(dec_input)

        # Detect decoder layer names
        layer_names = [l.name for l in ae.layers]

        # --- First block: dec_fc1 → dec_bn1 [→ dec_drop1] ---
        d = ae.get_layer('dec_fc1')(d, training=False)
        d = ae.get_layer('dec_bn1')(d, training=False)
        if 'dec_drop1' in layer_names:
            d = ae.get_layer('dec_drop1')(d, training=False)

        # --- Second block: dec_fc2 → dec_bn2 ---
        d = ae.get_layer('dec_fc2')(d, training=False)
        if 'dec_bn2' in layer_names:
            d = ae.get_layer('dec_bn2')(d, training=False)

        # --- SNR re-injection (full model only) ---
        if 'dec_concat2' in layer_names:
            snr_t = tf.constant(snr_normed)
            d = tf.concat([d, snr_t], axis=-1)

        # --- Third block (if exists) ---
        if 'dec_fc3' in layer_names:
            d = ae.get_layer('dec_fc3')(d, training=False)
        if 'dec_bn3' in layer_names:
            d = ae.get_layer('dec_bn3')(d, training=False)
        if 'dec_drop2' in layer_names:
            d = ae.get_layer('dec_drop2')(d, training=False)
        if 'dec_fc4' in layer_names:
            d = ae.get_layer('dec_fc4')(d, training=False)

        # --- L output heads ---
        sym_preds = []
        for i in range(L):
            out_i = ae.get_layer(f'dec_sym{i}')(d, training=False)
            sym_preds.append(tf.argmax(out_i, axis=-1).numpy())

        det_batch = np.stack(sym_preds, axis=1)
        all_det.append(det_batch)

    return np.concatenate(all_det, axis=0)


def eval_ae_ber_cfo(encoder, ae, snr_arr, cfo_val,
                     M=16, L=4, N=100_000, trials=3,
                     a_sat=1.2, p_rapp=3,
                     alpha_pm=0.08, beta_pm=0.0):
    """Evaluate AE BER at fixed CFO across SNR range.
    Channel: PA → Rayleigh → CFO → AWGN → real decoder."""
    k = int(np.log2(M))
    eye_M = np.eye(M, dtype=np.float32)
    ber_list = []

    for snr_db in snr_arr:
        errs, bits = 0, 0
        for _ in range(trials):
            lbl = np.random.randint(0, M, (N, L))
            oh = eye_M[lbl]

            tx_IQ = encoder.predict(oh, verbose=0, batch_size=4096)
            tx_complex = tx_IQ[:, :, 0] + 1j * tx_IQ[:, :, 1]

            # Channel: PA → Rayleigh → CFO → AWGN
            rx_IQ, h_complex = _apply_channel_numpy(
                tx_complex, snr_db, cfo_val, L,
                a_sat, p_rapp, alpha_pm, beta_pm)

            # Real decoder (with CSI)
            det = _run_decoder(ae, rx_IQ, h_complex, snr_db, M, L)

            tx_b = (lbl[:, :, None] >> np.arange(k)) & 1
            rx_b = (det[:, :, None] >> np.arange(k)) & 1
            errs += np.sum(tx_b != rx_b)
            bits += tx_b.size

        ber_list.append(max(errs / bits, 1e-6))
        print(f'  SNR {snr_db:2.0f} dB, CFO={cfo_val:.3f} -> BER={ber_list[-1]:.2e}')
    return np.array(ber_list)


def eval_ae_ber_vs_cfo(encoder, ae, snr_val, cfo_arr,
                        M=16, L=4, N=100_000, trials=3,
                        a_sat=1.2, p_rapp=3,
                        alpha_pm=0.08, beta_pm=0.0):
    """Evaluate AE BER at fixed SNR across CFO range.
    Channel: PA → Rayleigh → CFO → AWGN → real decoder."""
    k = int(np.log2(M))
    eye_M = np.eye(M, dtype=np.float32)
    ber_list = []

    for cfo_val in cfo_arr:
        errs, bits = 0, 0
        for _ in range(trials):
            lbl = np.random.randint(0, M, (N, L))
            oh = eye_M[lbl]

            tx_IQ = encoder.predict(oh, verbose=0, batch_size=4096)
            tx_complex = tx_IQ[:, :, 0] + 1j * tx_IQ[:, :, 1]

            rx_IQ, h_complex = _apply_channel_numpy(
                tx_complex, snr_val, cfo_val, L,
                a_sat, p_rapp, alpha_pm, beta_pm)

            det = _run_decoder(ae, rx_IQ, h_complex, snr_val, M, L)

            tx_b = (lbl[:, :, None] >> np.arange(k)) & 1
            rx_b = (det[:, :, None] >> np.arange(k)) & 1
            errs += np.sum(tx_b != rx_b)
            bits += tx_b.size

        ber_list.append(max(errs / bits, 1e-6))
        print(f'  CFO={cfo_val:.4f}, SNR={snr_val:.0f} dB -> BER={ber_list[-1]:.2e}')
    return np.array(ber_list)


def compare_ber_cfo(encoder, ae, snr_arr, cfo_val,
                     M=16, L=4,
                     a_sat=1.2, p_rapp=3,
                     alpha_pm=0.08, beta_pm=0.0,
                     min_errors=200, max_symbols=5_000_000,
                     batch_size=200_000, verbose=True):
    """Fair Monte Carlo BER: AE system vs 16-QAM + ZF + slicer.

    Channel: PA → Rayleigh (block fading) → CFO → AWGN
    SAME symbols + SAME h + SAME noise for both systems.

    AE path:
      encoder → PA → h*x → CFO → AWGN → [rx, h, snr] → neural decoder → bits
    QAM path:
      QAM map → PA → h*x → CFO → AWGN → ZF(y/h) → grid slicer → bits
    """
    k = int(np.log2(M))
    eye_M = np.eye(M, dtype=np.float32)
    const = qam16_constellation()
    scale = 1.0 / np.sqrt(10.0)

    if verbose:
        print(f'  PA: Rapp p={p_rapp}, A_sat={a_sat}')
        print(f'  PA AM/PM: alpha={alpha_pm}, beta={beta_pm}')
        print(f'  CFO: {cfo_val:.4f} (normalized)')
        print(f'  Channel: PA + Rayleigh + CFO + AWGN')
        print()

    ber_ae_out, ber_qam_nocomp_out, ber_qam_perfect_out = [], [], []
    stats = []

    for snr_db in snr_arr:
        ae_err, qam_nocomp_err, qam_perfect_err = 0, 0, 0
        total_bits, n_sym = 0, 0
        sigma = np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))

        while True:
            n = min(batch_size, max_symbols - n_sym)
            if n <= 0:
                break

            # ── SAME random state ──
            tx_idx = np.random.randint(0, M, (n, L))
            h_complex = (np.random.randn(n, 1) + 1j * np.random.randn(n, 1))
            noise = (np.random.randn(n, L) + 1j * np.random.randn(n, L)) * sigma
            n_idx = np.arange(L).reshape(1, L)
            phase = 2.0 * np.pi * cfo_val * n_idx

            # ═══ AE path ═══
            oh = eye_M[tx_idx]
            ae_tx_IQ = encoder.predict(oh, verbose=0, batch_size=4096)
            ae_tx_complex = ae_tx_IQ[:, :, 0] + 1j * ae_tx_IQ[:, :, 1]
            ae_tx_pa = apply_pa_numpy(ae_tx_complex, a_sat, p_rapp, alpha_pm, beta_pm)
            ae_rx = h_complex * ae_tx_pa * np.exp(1j * phase) + noise
            ae_rx_IQ = np.stack([ae_rx.real, ae_rx.imag], axis=-1).astype(np.float32)
            ae_det = _run_decoder(ae, ae_rx_IQ, h_complex, snr_db, M, L)

            # ═══ QAM path (SAME h, SAME noise) ═══
            qam_tx = const[tx_idx.flatten()].reshape(n, L)
            qam_tx_pa = apply_pa_numpy(qam_tx, a_sat, p_rapp, alpha_pm, beta_pm)
            qam_rx = h_complex * qam_tx_pa * np.exp(1j * phase) + noise

            # QAM — ZF equalize, no CFO compensation, grid slicer
            qam_eq = qam_rx / (h_complex + 1e-10)
            qam_nocomp_det = _qam16_grid_slicer(qam_eq, scale)

            # QAM — perfect CFO de-rotation + ZF + grid slicer
            qam_rx_comp = qam_rx * np.exp(-1j * phase)
            qam_eq_comp = qam_rx_comp / (h_complex + 1e-10)
            qam_perfect_det = _qam16_grid_slicer(qam_eq_comp, scale)

            # ── Count bit errors ──
            tx_b = (tx_idx[:, :, None] >> np.arange(k)) & 1
            ae_rx_b = (ae_det[:, :, None] >> np.arange(k)) & 1
            qam_nc_rx_b = (qam_nocomp_det[:, :, None] >> np.arange(k)) & 1
            qam_pf_rx_b = (qam_perfect_det[:, :, None] >> np.arange(k)) & 1

            ae_err += int(np.sum(tx_b != ae_rx_b))
            qam_nocomp_err += int(np.sum(tx_b != qam_nc_rx_b))
            qam_perfect_err += int(np.sum(tx_b != qam_pf_rx_b))
            total_bits += tx_b.size
            n_sym += n

            if (ae_err >= min_errors and qam_nocomp_err >= min_errors):
                break

        b_ae = ae_err / total_bits if total_bits else 0
        b_qam_nc = qam_nocomp_err / total_bits if total_bits else 0
        b_qam_pf = qam_perfect_err / total_bits if total_bits else 0
        ber_ae_out.append(max(b_ae, 1e-7))
        ber_qam_nocomp_out.append(max(b_qam_nc, 1e-7))
        ber_qam_perfect_out.append(max(b_qam_pf, 1e-7))

        stats.append({
            'snr': snr_db,
            'ae_ber': b_ae, 'ae_err': ae_err,
            'qam_nocomp_ber': b_qam_nc, 'qam_nocomp_err': qam_nocomp_err,
            'total_bits': total_bits, 'n_sym': n_sym,
        })

        if verbose:
            gain_nc = (10 * np.log10(b_qam_nc / b_ae)
                       if b_ae > 1e-7 and b_qam_nc > 1e-7 else float('nan'))
            print(f'  SNR {snr_db:5.1f} dB | '
                  f'AE {b_ae:.3e} ({ae_err:,}) | '
                  f'QAM+ZF {b_qam_nc:.3e} ({qam_nocomp_err:,}) | '
                  f'Gain {gain_nc:+.2f} dB | {n_sym:,} syms')

    return (np.array(ber_ae_out),
            np.array(ber_qam_nocomp_out),
            np.array(ber_qam_perfect_out),
            stats)


# ============================================================
# 8. Visualization
# ============================================================

def get_learned_constellation(encoder, M=16, L=4):
    """Extract learned 16-point constellation from encoder."""
    test_in = np.zeros((M, L, M), dtype=np.float32)
    for i in range(M):
        test_in[i, 0, i] = 1.0
    pts = encoder.predict(test_in, verbose=0)[:, 0, :]
    return pts


def plot_cfo_effect(cfo_val=0.02, L=8):
    """Visualize CFO phase rotation on 16-QAM constellation."""
    const = qam16_constellation()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, L))

    # Original constellation
    axes[0].scatter(const.real, const.imag, c='blue', s=150,
                    edgecolors='k', zorder=5)
    axes[0].set_title('16-QAM (no CFO)', fontweight='bold')
    axes[0].set_xlim(-2, 2); axes[0].set_ylim(-2, 2)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    # Single symbol at different time indices
    sym_idx = 5  # pick one symbol
    axes[1].scatter(const.real, const.imag, c='lightgray', s=80, alpha=0.5)
    for n in range(L):
        phase = 2 * np.pi * cfo_val * n
        rotated = const[sym_idx] * np.exp(1j * phase)
        axes[1].scatter(rotated.real, rotated.imag, c=[colors[n]], s=120,
                       edgecolors='k', zorder=5, label=f'n={n}')
        if n > 0:
            prev = const[sym_idx] * np.exp(1j * 2 * np.pi * cfo_val * (n-1))
            axes[1].plot([prev.real, rotated.real],
                        [prev.imag, rotated.imag], 'r-', alpha=0.4)
    axes[1].set_title(f'Symbol #{sym_idx} across time\n'
                      f'(CFO={cfo_val})', fontweight='bold')
    axes[1].legend( ncol=2)
    axes[1].set_xlim(-2, 2); axes[1].set_ylim(-2, 2)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    # Full constellation at last symbol (max rotation)
    phase_max = 2 * np.pi * cfo_val * (L-1)
    rotated_all = const * np.exp(1j * phase_max)
    axes[2].scatter(const.real, const.imag, c='gray', s=80,
                    alpha=0.3, marker='x', label='Original')
    axes[2].scatter(rotated_all.real, rotated_all.imag, c='red', s=150,
                    edgecolors='k', zorder=5, label=f'n={L-1}')
    for i in range(16):
        axes[2].plot([const[i].real, rotated_all[i].real],
                    [const[i].imag, rotated_all[i].imag], 'r-', alpha=0.3)
    axes[2].set_title(f'All symbols at n={L-1}\n'
                      f'(phase={np.degrees(phase_max):.1f}°)',
                      fontweight='bold', color='red')
    axes[2].legend()
    axes[2].set_xlim(-2, 2); axes[2].set_ylim(-2, 2)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_constellation_learned(encoder, M=16, L=4, a_sat=1.2, p_rapp=3, alpha_pm=0.08, beta_pm=0.0):
    """Plot learned constellation before/after PA vs standard 16-QAM."""
    pts_raw = get_learned_constellation(encoder, M, L)
    pts = pts_raw[:, 0] + 1j * pts_raw[:, 1]  # convert (M,2) to complex (M,)
    pts_pa = apply_pa_numpy(pts, a_sat, p_rapp, alpha_pm, beta_pm)
    
    qam = qam16_constellation()
    qam_pa = apply_pa_numpy(qam, a_sat, p_rapp, alpha_pm, beta_pm)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, M))
    
    for i in range(M):
        axes[0].scatter(pts[i].real, pts[i].imag, c=[colors[i]], s=200, edgecolors='k', linewidth=1.5, zorder=5)
        axes[0].annotate(f'{i}', (pts[i].real, pts[i].imag), ha='center', va='center', fontsize=9, fontweight='bold')

    c0 = plt.Circle((0,0), a_sat, fill=False, color='red', ls='--')
    axes[0].add_patch(c0)
    axes[0].set_title('Learned (before PA)', fontweight='bold', color='#2A9D8F')
    axes[0].set_xlim(-2,2); axes[0].set_ylim(-2,2)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

    for i in range(M):
        axes[1].scatter(pts_pa[i].real, pts_pa[i].imag, c=[colors[i]], s=200,
                       edgecolors='k', linewidth=1.5, zorder=5)
    axes[1].set_title('Learned (after PA)', fontweight='bold', color='#E63946')
    axes[1].set_xlim(-2,2); axes[1].set_ylim(-2,2)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    for i in range(M):
        axes[2].scatter(qam_pa[i].real, qam_pa[i].imag, c=[colors[i]], s=200,
                       edgecolors='k', linewidth=1.5, zorder=5)
    axes[2].set_title('16-QAM (after PA)', fontweight='bold', color='#457B9D')
    axes[2].set_xlim(-2,2); axes[2].set_ylim(-2,2)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

    pass # suptitle removed
    plt.tight_layout()
    return fig


def plot_ber_vs_snr(snr, results_dict, title_suffix=''):
    """Plot BER vs SNR for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 7))
    styles = [
        ('o-', '#E63946', 2.5), ('s--', '#457B9D', 2.5),
        ('^--', '#2A9D8F', 2.0), ('D:', '#E9C46A', 2.0),
        ('v-', '#264653', 2.0),
    ]
    for idx, (name, ber) in enumerate(results_dict.items()):
        st = styles[idx % len(styles)]
        ax.semilogy(snr, ber, st[0], color=st[1], lw=st[2], ms=8,
                    label=name, mfc='white', mew=2)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('BER')
    ax.set_title(f'BER vs SNR{title_suffix}',
                  fontweight='bold')
    ax.legend( loc='lower left')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(snr[0], snr[-1]); ax.set_ylim(1e-5, 1)
    plt.tight_layout()
    return fig


def plot_ber_vs_cfo(cfo_arr, results_dict, snr_val=15):
    """Plot BER vs CFO at fixed SNR."""
    fig, ax = plt.subplots(figsize=(10, 7))
    styles = [
        ('o-', '#E63946', 2.5), ('s--', '#457B9D', 2.5),
        ('^--', '#2A9D8F', 2.0),
    ]
    for idx, (name, ber) in enumerate(results_dict.items()):
        st = styles[idx % len(styles)]
        ax.semilogy(cfo_arr, ber, st[0], color=st[1], lw=st[2], ms=8,
                    label=name, mfc='white', mew=2)
    ax.set_xlabel('Normalized CFO (Δf·Ts)')
    ax.set_ylabel('BER')
    ax.set_title(f'BER vs CFO @ SNR={snr_val} dB\n'
                 f'(Rayleigh + PA + AWGN)',
                  fontweight='bold')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-5, 1)
    plt.tight_layout()
    return fig


def plot_history(hist):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist['loss'], lw=2, color='#E63946', label='Train')
    axes[0].plot(hist['val_loss'], lw=2, color='#457B9D', label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist['symbol_accuracy'], lw=2, color='#E63946', label='Train')
    axes[1].plot(hist['val_symbol_accuracy'], lw=2, color='#457B9D', label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Symbol Accuracy')
    axes[1].set_title('Per-Symbol Accuracy', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    pass # suptitle removed
    plt.tight_layout()
    return fig
