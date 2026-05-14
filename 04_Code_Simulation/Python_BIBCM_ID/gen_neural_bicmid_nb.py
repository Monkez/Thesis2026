import json
import os


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip().splitlines(True),
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip().splitlines(True),
    }


cells = []

cells.append(md(r"""
# Robust Residual Neural Demapper cho BICM-ID dưới kênh PA không biết chính xác

Notebook này xây dựng một hướng nghiên cứu có thể đưa vào báo cáo:

- Hệ thống: **BICM-ID, 16-QAM MSEW, Extended Hamming (8,4), interleaver 4096 bits**.
- Kênh khảo sát: **Rapp PA + AWGN**, nhưng tham số PA **không được đưa vào receiver đề xuất**.
- Training: sinh dữ liệu với nhiều PA ngẫu nhiên trong một dải tham số.
- Testing: đánh giá trên các PA khác nhau, gồm cả PA nằm ngoài/ven dải training.
- Oracle: demapper classic biết đúng PA test, chỉ dùng làm mốc trần tham chiếu.
- Đề xuất: **robust residual neural demapper** học cách sửa LLR từ dữ liệu, kèm **LLR calibration**.

Ý tưởng chính:

```text
Le_proposed = a(N0, iteration) * (Le_mismatch + Delta_NN(rx, La, N0, Le_mismatch)) + b(N0, iteration)
```

Trong đó `Le_mismatch` là LLR từ demapper classic dùng constellation gốc. `Delta_NN` học phần sửa sai do nhiều loại PA gây ra, nhưng **không nhận trực tiếp a_sat, p_rapp hay AM/PM** tại inference.
"""))

cells.append(code(r"""
import os
import sys
import time
import json
import itertools
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 12, "figure.figsize": (10, 6)})
%matplotlib inline

# Chạy notebook từ thư mục python_code hoặc repo root đều được.
repo_root = os.path.abspath("..") if os.path.basename(os.getcwd()) == "python_code" else os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_code.modulation import build_constellation, modulate_bits, symbol_bit_matrix
from python_code.demodulation import softdemod_qam, soft_demapper
from python_code.encoders import HammingEncoder
from python_code.decoders import DualDecoder
from python_code.interleavers import load_interleaver

np.random.seed(7)
tf.random.set_seed(7)

BPS = 4
CODE_RATE = 0.5
ES_NORM = 1.0
MAPRULE_MSEW = [11, 2, 1, 12, 4, 9, 10, 3, 5, 16, 15, 6, 14, 7, 8, 13]

def snr_to_N0(dB):
    snr_lin = 10 ** (np.asarray(dB, dtype=np.float32) / 10.0) * BPS
    n0_uncoded = 1.0 / snr_lin
    return n0_uncoded / CODE_RATE

S_ORIG = build_constellation(16, MAPRULE_MSEW, 10.0)
SYM_MATRIX = symbol_bit_matrix(16)

print("Project root:", repo_root)
print("Es original:", np.mean(np.abs(S_ORIG) ** 2))
"""))

cells.append(md(r"""
## 1. Mô hình PA và bộ kịch bản robust

PA Rapp là méo phi tuyến không nhớ. Với 16-QAM, các điểm biên độ lớn bị nén mạnh hơn các điểm gần tâm. Nếu receiver không biết đúng PA, demapper dùng `S_ORIG` sẽ bị mismatch.

Trong notebook này:

- `TRAIN_PA_RANGE`: dải PA dùng để sinh dữ liệu train.
- `TEST_PA_SCENARIOS`: các PA dùng để test BER.
- `pa_aware` là oracle vì nó dùng đúng `S_PA` của từng PA test.
- `proposed` không nhận tham số PA, chỉ nhận `rx_iq, La, N0, Le_mismatch, iteration`.
"""))

cells.append(code(r"""
def rapp_pa_complex(x, a_sat=1.20, p_rapp=2.0, alpha_pm=0.0, beta_pm=0.0):
    x = np.asarray(x, dtype=np.complex128)
    r = np.abs(x)
    theta = np.angle(x)
    ratio = r / a_sat
    g_r = r / ((1.0 + ratio ** (2.0 * p_rapp)) ** (1.0 / (2.0 * p_rapp)))
    phi_pa = alpha_pm * r**2 / (1.0 + beta_pm * r**2)
    return g_r * np.exp(1j * (theta + phi_pa))

def awgn_complex(x, N0):
    noise = np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
    return x + np.sqrt(N0 / 2.0) * noise

TRAIN_PA_RANGE = {
    "a_sat": (1.05, 1.55),
    "p_rapp": (1.5, 3.5),
    "alpha_pm": (0.00, 0.08),
    "beta_pm": (0.00, 0.00),
}

TEST_PA_SCENARIOS = {
    "mild_seen": {"a_sat": 1.45, "p_rapp": 2.5, "alpha_pm": 0.02, "beta_pm": 0.0},
    "strong_seen": {"a_sat": 1.12, "p_rapp": 2.0, "alpha_pm": 0.06, "beta_pm": 0.0},
    "edge_unseen": {"a_sat": 0.98, "p_rapp": 1.35, "alpha_pm": 0.10, "beta_pm": 0.0},
}

def sample_pa_cfg():
    return {
        "a_sat": float(np.random.uniform(*TRAIN_PA_RANGE["a_sat"])),
        "p_rapp": float(np.random.uniform(*TRAIN_PA_RANGE["p_rapp"])),
        "alpha_pm": float(np.random.uniform(*TRAIN_PA_RANGE["alpha_pm"])),
        "beta_pm": float(np.random.uniform(*TRAIN_PA_RANGE["beta_pm"])),
    }

def constellation_after_pa(pa_cfg):
    return rapp_pa_complex(S_ORIG, **pa_cfg)

DEMO_PA_CFG = TEST_PA_SCENARIOS["strong_seen"]
S_DEMO_PA = constellation_after_pa(DEMO_PA_CFG)

def iq_from_complex(x):
    return np.stack([np.real(x), np.imag(x)], axis=-1).astype(np.float32)

all_bits = np.array(list(itertools.product([0, 1], repeat=4)), dtype=np.float32)
bit_labels = ["".join(map(str, b.astype(int))) for b in all_bits]

plt.figure(figsize=(6.5, 6.5))
plt.scatter(S_ORIG.real, S_ORIG.imag, s=90, label="Original 16-QAM MSEW", alpha=0.55)
plt.scatter(S_DEMO_PA.real, S_DEMO_PA.imag, s=90, label="Sau PA demo", marker="s")
for i, lab in enumerate(bit_labels):
    plt.annotate(lab, (S_DEMO_PA.real[i], S_DEMO_PA.imag[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
plt.axhline(0, color="black", lw=0.8)
plt.axvline(0, color="black", lw=0.8)
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.title("Constellation trước và sau PA")
plt.legend()
plt.show()

print("Es trước PA:", np.mean(np.abs(S_ORIG) ** 2))
print("Es sau PA demo:", np.mean(np.abs(S_DEMO_PA) ** 2))
print("Train PA range:", TRAIN_PA_RANGE)
print("Test PA scenarios:", TEST_PA_SCENARIOS)
"""))

cells.append(md(r"""
## 2. Sinh dữ liệu train residual demapper

Mục tiêu train không phải là thay thế hoàn toàn demapper. Với mỗi batch, ta random một PA trong `TRAIN_PA_RANGE`, sau đó dùng **teacher PA-aware classic demapper** tương ứng với PA đó:

```text
target = Le_PA_aware
base   = Le_mismatch
residual_target = target - base
```

NN chỉ học `residual_target`. Ở inference, mạng không biết PA test. Nó chỉ dùng các dấu hiệu trong tín hiệu nhận và LLR nền để sửa sai.
"""))

cells.append(code(r"""
def random_apriori_llr(bits, sigma_min=0.45, sigma_max=4.0, drop_prob=0.25):
    sigma = np.random.uniform(sigma_min, sigma_max, size=(bits.shape[0], 1)).astype(np.float32)
    bpsk = 2.0 * bits.astype(np.float32) - 1.0
    ap_rx = bpsk + np.random.randn(*bits.shape).astype(np.float32) * sigma
    La = 2.0 * ap_rx / (sigma**2 + 1e-8)
    drop = (np.random.rand(bits.shape[0], 1) > drop_prob).astype(np.float32)
    return (La * drop).astype(np.float32)

def demap_frame(rx, La_flat, N0, S_ref):
    metrics = softdemod_qam(rx, float(N0), S_ref)
    return soft_demapper(metrics, La_flat.astype(np.float64), SYM_MATRIX).astype(np.float32)

def generate_teacher_batch(num_symbols=2048, snr_min=0.0, snr_max=7.0):
    pa_cfg = sample_pa_cfg()
    S_pa = constellation_after_pa(pa_cfg)

    bits = np.random.randint(0, 2, size=(num_symbols, BPS)).astype(np.float32)
    idx = (bits[:, 0] * 8 + bits[:, 1] * 4 + bits[:, 2] * 2 + bits[:, 3]).astype(np.int64)
    tx = S_ORIG[idx]
    tx_pa = rapp_pa_complex(tx, **pa_cfg)

    snr_db = np.random.uniform(snr_min, snr_max)
    N0 = float(snr_to_N0(snr_db))
    rx = awgn_complex(tx_pa, N0)

    La = random_apriori_llr(bits)
    La_flat = La.reshape(-1)

    Le_mismatch = demap_frame(rx, La_flat, N0, S_ORIG).reshape(-1, BPS)
    Le_teacher = demap_frame(rx, La_flat, N0, S_pa).reshape(-1, BPS)

    rx_iq = iq_from_complex(rx)
    N0_col = np.full((num_symbols, 1), N0, dtype=np.float32)
    snr_col = np.full((num_symbols, 1), snr_db, dtype=np.float32)

    return {
        "bits": bits.astype(np.float32),
        "rx_iq": rx_iq,
        "La": La.astype(np.float32),
        "N0": N0_col,
        "snr_db": snr_col,
        "Le_mismatch": Le_mismatch.astype(np.float32),
        "Le_teacher": Le_teacher.astype(np.float32),
        "residual": (Le_teacher - Le_mismatch).astype(np.float32),
        "pa_cfg": pa_cfg,
    }

sample = generate_teacher_batch(128)
for k, v in sample.items():
    if isinstance(v, dict):
        print(k, v)
    else:
        print(k, v.shape, v.dtype)
print("Residual mean/std:", sample["residual"].mean(), sample["residual"].std())
"""))

cells.append(md(r"""
## 3. Residual neural demapper và LLR calibration

Model gồm hai phần:

- `ResidualDemapper`: dự đoán `Delta_NN`.
- `LLRCalibration`: học scale/offset theo `N0` và `iteration`.

Trong training teacher-level, iteration được random từ 1 đến 10 để calibration học nhiều mức lặp. Khi mô phỏng BICM-ID, ta truyền đúng iteration hiện tại.
"""))

cells.append(code(r"""
class ResidualDemapper(tf.keras.Model):
    def __init__(self, m_bits=4, hidden=128):
        super().__init__()
        self.m_bits = m_bits
        self.backbone = tf.keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(m_bits, activation="linear"),
        ])

    def call(self, rx_iq, La, N0, Le_base, iteration, training=False):
        log_N0 = tf.math.log(N0 + 1e-8) / 3.0
        it_norm = tf.cast(iteration, tf.float32) / 10.0
        radius = tf.sqrt(tf.reduce_sum(tf.square(rx_iq), axis=-1, keepdims=True) + 1e-8)
        phase_feat = tf.concat([
            tf.math.sin(tf.atan2(rx_iq[:, 1:2], rx_iq[:, 0:1])),
            tf.math.cos(tf.atan2(rx_iq[:, 1:2], rx_iq[:, 0:1])),
        ], axis=-1)
        x = tf.concat([
            rx_iq,
            radius,
            phase_feat,
            tf.clip_by_value(La, -12.0, 12.0),
            tf.clip_by_value(Le_base, -20.0, 20.0),
            log_N0,
            it_norm,
        ], axis=-1)
        return self.backbone(x, training=training)

class LLRCalibration(tf.keras.Model):
    def __init__(self, m_bits=4):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(2 * m_bits, activation="linear"),
        ])
        self.m_bits = m_bits

    def call(self, Le, N0, iteration, training=False):
        log_N0 = tf.math.log(N0 + 1e-8) / 3.0
        it_norm = tf.cast(iteration, tf.float32) / 10.0
        params = self.net(tf.concat([log_N0, it_norm], axis=-1), training=training)
        raw_scale, bias = tf.split(params, 2, axis=-1)
        scale = 0.25 + tf.nn.softplus(raw_scale)
        return scale * Le + bias

class ProposedDemapper(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.residual = ResidualDemapper()
        self.calibration = LLRCalibration()

    def call(self, rx_iq, La, N0, Le_base, iteration, training=False):
        delta = self.residual(rx_iq, La, N0, Le_base, iteration, training=training)
        Le_uncal = Le_base + delta
        return self.calibration(Le_uncal, N0, iteration, training=training), delta

model = ProposedDemapper()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Build model variables.
dummy = generate_teacher_batch(8)
_ = model(
    tf.constant(dummy["rx_iq"]),
    tf.constant(dummy["La"]),
    tf.constant(dummy["N0"]),
    tf.constant(dummy["Le_mismatch"]),
    tf.ones((8, 1), dtype=tf.float32),
)
print("Trainable variables:", len(model.trainable_variables))
"""))

cells.append(md(r"""
## 4. Huấn luyện

Loss gồm 2 thành phần:

- `residual_loss`: ép `Delta_NN` học phần sai lệch do PA.
- `calibrated_loss`: ép LLR sau calibration khớp teacher PA-aware.

Sau khi train, lưu checkpoint để chạy lại mô phỏng BER mà không cần train lại.
"""))

cells.append(code(r"""
QUICK_TRAIN = True
TRAIN_STEPS = 200 if QUICK_TRAIN else 800
BATCH_SYMBOLS = 1024 if QUICK_TRAIN else 2048
LOG_EVERY = 50
CKPT_PATH = os.path.join(repo_root, "python_code", "robust_pa_residual_demapper.weights.h5")

train_log = []
start = time.time()

for step in range(1, TRAIN_STEPS + 1):
    batch = generate_teacher_batch(BATCH_SYMBOLS)
    iteration = np.random.randint(1, 11, size=(BATCH_SYMBOLS, 1)).astype(np.float32)

    rx_iq = tf.constant(batch["rx_iq"])
    La = tf.constant(batch["La"])
    N0 = tf.constant(batch["N0"])
    Le_base = tf.constant(batch["Le_mismatch"])
    Le_teacher = tf.constant(batch["Le_teacher"])
    residual_target = tf.constant(batch["residual"])
    it_tf = tf.constant(iteration)

    with tf.GradientTape() as tape:
        Le_cal, delta = model(rx_iq, La, N0, Le_base, it_tf, training=True)
        residual_loss = tf.reduce_mean(tf.keras.losses.huber(residual_target, delta, delta=2.0))
        calibrated_loss = tf.reduce_mean(tf.keras.losses.huber(Le_teacher, Le_cal, delta=2.0))
        loss = 0.65 * residual_loss + 0.35 * calibrated_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % LOG_EVERY == 0 or step == 1:
        elapsed = time.time() - start
        rec = {
            "step": step,
            "loss": float(loss.numpy()),
            "residual_loss": float(residual_loss.numpy()),
            "calibrated_loss": float(calibrated_loss.numpy()),
            "elapsed_s": elapsed,
        }
        train_log.append(rec)
        print(
            f"step {step:4d}/{TRAIN_STEPS} | "
            f"loss={rec['loss']:.4f} residual={rec['residual_loss']:.4f} "
            f"cal={rec['calibrated_loss']:.4f} | {elapsed:.1f}s"
        )

model.save_weights(CKPT_PATH)
print("Saved:", CKPT_PATH)

plt.figure(figsize=(8, 4))
plt.plot([r["step"] for r in train_log], [r["loss"] for r in train_log], "-o", label="total")
plt.plot([r["step"] for r in train_log], [r["residual_loss"] for r in train_log], "-o", label="residual")
plt.plot([r["step"] for r in train_log], [r["calibrated_loss"] for r in train_log], "-o", label="calibrated")
plt.grid(True, alpha=0.3)
plt.xlabel("Training step")
plt.ylabel("Huber loss")
plt.title("Training loss")
plt.legend()
plt.show()
"""))

cells.append(md(r"""
## 5. Kiểm tra chất lượng LLR ở mức symbol

Đây là kết quả trung gian hữu ích cho báo cáo: residual neural demapper có làm LLR gần PA-aware teacher hơn baseline mismatch không?
"""))

cells.append(code(r"""
def evaluate_symbol_level(num_batches=8, num_symbols=1024):
    mse_base = []
    mse_prop = []
    corr_base = []
    corr_prop = []
    for _ in range(num_batches):
        b = generate_teacher_batch(num_symbols)
        it = np.random.randint(1, 11, size=(num_symbols, 1)).astype(np.float32)
        Le_cal, _ = model(
            tf.constant(b["rx_iq"]),
            tf.constant(b["La"]),
            tf.constant(b["N0"]),
            tf.constant(b["Le_mismatch"]),
            tf.constant(it),
            training=False,
        )
        prop = Le_cal.numpy()
        teacher = b["Le_teacher"]
        base = b["Le_mismatch"]
        mse_base.append(np.mean((base - teacher) ** 2))
        mse_prop.append(np.mean((prop - teacher) ** 2))
        corr_base.append(np.corrcoef(base.reshape(-1), teacher.reshape(-1))[0, 1])
        corr_prop.append(np.corrcoef(prop.reshape(-1), teacher.reshape(-1))[0, 1])
    return {
        "MSE mismatch": np.mean(mse_base),
        "MSE proposed": np.mean(mse_prop),
        "Corr mismatch": np.mean(corr_base),
        "Corr proposed": np.mean(corr_prop),
    }

symbol_metrics = evaluate_symbol_level()
for k, v in symbol_metrics.items():
    print(f"{k:16s}: {v:.5f}")

plt.figure(figsize=(5, 4))
plt.bar(["Mismatch", "Proposed"], [symbol_metrics["MSE mismatch"], symbol_metrics["MSE proposed"]])
plt.ylabel("MSE to PA-aware teacher")
plt.title("Symbol-level LLR error")
plt.grid(True, axis="y", alpha=0.3)
plt.show()
"""))

cells.append(md(r"""
## 6. Tích hợp vào vòng lặp BICM-ID

Ba chế độ demapper:

- `mismatch`: classic demapper dùng `S_ORIG` dù kênh có PA.
- `pa_aware`: oracle classic demapper dùng `S_PA` đúng của từng kịch bản test.
- `proposed`: `mismatch + residual NN + calibration`.
"""))

cells.append(code(r"""
hamming_m = 3
encoder = HammingEncoder(hamming_m)
decoder = DualDecoder(encoder.H)
alpha = load_interleaver("BIBCM-ID_4096Algeb.mat")

num_channel_bits = len(alpha)
n_code = encoder.n_code
k_info = encoder.k_info
frame_len = num_channel_bits // n_code
info_len = frame_len * k_info
scaling_factor = 0.85

print("n_code, k_info:", n_code, k_info)
print("interleaver bits:", num_channel_bits, "symbols:", num_channel_bits // BPS)

def proposed_demap_numpy(rx, La_interleaved, N0, iteration):
    Le_base = demap_frame(rx, La_interleaved.reshape(-1), N0, S_ORIG).reshape(-1, BPS)
    rx_iq = iq_from_complex(rx)
    nsym = rx_iq.shape[0]
    Le_cal, _ = model(
        tf.constant(rx_iq, dtype=tf.float32),
        tf.constant(La_interleaved.astype(np.float32)),
        tf.constant(np.full((nsym, 1), N0, dtype=np.float32)),
        tf.constant(Le_base.astype(np.float32)),
        tf.constant(np.full((nsym, 1), iteration + 1, dtype=np.float32)),
        training=False,
    )
    return Le_cal.numpy().reshape(-1)

def simulate_hamming_frame_pa(snr_dB, pa_cfg, mode="mismatch", max_iter=10):
    S_pa_oracle = constellation_after_pa(pa_cfg)

    u = np.random.randint(0, 2, info_len)
    v = encoder.encode_frame(u)
    vv = v[alpha]

    tx = modulate_bits(vv, S_ORIG)
    tx_pa = rapp_pa_complex(tx, **pa_cfg)
    N0 = float(snr_to_N0(snr_dB))
    rx = awgn_complex(tx_pa, N0)

    La_bits = np.zeros(num_channel_bits, dtype=np.float32)
    bit_errors = np.zeros(max_iter, dtype=np.float64)

    for iteration in range(max_iter):
        La_interleaved = La_bits[alpha].reshape(-1, BPS).astype(np.float32)

        if mode == "mismatch":
            Le_ext = demap_frame(rx, La_interleaved.reshape(-1), N0, S_ORIG)
        elif mode == "pa_aware":
            Le_ext = demap_frame(rx, La_interleaved.reshape(-1), N0, S_pa_oracle)
        elif mode == "proposed":
            Le_ext = proposed_demap_numpy(rx, La_interleaved, N0, iteration)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        Lc = np.zeros(num_channel_bits, dtype=np.float32)
        Lc[alpha] = Le_ext.astype(np.float32)

        Lc_post = decoder.decode_frame(Lc)
        La_bits = scaling_factor * (Lc_post - Lc)

        vhat = ((np.sign(Lc_post) + 1) / 2).astype(int)
        bit_errors[iteration] = np.sum(v != vhat)

    return bit_errors, num_channel_bits

print("Receiver functions are ready.")
print("Quick check example: simulate_hamming_frame_pa(4.0, TEST_PA_SCENARIOS['mild_seen'], mode='proposed', max_iter=2)")
"""))

cells.append(md(r"""
## 7. Mô phỏng BER

Để chạy nhanh trong notebook, cấu hình mặc định dùng số block vừa phải. Khi lấy số liệu báo cáo cuối cùng, tăng `BLOCKS_BY_SNR` hoặc đặt tiêu chí dừng theo số block lỗi.
"""))

cells.append(code(r"""
dB_range = [0, 1, 2, 3, 4, 4.5, 5, 5.5, 6]
MAX_ITER = 10

# Draft mode tạo kết quả nhanh để kiểm tra pipeline. Khi lấy số liệu báo cáo,
# đổi QUICK_RUN = False hoặc tăng trực tiếp BLOCKS_BY_SNR.
QUICK_RUN = True

BLOCKS_DRAFT = {
    0: 5,
    1: 5,
    2: 8,
    3: 10,
    4: 12,
    4.5: 15,
    5: 20,
    5.5: 25,
    6: 30,
}

BLOCKS_REPORT = {
    0: 80,
    1: 80,
    2: 120,
    3: 160,
    4: 250,
    4.5: 400,
    5: 700,
    5.5: 900,
    6: 1200,
}

BLOCKS_BY_SNR = BLOCKS_DRAFT if QUICK_RUN else BLOCKS_REPORT

def simulate_ber_scenarios(modes=("mismatch", "proposed", "pa_aware")):
    results = {
        scenario: {mode: np.zeros((MAX_ITER, len(dB_range))) for mode in modes}
        for scenario in TEST_PA_SCENARIOS
    }
    report_rows = []
    for scenario, pa_cfg in TEST_PA_SCENARIOS.items():
        print("\nScenario:", scenario, pa_cfg)
        for mode in modes:
            print("  Mode:", mode)
            for z, snr in enumerate(dB_range):
                max_blocks = BLOCKS_BY_SNR[snr]
                total_errs = np.zeros(MAX_ITER, dtype=np.float64)
                total_bits = 0
                start = time.time()
                for blk in range(max_blocks):
                    errs, nbits = simulate_hamming_frame_pa(snr, pa_cfg, mode=mode, max_iter=MAX_ITER)
                    total_errs += errs
                    total_bits += nbits
                ber = total_errs / total_bits
                results[scenario][mode][:, z] = ber
                row = {
                    "scenario": scenario,
                    "mode": mode,
                    "snr_db": snr,
                    "blocks": max_blocks,
                    "bits": total_bits,
                    "ber_iter1": float(ber[0]),
                    "ber_iter5": float(ber[4]),
                    "ber_iter10": float(ber[9]),
                    "elapsed_s": time.time() - start,
                }
                report_rows.append(row)
                print(
                    f"    SNR={snr:>4} dB | "
                    f"Iter1={ber[0]:.3e} Iter5={ber[4]:.3e} Iter10={ber[9]:.3e} | "
                    f"blocks={max_blocks} time={row['elapsed_s']:.1f}s"
                )
    return results, report_rows

BER_RESULTS, REPORT_ROWS = simulate_ber_scenarios()
"""))

cells.append(code(r"""
def save_report_artifacts():
    out_dir = os.path.join(repo_root, "python_code", "robust_pa_residual_results")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ber_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = ["scenario", "mode", "snr_db", "blocks", "bits", "ber_iter1", "ber_iter5", "ber_iter10", "elapsed_s"]
        f.write(",".join(headers) + "\n")
        for row in REPORT_ROWS:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    npz_path = os.path.join(out_dir, "ber_curves.npz")
    flat_curves = {}
    for scenario, per_mode in BER_RESULTS.items():
        for mode, curve in per_mode.items():
            flat_curves[f"{scenario}__{mode}"] = curve
    np.savez(npz_path, dB_range=np.array(dB_range), **flat_curves)

    def json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    json_path = os.path.join(out_dir, "config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "TRAIN_PA_RANGE": TRAIN_PA_RANGE,
            "TEST_PA_SCENARIOS": TEST_PA_SCENARIOS,
            "MAPRULE_MSEW": MAPRULE_MSEW,
            "dB_range": dB_range,
            "QUICK_TRAIN": QUICK_TRAIN,
            "TRAIN_STEPS": TRAIN_STEPS,
            "BATCH_SYMBOLS": BATCH_SYMBOLS,
            "QUICK_RUN": QUICK_RUN,
            "BLOCKS_BY_SNR": BLOCKS_BY_SNR,
            "symbol_metrics": symbol_metrics,
        }), f, indent=2)

    return out_dir, csv_path, npz_path, json_path

styles = {
    "mismatch": ("--o", "Classic mismatch, S_ORIG"),
    "pa_aware": ("-s", "Oracle PA-aware"),
    "proposed": ("-^", "Proposed residual + calibration"),
}

for scenario in TEST_PA_SCENARIOS:
    plt.figure(figsize=(10, 6))
    for mode, (style, label) in styles.items():
        plt.semilogy(dB_range, np.maximum(BER_RESULTS[scenario][mode][9], 1e-7), style, lw=2, label=label)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER Iter10 - {scenario}")
    plt.ylim(1e-6, 1)
    plt.legend()
    plt.show()

for scenario in TEST_PA_SCENARIOS:
    plt.figure(figsize=(10, 6))
    for mode, (style, label) in styles.items():
        plt.semilogy(dB_range, np.maximum(BER_RESULTS[scenario][mode][0], 1e-7), style, lw=2, label=label + " - Iter1")
        plt.semilogy(dB_range, np.maximum(BER_RESULTS[scenario][mode][4], 1e-7), style, lw=1.4, alpha=0.55, label=label + " - Iter5")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"Iterative gain - {scenario}")
    plt.ylim(1e-6, 1)
    plt.legend(fontsize=9)
    plt.show()

out_dir, csv_path, npz_path, json_path = save_report_artifacts()
print("Saved artifacts:")
print(csv_path)
print(npz_path)
print(json_path)
"""))

cells.append(md(r"""
## 8. Bảng kết quả cho báo cáo

Cell này in bảng tóm tắt có thể chép trực tiếp vào báo cáo. Khi cần kết quả chính thức, tăng số block và chạy lại từ phần mô phỏng BER.
"""))

cells.append(code(r"""
try:
    import pandas as pd
    df = pd.DataFrame(REPORT_ROWS)
    display(df[["scenario", "mode", "snr_db", "blocks", "ber_iter1", "ber_iter5", "ber_iter10"]])

    pivot = df.pivot_table(index=["scenario", "snr_db"], columns="mode", values="ber_iter10")
    display(pivot)
except Exception:
    for row in REPORT_ROWS:
        print(
            f"{row['scenario']:12s} | {row['mode']:10s} | SNR={row['snr_db']:>4} | "
            f"Iter1={row['ber_iter1']:.3e} Iter5={row['ber_iter5']:.3e} Iter10={row['ber_iter10']:.3e}"
        )
"""))

cells.append(md(r"""
## 9. Gợi ý diễn giải kết quả

Khi viết báo cáo, nên trình bày theo logic:

1. PA làm constellation sau kênh lệch khỏi 16-QAM gốc, nên demapper AWGN bị mismatch.
2. Nếu receiver biết chính xác PA, PA-aware demapper là oracle rất mạnh. Vì vậy nó chỉ nên được xem là mốc trần, không phải baseline thực tế.
3. Bài toán nghiên cứu là receiver **không biết PA chính xác**. Residual NN được train trên nhiều PA ngẫu nhiên và không nhận tham số PA khi inference.
4. Nếu proposed tốt hơn mismatch trên nhiều test scenario, đặc biệt ở `edge_unseen`, điều đó cho thấy mạng học được hiệu chỉnh LLR robust từ dữ liệu.
5. LLR calibration giúp kiểm soát độ tin cậy LLR, yếu tố quan trọng trong iterative decoding.
6. So sánh BER Iter1, Iter5, Iter10 để chứng minh không chỉ symbol demapping tốt hơn mà còn cải thiện iterative gain.
"""))


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "cells": cells,
}

path = os.path.join(os.path.dirname(__file__), "Test_PA_Residual_Neural_BICMID.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"OK - Generated {os.path.basename(path)}")
