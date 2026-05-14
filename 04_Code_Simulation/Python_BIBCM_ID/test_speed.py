import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np

from python_code.config import ConvCodeConfig
from python_code.encoders import RSCEncoder
from python_code.decoders import SISODecoder

cfg = ConvCodeConfig()
enc = RSCEncoder(cfg.g_matrix)
siso = SISODecoder(cfg.g_matrix, code_type=0, dec_type=0)

num_bits = 4096
u = np.random.randint(0, 2, num_bits)
v = enc.encode(u)

input_u = np.zeros(num_bits)
input_c = np.random.randn(len(v) + enc.m * enc.n)

print("Starting decode (first run will compile)...")
t0 = time.time()
output_u, output_c = siso.decode(input_u, input_c)
print(f"First run: {time.time() - t0:.4f} s")

print("Starting decode (second run)...")
t0 = time.time()
for _ in range(10):
    output_u, output_c = siso.decode(input_u, input_c)
print(f"10 runs: {time.time() - t0:.4f} s")
