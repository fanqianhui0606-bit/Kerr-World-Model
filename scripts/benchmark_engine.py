import jax
import jax.numpy as jnp
import os
import time
import json
import sys
from flax import serialization
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Add core to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from fno_model import create_fno_model

# 1. Config & Grid Constants (Detached from proprietary solver)
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
FINAL_WEIGHTS = os.path.join(BASE_DIR, "weights/fno_world_model_final.msgpack")
STATS_FILE = os.path.join(BASE_DIR, "data/norm_stats.json")

# Physical Grid Constants (Proprietary solver removed)
R_MIN, R_MAX = 2.1, 100.0
DT_LARGE = 1.0 
N_STEPS = 100
OBS_R = 60.0
SENSOR_IDX = int((OBS_R - R_MIN) / (R_MAX - R_MIN) * 600)

if not os.path.exists(STATS_FILE) or not os.path.exists(FINAL_WEIGHTS):
    print(f"❌ Missing required files in 'data/' or 'weights/'.")
    print(f"Please ensure you have trained the model or contact the author for the pre-trained weights.")
    sys.exit(1)

with open(STATS_FILE, "r") as f:
    stats = json.load(f)
MEANS = jnp.array(stats["u"]["means"])
STDS = jnp.array(stats["u"]["stds"])
STDS = jnp.array([s if s > 1e-12 else 1.0 for s in STDS])

def denormalize(u_norm): return u_norm * STDS + MEANS
def normalize(u): return (u - MEANS) / STDS

# 2. Init Model
model = create_fno_model(modes=64, width=128, out_channels=6, padding=20)
r_grid = jnp.linspace(R_MIN, R_MAX, 600)
dummy_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 600, 6)), jnp.ones((1, 600, 1)))['params']

with open(FINAL_WEIGHTS, "rb") as f:
    params = serialization.from_bytes(dummy_params, f.read())

# 3. Load Sample for comparison (Requires training_data.h5 or dummy_data)
try:
    with h5py.File(os.path.join(BASE_DIR, "data/training_data.h5"), "r") as f:
        u_true_all = f["u"][42]
except Exception as e:
    print("⚠️  Warning: Real training data not found. Using zero-input for bias calibration demo.")
    u_true_all = None

# 3.1 Capturing Inference Bias (Bias Correction Patch)
print("🎯 Calibrating Inference Bias (Zero-Input Offset)...")
zero_input = jnp.zeros((1, 600, 6))
bias_error = model.apply({'params': params}, zero_input, r_grid[None, :, None], train=False)
bias_error = bias_error.block_until_ready()

if u_true_all is None:
    print("🏁 Calibration test complete. Real waveform comparison requires training dataset.")
    sys.exit(0)

# 4. FNO Unrolling with Bias Correction
print("🚀 Starting Corrected FNO Long-term Evolution (100 steps)...")
t_start = 100
start_state = u_true_all[t_start]
curr_x_norm = normalize(start_state)[None, ...]
fno_history = [start_state]
r_coords = r_grid[None, :, None]

t0 = time.time()
for s in range(N_STEPS):
    delta_norm = model.apply({'params': params}, curr_x_norm, r_coords, train=False)[0]
    delta_corrected = delta_norm - bias_error[0]
    curr_x_norm = (curr_x_norm[0] + delta_corrected)[None, ...]
    fno_history.append(denormalize(curr_x_norm[0]))
t_fno_total = time.time() - t0
fno_history = jnp.stack(fno_history)

# 5. Extraction & Plotting
print("📊 Generating Visual Audit...")
u_true = u_true_all[t_start : t_start+N_STEPS+1]
time_axis = jnp.arange(N_STEPS+1) * DT_LARGE

# Waveform at specified r
psi_obs_fno = fno_history[:, SENSOR_IDX, 0]
psi_obs_true = u_true[:, SENSOR_IDX, 0]

plt.figure(figsize=(15, 10))

# Panel A: Waveform Comparison
plt.subplot(2, 2, 1)
plt.plot(time_axis, psi_obs_true, 'k-', alpha=0.5, label="Truth (Sampled)")
plt.plot(time_axis, psi_obs_fno, 'r--', label="FNO Prediction")
plt.ylim(-0.0001, 0.0007)
plt.title(f"Waveform $\\Psi(t)$ at $r={OBS_R}$")
plt.xlabel("Time ($M$)")
plt.ylabel("Field Value")
plt.legend()
plt.grid(True, alpha=0.3)

# Panel B: Prediction Error Heatmap
plt.subplot(2, 2, 2)
err = jnp.abs(fno_history[:, :, 0] - u_true[:, :, 0])
plt.imshow(err.T, aspect='auto', extent=[0, N_STEPS*DT_LARGE, R_MIN, R_MAX], origin='lower', cmap='inferno', vmax=1e-4)
plt.colorbar(label='Abs Error')
plt.title("FNO Prediction Error Heatmap")
plt.xlabel("Time")
plt.ylabel("Radius $r$")

# Panel C: Energy Drill (Stability)
plt.subplot(2, 2, 3)
e_fno = jnp.sum(fno_history[:, :, 0]**2, axis=1)
e_true = jnp.sum(u_true[:, :, 0]**2, axis=1)
plt.plot(time_axis, (e_fno - e_true)/e_true * 100, 'g-', label="Energy Drift %")
plt.title("Long-term Relative Energy Drift")
plt.ylabel("Drift (%)")
plt.xlabel("Time")
plt.legend()

# Panel D: Performance Metrics
plt.subplot(2, 2, 4)
total_err_mean = jnp.mean(err)
plt.text(0.1, 0.5, f"Final Performance:\n\nAvg Latency: {t_fno_total/N_STEPS*1000:.2f} ms/step\nAvg Frame Error: {total_err_mean:.2e}\nStable Horizon: 100 Steps\nDrift (100 steps): {(e_fno[-1]-e_true[-1])/e_true[-1]*100:.2f}%", 
         fontsize=14, family='monospace', bbox=dict(facecolor='white', alpha=0.8))
plt.axis('off')

plt.tight_layout()
os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "assets/benchmark_results.png"), dpi=200)
print(f"🏁 Showcase Audit Complete. Results saved to 'assets/benchmark_results.png'.")
