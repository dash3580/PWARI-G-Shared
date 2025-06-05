
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# --- Configuration ---
GRID_SIZE = (256, 256)
DX = 1.0
DT = 0.1
STEPS = 500
SIGMA = 5
SNAPSHOT_INTERVAL = 100
SEED = 42

np.random.seed(SEED)

# --- Initialize Fields ---
phi = np.random.randn(*GRID_SIZE) * 0.1
theta = np.random.uniform(-np.pi, np.pi, GRID_SIZE)
phi_dot = np.zeros(GRID_SIZE)
theta_dot = np.zeros(GRID_SIZE)
rho_b = np.zeros(GRID_SIZE)  # new: baryon field

def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4 * field
    )

# --- Time Evolution ---
os.makedirs("snapshots_stage1", exist_ok=True)

for step in range(STEPS):
    # --- Safe breathing and twist terms ---
    phi_ddot = laplacian(phi) - np.clip(phi, -10, 10)**3 + 0.2 * np.sin(theta)
    denom = np.clip(phi**2, 1e-2, 100)
    theta_ddot = laplacian(theta) + 2 * (phi_dot * theta_dot + phi * laplacian(theta)) / denom - 0.1 * theta_dot

    # --- Update twist and breathing ---
    phi_dot += DT * phi_ddot
    theta_dot += DT * theta_ddot

    # --- Snap Rule ---
    grad_x = np.roll(theta, -1, axis=0) - np.roll(theta, 1, axis=0)
    grad_y = np.roll(theta, -1, axis=1) - np.roll(theta, 1, axis=1)
    grad_theta_sq = np.clip(grad_x**2 + grad_y**2, 0, 1e4)

    SNAP_THRESHOLD = 10.0
    snap_zone = grad_theta_sq > SNAP_THRESHOLD

    theta[snap_zone] = 0.0
    theta_dot[snap_zone] = 0.0
    phi_dot[snap_zone] *= -0.5

    # --- Evolve Fields ---
    phi += DT * phi_dot
    theta += DT * theta_dot

    # --- Baryon-Twist Coupling ---
    grad_theta_x = np.roll(theta, -1, axis=0) - np.roll(theta, 1, axis=0)
    grad_theta_y = np.roll(theta, -1, axis=1) - np.roll(theta, 1, axis=1)
    grad_theta_mag = np.sqrt(grad_theta_x**2 + grad_theta_y**2)
    rho_b += 0.05 * phi * grad_theta_mag  # coupling term

    # --- NaN/Inf Check ---
    if np.isnan(phi).any() or np.isinf(phi).any():
        print(f"Numerical instability detected at step {step}")
        break

    # --- Save Snapshot with Silk Damping ---
    if step % SNAPSHOT_INTERVAL == 0 or step == STEPS - 1:
        cmb_snapshot = gaussian_filter(phi**2 + 0.5 * rho_b, sigma=SIGMA)
        plt.imshow(cmb_snapshot, cmap="inferno")
        plt.title(f"CMB Snapshot Stage 1 (step={step})")
        plt.axis("off")
        plt.savefig(f"snapshots_stage1/cmb_snapshot_stage1_step{step:04d}.png")
        plt.close()
