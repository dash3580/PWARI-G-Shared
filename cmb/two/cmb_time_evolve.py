
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# --- Configuration ---
GRID_SIZE = (256, 256)
DX = 1.0
DT = 0.1
STEPS = 500
SIGMA = 4
SNAPSHOT_INTERVAL = 100
SEED = 42

np.random.seed(SEED)

# --- Initialize Fields ---
phi = np.random.randn(*GRID_SIZE) * 0.1
theta = np.random.uniform(-np.pi, np.pi, GRID_SIZE)
phi_dot = np.zeros(GRID_SIZE)
theta_dot = np.zeros(GRID_SIZE)

def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4 * field
    )

# --- Time Evolution ---
os.makedirs("snapshots", exist_ok=True)

for step in range(STEPS):
    phi_ddot = laplacian(phi) - phi**3 + 0.2 * np.sin(theta)
    theta_ddot = laplacian(theta) + 2 * (phi_dot * theta_dot + phi * laplacian(theta)) / np.maximum(phi**2 + 1e-6, 1e-6)

    phi_dot += DT * phi_ddot
    theta_dot += DT * theta_ddot

    # --- Snap Rule Based on High Twist Gradient Energy ---
    grad_x = np.roll(theta, -1, axis=0) - np.roll(theta, 1, axis=0)
    grad_y = np.roll(theta, -1, axis=1) - np.roll(theta, 1, axis=1)
    grad_theta_sq = grad_x**2 + grad_y**2

    SNAP_THRESHOLD = 10.0  # tune this value based on behavior
    snap_zone = grad_theta_sq > SNAP_THRESHOLD

    # Apply snap: wipe twist where threshold is exceeded
    theta[snap_zone] = 0.0
    theta_dot[snap_zone] = 0.0

    # Optional: recoil on breathing field (simulate energy release)
    phi_dot[snap_zone] *= -0.5  # weak recoil


    phi += DT * phi_dot
    theta += DT * theta_dot

    if step % SNAPSHOT_INTERVAL == 0 or step == STEPS - 1:
        cmb_snapshot = gaussian_filter(phi**2, sigma=SIGMA)
        plt.imshow(cmb_snapshot, cmap="inferno")
        plt.title(f"CMB Snapshot (step={step})")
        plt.axis("off")
        plt.savefig(f"snapshots/cmb_snapshot_step{step:04d}.png")
        plt.close()
