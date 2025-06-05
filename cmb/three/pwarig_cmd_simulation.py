
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# --- Configuration ---
GRID_SIZE = (256, 256)
DX = 1.0
DT = 0.1
STEPS = 500
PHI0 = 1.0
LAMBDA = 1.0
DECAY_RATE = 0.0002
DAMPING = 0.01
SIGMA = 5
SEED = 42

np.random.seed(SEED)

# --- Initialize Fields ---
phi = np.random.randn(*GRID_SIZE) * 0.01
phi_dot = np.zeros(GRID_SIZE)
theta = np.random.uniform(-np.pi, np.pi, GRID_SIZE)
theta_dot = np.zeros(GRID_SIZE)
A = np.zeros(GRID_SIZE)  # Photon-like field
A_dot = np.zeros(GRID_SIZE)

# Cosmology
a = 1.0
a_list = []
rho_list = []

# --- Potential ---
def potential(phi):
    return LAMBDA * 0.25 * (phi**2 - PHI0**2)**2

def dV_dphi(phi):
    return LAMBDA * phi * (phi**2 - PHI0**2)

def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4 * field
    )

os.makedirs("pwarig_cmd_output", exist_ok=True)

for step in range(STEPS):
    # φ evolution (with potential and twist/A coupling)
    phi_ddot = laplacian(phi) - dV_dphi(phi) + 0.1 * np.sin(theta) + 0.05 * A
    theta_ddot = laplacian(theta) + 2 * (phi_dot * theta_dot + phi * laplacian(theta)) / np.clip(phi**2, 1e-2, 1e2) - 0.1 * theta_dot
    A_ddot = laplacian(A) - 0.1 * A + 0.05 * np.sin(theta)  # light-like field with twist driving

    # Snap rule based on gradient of θ
    grad_x = np.roll(theta, -1, axis=0) - np.roll(theta, 1, axis=0)
    grad_y = np.roll(theta, -1, axis=1) - np.roll(theta, 1, axis=1)
    grad_theta_sq = np.clip(grad_x**2 + grad_y**2, 0, 1e4)
    snap_zone = grad_theta_sq > 10.0

    theta[snap_zone] = 0.0
    theta_dot[snap_zone] = 0.0
    phi_dot[snap_zone] *= -0.5

    # Update fields
    phi_dot += DT * phi_ddot
    phi += DT * phi_dot

    theta_dot += DT * theta_ddot
    theta += DT * theta_dot

    A_dot += DT * A_ddot
    A += DT * A_dot

    # Apply damping (optional cosmological decay)
    phi *= (1 - DAMPING)
    phi_dot *= (1 - DAMPING)

    # Update scale factor and energy density
    rho = np.mean(phi_dot**2 + (np.gradient(phi, axis=0)**2 + np.gradient(phi, axis=1)**2) + potential(phi))
    rho_list.append(rho)
    a *= 1 + DT * np.sqrt(rho / 3)
    a_list.append(a)

    # Output
    if step % 100 == 0 or step == STEPS - 1:
        cmb_map = gaussian_filter(phi**2 + 0.5 * A, sigma=SIGMA)
        plt.imshow(cmb_map, cmap="inferno")
        plt.title(f"PWARI-G CMB Snapshot (step={step})")
        plt.axis("off")
        plt.savefig(f"pwarig_cmd_output/cmb_step{step:04d}.png")
        plt.close()
