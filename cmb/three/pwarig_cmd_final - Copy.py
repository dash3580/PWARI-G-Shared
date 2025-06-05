
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# --- Configuration ---
GRID_SIZE = (256, 256)
DX = 1.0
DT = 0.1
STEPS = 20000
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

# Optional: baryon feedback field
rho_b = np.zeros(GRID_SIZE)

# Cosmology
a = 1.0
a_list = []
rho_list = []

# --- Field Functions ---
def potential(phi):
    return LAMBDA * 0.25 * (phi**2 - PHI0**2)**2

def dV_dphi(phi):
    return LAMBDA * phi * (phi**2 - PHI0**2)

def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4 * field
    ) / DX**2

# --- Output Directory ---
os.makedirs("pwarig_cmd_output", exist_ok=True)

for step in range(STEPS):
    # --- Compute Field Updates ---
    phi_ddot = laplacian(phi) - dV_dphi(phi) + 0.1 * np.sin(theta) + 0.05 * A
    theta_ddot = laplacian(theta) + 2 * (phi_dot * theta_dot + phi * laplacian(theta)) / np.clip(phi**2, 1e-2, 1e2) - 0.1 * theta_dot
    A_ddot = laplacian(A) - 0.1 * A + 0.05 * np.sin(theta)

    # --- Snap Rule Based on ∇θ² ---
    grad_x = (np.roll(theta, -1, axis=0) - np.roll(theta, 1, axis=0)) / (2 * DX)
    grad_y = (np.roll(theta, -1, axis=1) - np.roll(theta, 1, axis=1)) / (2 * DX)
    grad_theta_sq = grad_x**2 + grad_y**2
    snap_zone = grad_theta_sq > 10.0

    theta[snap_zone] = 0.0
    theta_dot[snap_zone] = 0.0
    phi_dot[snap_zone] *= -0.5

    # --- Evolve Fields ---
    phi_dot += DT * phi_ddot
    phi += DT * phi_dot

    theta_dot += DT * theta_ddot
    theta += DT * theta_dot

    A_dot += DT * A_ddot
    A += DT * A_dot

    # --- Baryon Feedback (optional pressure coupling) ---
    rho_b += 0.05 * phi * np.sqrt(grad_x**2 + grad_y**2)

    # --- Apply Cosmological Damping ---
    phi *= (1 - DAMPING)
    phi_dot *= (1 - DAMPING)

    # --- NaN/Inf Check ---
    if np.isnan(phi).any() or np.isinf(phi).any():
        print(f"Numerical instability at step {step}")
        break

    # --- Energy Density and Scale Factor ---
    grad_phi_sq = (np.gradient(phi, axis=0)**2 + np.gradient(phi, axis=1)**2)
    rho = np.mean(phi_dot**2 + grad_phi_sq + potential(phi))
    rho_list.append(rho)
    a *= 1 + DT * np.sqrt(rho / 3)
    a_list.append(a)

    

    # --- Output Snapshot ---
    if step % 100 == 0 or step == STEPS - 1:
        cmb_map = gaussian_filter(phi**2 + 0.5 * A + 0.5 * rho_b, sigma=SIGMA)
        plt.imshow(cmb_map, cmap="inferno")
        plt.title(f"PWARI-G CMB Snapshot (step={step})")
        plt.axis("off")
        plt.savefig(f"pwarig_cmd_output/cmb_step{step:04d}.png")
        plt.close()

# --- Save Expansion and Density History ---
np.savetxt("pwarig_cmd_output/scale_factor.txt", a_list)
np.savetxt("pwarig_cmd_output/density.txt", rho_list)
