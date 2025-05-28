# hydrogen_model.py
# Clean PWARI-G hydrogen atom simulator with breathing soliton, twist, gauge, and gravity
# Fully causal, math-driven evolution. No hard clips. Redshift trails energy. Shells form naturally.

import numpy as np
import matplotlib.pyplot as plt
import os

# === Simulation Constants ===
GRID_SIZE = (64, 64, 64)
DX = 0.2
DT = 0.01
STEPS = 1000

# Field Parameters
LAMBDA_PHI = 1.0     # Scalar self-interaction strength
ALPHA_TWIST = 0.2     # Twist regularization
G = 1.0               # Gravitational strength
DAMPING = 0.01        # Gentle physical damping to prevent runaway growth

# Output folder for diagnostics
os.makedirs("output", exist_ok=True)

# === Field Initialization ===
def initialize_fields():
    kx, ky = 0.1, 0.1
    shape = GRID_SIZE
    grid = np.indices(shape).astype(np.float32)
    center = np.array(shape)[:, None, None, None] / 2.0
    r2 = np.sum((grid - center)**2, axis=0) * DX**2
    phi = np.exp(-r2 * 5.0)
    phi_dot = np.zeros_like(phi)
    x = np.linspace(-shape[0]//2, shape[0]//2, shape[0]) * DX
    y = np.linspace(-shape[1]//2, shape[1]//2, shape[1]) * DX
    X, Y = np.meshgrid(x, y, indexing='ij')
    theta = np.zeros_like(phi)
    for z in range(shape[2]):
        theta[:, :, z] = kx * X + ky * Y
    theta_dot = np.zeros_like(phi)
    A0 = np.zeros_like(phi)
    A = np.ones_like(phi)
    return phi, phi_dot, theta, theta_dot, A0, A  # includes initial twist

# === Evolution Functions ===
def laplacian(f):
    return sum(np.gradient(np.gradient(f, DX, axis=i), DX, axis=i) for i in range(3))

def evolve_phi(phi, phi_dot, theta, theta_dot, A0, A):
    grad_theta = np.gradient(theta, DX)
    grad_theta_sq = sum(g**2 / (1 + ALPHA_TWIST * g**2) for g in grad_theta)
    phase_error = (theta_dot - A0)**2

    phi_ddot = A * (
        laplacian(phi)
        - LAMBDA_PHI * phi**3
        - phi * phase_error
        + phi * grad_theta_sq
    )
    phi_dot += DT * phi_ddot
    phi_dot *= (1 - DAMPING)
    phi += DT * phi_dot
    return phi, phi_dot

def evolve_theta(phi, phi_dot, theta, theta_dot, A0):
    grad_theta = np.gradient(theta, DX)
    grad_phi = np.gradient(phi, DX)
    laplacian_theta = sum(np.gradient(g, DX, axis=i) for i, g in enumerate(grad_theta))
    dot_grad = sum(gp * gt for gp, gt in zip(grad_phi, grad_theta))
    mismatch = A0 - theta_dot

    theta_ddot = (
        - phi**2 * laplacian_theta
        - 2 * phi * phi_dot * mismatch
        - 2 * phi * dot_grad
    ) / (phi**2 + 1e-12)
    theta_dot += DT * theta_ddot
    theta_dot *= (1 - DAMPING)
    theta += DT * theta_dot
    return theta, theta_dot

def evolve_gravity(A, phi, phi_dot, theta, theta_dot, A0):
    grad_phi = np.gradient(phi, DX)
    grad_theta = np.gradient(theta, DX)
    energy_density = (
        0.5 * phi_dot**2 +
        0.25 * phi**4 +
        0.5 * phi**2 * sum(g**2 for g in grad_theta) +
        0.5 * phi**2 * (theta_dot - A0)**2
    )
    curvature_source = -G * energy_density
    A_new = A + DT * curvature_source
    return A_new

# === Diagnostics ===
def estimate_winding(theta):
    center = GRID_SIZE[0] // 2
    dtheta = np.gradient(theta[center, center, :], DX)
    return np.sum(dtheta) * DX / (2 * np.pi)

def plot_phi_slice(phi, step):
    center = GRID_SIZE[2] // 2
    plt.imshow(phi[:, :, center], cmap='inferno', origin='lower')
    plt.colorbar(label='phi')
    plt.title(f"phi slice at step {step}")
    plt.savefig(f"output/phi_step_{step:04d}.png")
    plt.close()

def plot_twist_energy(phi, theta, theta_dot, A0, step):
    center = GRID_SIZE[2] // 2
    grad_theta = np.gradient(theta, DX)
    grad_theta_sq = sum(g**2 for g in grad_theta)
    twist_energy = phi**2 * (grad_theta_sq + (theta_dot - A0)**2)
    plt.imshow(twist_energy[:, :, center], cmap='magma', origin='lower')
    plt.colorbar(label='twist energy')
    plt.title(f"twist energy at step {step}")
    plt.savefig(f"output/twist_energy_{step:04d}.png")
    plt.close()

# === Main Simulation Loop ===
def evolve_A0(A0, phi, theta_dot):
    source = phi**2 * theta_dot
    A0_ddot = laplacian(A0) + source
    A0_dot = DT * A0_ddot
    return A0 + A0_dot

def run_simulation():
    phi, phi_dot, theta, theta_dot, A0, A = initialize_fields()
    for step in range(STEPS):
        phi, phi_dot = evolve_phi(phi, phi_dot, theta, theta_dot, A0, A)
        theta, theta_dot = evolve_theta(phi, phi_dot, theta, theta_dot, A0)
        A = evolve_gravity(A, phi, phi_dot, theta, theta_dot, A0)
        A0 = evolve_A0(A0, phi, theta_dot)

        if step % 5 == 0:
            winding = estimate_winding(theta)
            print(f"Step {step}: max(phi) = {np.max(phi):.4f}, winding ≈ {winding:.2f}")
            plot_phi_slice(phi, step)
            plot_twist_energy(phi, theta, theta_dot, A0, step)

if __name__ == "__main__":
    run_simulation()
