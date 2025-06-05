import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

# --- Parameters ---
GRID_SIZE = 1024
DX = 0.5
DT = 0.001
STEPS = 50000
SNAP_INTERVAL = 100
NUM_SOLITONS = 1000
DARK_RATIO = 0.8

# Cosmology
G = 1.0
phi0 = 1.0
lambda_ = 1.0

# Field arrays
phi = np.zeros((GRID_SIZE, GRID_SIZE))
phi_dot = np.zeros_like(phi)
a = 1.0  # scale factor

# Tracking arrays
a_values = []
rho_values = []
t_values = []

# Soliton parameters
np.random.seed()
coords = np.random.randint(10, GRID_SIZE - 10, size=(NUM_SOLITONS, 2))
dark_flags = np.random.rand(NUM_SOLITONS) < DARK_RATIO
snap_mask = np.zeros_like(phi)

# Localized placement of solitons
bump_radius = 8
for i in range(NUM_SOLITONS):
    x, y = coords[i]
    is_dark = dark_flags[i]
    x_min, x_max = x - bump_radius, x + bump_radius + 1
    y_min, y_max = y - bump_radius, y + bump_radius + 1

    if x_min < 0 or x_max >= GRID_SIZE or y_min < 0 or y_max >= GRID_SIZE:
        continue  # skip out-of-bound placements

    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
    r2 = (x_grid - x)**2 + (y_grid - y)**2
    bump = phi0 * np.exp(-r2 / (bump_radius**2))

    phi[x_min:x_max, y_min:y_max] += bump

    if not is_dark:
        velocity = 0.2 * bump * (np.random.rand(*bump.shape) - 0.5)
        phi_dot[x_min:x_max, y_min:y_max] += velocity
        snap_mask[x_min:x_max, y_min:y_max] += bump

# --- Output directory ---
os.makedirs("cmb_snaps", exist_ok=True)

# --- Simulation loop ---
def potential(phi):
    return (lambda_ / 4) * (phi ** 2 - phi0 ** 2) ** 2

def dV_dphi(phi):
    return lambda_ * phi * (phi ** 2 - phi0 ** 2)

for step in range(STEPS):
    # Laplacian with simple absorbing boundary (zero-gradient)
    laplacian = (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 4 * phi
    ) / DX**2

    # Apply weak boundary damping to avoid reflections
    damping = np.ones_like(phi)
    edge = 10
    damping[:edge, :] *= np.linspace(0.1, 1, edge)[:, None]
    damping[-edge:, :] *= np.linspace(1, 0.1, edge)[:, None]
    damping[:, :edge] *= np.linspace(0.1, 1, edge)[None, :]
    damping[:, -edge:] *= np.linspace(1, 0.1, edge)[None, :]

    # Snap mechanism to prevent uncontrolled growth of visible solitons
    snap_threshold = 1.5 * phi0
    snap_zone = snap_mask > 0.1
    phi = np.where(snap_zone & (phi > snap_threshold), snap_threshold, phi)
    phi_dot = np.where(snap_zone & (phi > snap_threshold), phi_dot * 0.5, phi_dot)

    # Global clamp to limit instability
    phi = np.clip(phi, -5.0, 5.0)
    phi_dot = np.clip(phi_dot, -5.0, 5.0)

    # Suppress outermost edges completely
    border = 2
    phi[:border, :] = 0
    phi[-border:, :] = 0
    phi[:, :border] = 0
    phi[:, -border:] = 0
    phi_dot[:border, :] = 0
    phi_dot[-border:, :] = 0
    phi_dot[:, :border] = 0
    phi_dot[:, -border:] = 0

    # Update field with decay
    phi_ddot = laplacian - dV_dphi(phi) - 3 * (phi_dot / a) - 0.001 * phi_dot
    phi_dot += DT * phi_ddot * damping
    phi += DT * phi_dot * damping

    # Evolve scale factor (Friedmann equation)
    rho = 0.5 * phi_dot**2 + potential(phi)
    H = np.sqrt((8 * np.pi * G / 3) * np.mean(rho))
    a += a * H * DT

    # Track evolution
    if step % SNAP_INTERVAL == 0:
        a_values.append(a)
        rho_values.append(np.mean(rho))
        t_values.append(step * DT)

        plt.imshow(phi, cmap='plasma', origin='lower')
        plt.title(f"Step {step}, a={a:.3f}")
        plt.colorbar(label='φ')
        plt.savefig(f"cmb_snaps/snapshot_{step:05d}.png")
        plt.close()

# Plot scale factor over time
plt.figure()
plt.plot(t_values, a_values, label='a(t)')
plt.xlabel('Time')
plt.ylabel('Scale Factor')
plt.title('Expansion Driven by Breathing Energy')
plt.grid()
plt.legend()
plt.savefig('cmb_snaps/scale_factor_vs_time.png')
plt.close()

# Plot energy density over time
plt.figure()
plt.plot(t_values, rho_values, label='⟨ρ⟩(t)', color='orange')
plt.xlabel('Time')
plt.ylabel('Mean Energy Density')
plt.title('Decay of Breathing Field Energy')
plt.grid()
plt.legend()
plt.savefig('cmb_snaps/rho_vs_time.png')
plt.close()

# Create final CMB snapshot from smoothed φ² field
cmb_map = gaussian_filter(phi**2, sigma=2)
cmb_map -= np.mean(cmb_map)
cmb_map /= np.max(np.abs(cmb_map))  # normalize to [-1, 1]

plt.figure(figsize=(6, 5))
plt.imshow(cmb_map, cmap='coolwarm', origin='lower')
plt.title("Simulated CMB Snapshot (φ² pattern)")
plt.colorbar(label='Normalized Temperature Fluctuation')
plt.savefig("cmb_snaps/cmb_snapshot_final.png")
plt.close()

# Compute and plot angular power spectrum (2D FFT)
cmb_fft = np.fft.fftshift(np.fft.fft2(cmb_map))
power_2d = np.abs(cmb_fft)**2

# Radial binning to get C(l)
r = np.hypot(*np.meshgrid(
    np.arange(-GRID_SIZE//2, GRID_SIZE//2),
    np.arange(-GRID_SIZE//2, GRID_SIZE//2),
    indexing='ij'
))
r = r.astype(np.int32)
rlim = r < (GRID_SIZE // 2)
C_ell = np.bincount(r[rlim].ravel(), weights=power_2d[rlim].ravel())
counts = np.bincount(r[rlim].ravel())
C_ell /= np.maximum(counts, 1)

plt.figure()
plt.plot(np.arange(len(C_ell)), C_ell)
plt.yscale('log')
plt.xlabel('Multipole ℓ')
plt.ylabel('Power C(ℓ)')
plt.title('Angular Power Spectrum from φ²')
plt.grid()
plt.savefig("cmb_snaps/cmb_power_spectrum.png")
plt.close()

print("Simulation complete. Snapshots saved in 'cmb_snaps/' folder.")
