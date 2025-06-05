
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
DARK_RATIO = 0.3

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
ADD_MOTION = True
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
        continue

    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
    r2 = (x_grid - x)**2 + (y_grid - y)**2
    bump = phi0 * np.exp(-r2 / (bump_radius**2))

    phi[x_min:x_max, y_min:y_max] += bump

    vx = np.random.uniform(-0.5, 0.5)
    vy = np.random.uniform(-0.5, 0.5)
    x_grid_local, y_grid_local = np.meshgrid(np.arange(x_min, x_max) - x, np.arange(y_min, y_max) - y, indexing='ij')
    directional_bump = vx * x_grid_local + vy * y_grid_local
    phi_dot[x_min:x_max, y_min:y_max] += directional_bump * bump
    snap_mask[x_min:x_max, y_min:y_max] += bump

# --- Inject dispersive random waves ---
X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, GRID_SIZE), np.linspace(0, 2 * np.pi, GRID_SIZE), indexing='ij')
for _ in range(500):
    kx = np.random.uniform(-3.0, 3.0)
    ky = np.random.uniform(-3.0, 3.0)
    phase = np.random.uniform(0, 2 * np.pi)
    amplitude = np.random.uniform(0.01, 0.05)
    wave = amplitude * np.sin(kx * X + ky * Y + phase)
    phi += wave

# --- Output directory ---
os.makedirs("cmb_snaps", exist_ok=True)

# --- Simulation loop ---
def potential(phi):
    return (lambda_ / 4) * (phi ** 2 - phi0 ** 2) ** 2

def dV_dphi(phi):
    return lambda_ * phi * (phi ** 2 - phi0 ** 2)

for step in range(STEPS):
    laplacian = (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 4 * phi
    ) / DX**2

    damping = np.ones_like(phi)
    edge = 10
    damping[:edge, :] *= np.linspace(0.1, 1, edge)[:, None]
    damping[-edge:, :] *= np.linspace(1, 0.1, edge)[:, None]
    damping[:, :edge] *= np.linspace(0.1, 1, edge)[None, :]
    damping[:, -edge:] *= np.linspace(1, 0.1, edge)[None, :]

    snap_threshold = 1.5 * phi0
    snap_zone = snap_mask > 0.1
    phi = np.where(snap_zone & (phi > snap_threshold), snap_threshold, phi)
    phi_dot = np.where(snap_zone & (phi > snap_threshold), phi_dot * 0.5, phi_dot)

    phi = np.clip(phi, -5.0, 5.0)
    phi_dot = np.clip(phi_dot, -5.0, 5.0)

    border = 2
    phi[:border, :] = 0
    phi[-border:, :] = 0
    phi[:, :border] = 0
    phi[:, -border:] = 0
    phi_dot[:border, :] = 0
    phi_dot[-border:, :] = 0
    phi_dot[:, :border] = 0
    phi_dot[:, -border:] = 0

    phi_ddot = laplacian - dV_dphi(phi) - 3 * (phi_dot / a) - 0.001 * phi_dot
    phi_dot += DT * phi_ddot * damping
    phi += DT * phi_dot * damping

    rho = 0.5 * phi_dot**2 + potential(phi)
    H = np.sqrt((8 * np.pi * G / 3) * np.mean(rho))
    a += a * H * DT

    if step % SNAP_INTERVAL == 0:
        a_values.append(a)
        rho_values.append(np.mean(rho))
        t_values.append(step * DT)

        phi_squared_smoothed = gaussian_filter(phi**2, sigma=3)
        plt.imshow(phi_squared_smoothed, cmap='coolwarm', origin='lower')
        plt.title(f"PWARI-G CMB Snapshot (step={step}, a={a:.3f})")
        plt.colorbar(label='φ² (smoothed)')
        plt.savefig(f"cmb_snaps/cmb_snapshot_step{step:05d}.png")
        plt.close()

# Save expansion and energy plots
plt.figure()
plt.plot(t_values, a_values, label='a(t)')
plt.xlabel('Time')
plt.ylabel('Scale Factor')
plt.title('Expansion Driven by Breathing Energy')
plt.grid()
plt.legend()
plt.savefig('cmb_snaps/scale_factor_vs_time.png')
plt.close()

plt.figure()
plt.plot(t_values, rho_values, label='⟨ρ⟩(t)', color='orange')
plt.xlabel('Time')
plt.ylabel('Mean Energy Density')
plt.title('Decay of Breathing Field Energy')
plt.grid()
plt.legend()
plt.savefig('cmb_snaps/rho_vs_time.png')
plt.close()

cmb_map = gaussian_filter(phi**2, sigma=2)
cmb_map -= np.mean(cmb_map)
cmb_map /= np.max(np.abs(cmb_map))

plt.figure(figsize=(6, 5))
plt.imshow(cmb_map, cmap='coolwarm', origin='lower')
plt.title("Simulated CMB Snapshot (φ² pattern)")
plt.colorbar(label='Normalized Temperature Fluctuation')
plt.savefig("cmb_snaps/cmb_snapshot_final.png")
plt.close()

cmb_fft = np.fft.fftshift(np.fft.fft2(cmb_map))
power_2d = np.abs(cmb_fft)**2

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
