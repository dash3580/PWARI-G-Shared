
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
from powerbox import get_powerlaw
import os

# --- Configuration ---
GRID_SIZE = (256, 256)
DX = 1.0  # degrees per pixel, adjust as needed
SIGMA = 6  # Gaussian filter strength for smoothing
SEED = 42

np.random.seed(SEED)

# --- Initialize Scalar Field φ with Scale-Invariant Noise ---
phi = get_powerlaw(GRID_SIZE, dim=2, spectral_index=1)  # ns ~ 1

# --- Optional: Simulate Twist Field θ and Light Field A (Photon Coupling) ---
theta = np.random.uniform(-np.pi, np.pi, GRID_SIZE)
A = np.zeros(GRID_SIZE)

def update_photons(A, theta):
    return A + 0.1 * np.sin(theta)

A = update_photons(A, theta)

# --- Calculate φ² Map and Apply Smoothing ---
phi_squared = phi**2
cmb_map = gaussian_filter(phi_squared, sigma=SIGMA)

# --- Apply Periodic Laplacian for φ ---
def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4 * field
    )

laplacian_phi = laplacian(phi)

# --- Plot the Simulated CMB Map ---
plt.figure(figsize=(8, 8))
plt.imshow(cmb_map, cmap='inferno')
plt.title("Simulated PWARI-G CMB Snapshot (Smoothed φ²)")
plt.axis('off')
plt.savefig("cmb_snapshot_updated.png")
plt.show()

# --- Compute and Plot Angular Power Spectrum C(ℓ) ---
phi_ft = np.fft.fft2(cmb_map)
phi_ps = np.abs(phi_ft)**2
phi_ps_shifted = np.fft.fftshift(phi_ps)

# Convert pixel distances to multipole moments ℓ
N = GRID_SIZE[0]
freqs = np.fft.fftfreq(N, d=DX / 60.0)  # degrees -> arcmin
ell = 2 * np.pi * np.sqrt(np.add.outer(freqs**2, freqs**2))
ell_shifted = np.fft.fftshift(ell)

# Bin ℓ into bands
ell_bins = np.arange(0, np.max(ell_shifted), 25)
cl = np.zeros(len(ell_bins) - 1)

for i in range(len(ell_bins) - 1):
    mask = (ell_shifted >= ell_bins[i]) & (ell_shifted < ell_bins[i+1])
    cl[i] = phi_ps_shifted[mask].mean() if np.any(mask) else 0

# Plot the power spectrum
plt.figure()
plt.plot(0.5 * (ell_bins[:-1] + ell_bins[1:]), cl)
plt.xlabel("Multipole moment ℓ")
plt.ylabel("C(ℓ)")
plt.title("PWARI-G CMB Power Spectrum")
plt.grid(True)
plt.savefig("cmb_power_spectrum.png")
plt.show()

# --- Compute and Plot 2-Point Correlation Function ---
corr = correlate2d(cmb_map, cmb_map, mode='same')

plt.figure(figsize=(6, 6))
plt.imshow(corr, cmap='viridis')
plt.title("2-Point Angular Correlation")
plt.axis('off')
plt.savefig("cmb_2pt_correlation.png")
plt.show()
