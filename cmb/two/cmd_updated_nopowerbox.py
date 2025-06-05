
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import os

# --- Configuration ---
GRID_SIZE = (256, 256)
DX = 1.0  # degrees per pixel
SIGMA = 6  # Gaussian filter strength for smoothing
SEED = 42

np.random.seed(SEED)

# --- Generate Custom Scale-Invariant Noise ---
def generate_scale_invariant_noise(shape, spectral_index=1.0):
    kx = np.fft.fftfreq(shape[0])[:, None]
    ky = np.fft.fftfreq(shape[1])[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-6  # avoid division by zero

    amplitude = 1.0 / k**(spectral_index / 2.0)
    random_phase = np.exp(2j * np.pi * np.random.rand(*shape))
    fourier_field = amplitude * random_phase

    field = np.fft.ifft2(fourier_field).real
    field -= field.mean()
    field /= field.std()
    return field

phi = generate_scale_invariant_noise(GRID_SIZE)

# --- Simulate Twist Field θ and Light Field A ---
theta = np.random.uniform(-np.pi, np.pi, GRID_SIZE)
A = np.zeros(GRID_SIZE)

def update_photons(A, theta):
    return A + 0.1 * np.sin(theta)

A = update_photons(A, theta)

# --- φ² Map and Smoothing ---
phi_squared = phi**2
cmb_map = gaussian_filter(phi_squared, sigma=SIGMA)

# --- Periodic Laplacian ---
def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4 * field
    )

laplacian_phi = laplacian(phi)

# --- Plot CMB Map ---
plt.figure(figsize=(8, 8))
plt.imshow(cmb_map, cmap='inferno')
plt.title("Simulated PWARI-G CMB Snapshot (Smoothed φ²)")
plt.axis('off')
plt.savefig("cmb_snapshot_updated.png")
plt.show()

# --- Power Spectrum ---
phi_ft = np.fft.fft2(cmb_map)
phi_ps = np.abs(phi_ft)**2
phi_ps_shifted = np.fft.fftshift(phi_ps)

N = GRID_SIZE[0]
freqs = np.fft.fftfreq(N, d=DX / 60.0)
ell = 2 * np.pi * np.sqrt(np.add.outer(freqs**2, freqs**2))
ell_shifted = np.fft.fftshift(ell)

ell_bins = np.arange(0, np.max(ell_shifted), 25)
cl = np.zeros(len(ell_bins) - 1)

for i in range(len(ell_bins) - 1):
    mask = (ell_shifted >= ell_bins[i]) & (ell_shifted < ell_bins[i+1])
    cl[i] = phi_ps_shifted[mask].mean() if np.any(mask) else 0

plt.figure()
plt.plot(0.5 * (ell_bins[:-1] + ell_bins[1:]), cl)
plt.xlabel("Multipole moment ℓ")
plt.ylabel("C(ℓ)")
plt.title("PWARI-G CMB Power Spectrum")
plt.grid(True)
plt.savefig("cmb_power_spectrum.png")
plt.show()

# --- 2-Point Correlation ---
corr = correlate2d(cmb_map, cmb_map, mode='same')

plt.figure(figsize=(6, 6))
plt.imshow(corr, cmap='viridis')
plt.title("2-Point Angular Correlation")
plt.axis('off')
plt.savefig("cmb_2pt_correlation.png")
plt.show()
