# bell_sim.py - Main PWARI-G Bell violation simulation (2D)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftfreq
from tqdm import tqdm
import csv

# Simulation parameters
L = 20.0         # Domain size
N = 256          # Grid resolution
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

T = 200          # Number of time steps
dt = 0.01        # Time step

# Physics parameters
lambda_nl = 5.0
threshold = 0.5
soft = 1e-3

# Detector angles (in radians)
detector_pairs = [
    (0, np.pi/8),
    (0, 3*np.pi/8),
    (np.pi/4, np.pi/8),
    (np.pi/4, 3*np.pi/8)
]

# Initialize solitons
r1 = np.sqrt((X + 5)**2 + Y**2)
r2 = np.sqrt((X - 5)**2 + Y**2)

phi = np.exp(-r1**2) + np.exp(-r2**2)  # Two initial solitons
theta = np.pi * (np.tanh((X + 5)/2) - np.tanh((X - 5)/2)) / 2

# Fourier space wavevectors
kx = fftfreq(N, d=dx) * 2 * np.pi
ky = fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2

# Linear evolution operator
lin_phase = np.exp(-1j * k2 * dt)

# Record outcomes for each detector setting
outcomes = {ab: [] for ab in detector_pairs}

# Helper: nonlinear twist kick
def nonlinear_kick(phi, theta):
    amp2 = phi**2
    gate = 0.5 * (1 + np.tanh((amp2 - threshold)/soft))
    return np.exp(-1j * lambda_nl * amp2 * gate * dt)

# Helper: detector outcome at angle alpha
def detect(phi, theta, angle):
    integrand = phi**2 * np.cos(theta - angle)
    return np.sign(np.sum(integrand))

# Main simulation loop
for _ in tqdm(range(100)):
    phi_t = phi.copy()
    theta_t = theta.copy()

    for _ in range(T):
        # Split step: half linear in Fourier
        phi_hat = fft2(phi_t)
        theta_hat = fft2(theta_t)
        phi_t = np.real(ifft2(phi_hat * lin_phase))
        theta_t = np.real(ifft2(theta_hat * lin_phase))

        # Nonlinear twist phase kick
        kick = nonlinear_kick(phi_t, theta_t)
        theta_t = np.angle(kick * np.exp(1j * theta_t))

        # Second half linear step
        phi_hat = fft2(phi_t)
        theta_hat = fft2(theta_t)
        phi_t = np.real(ifft2(phi_hat * lin_phase))
        theta_t = np.real(ifft2(theta_hat * lin_phase))

    # Final measurement for each detector setting
    for (a, b) in detector_pairs:
        A = detect(phi_t, theta_t, a)
        B = detect(phi_t, theta_t, b)
        outcomes[(a, b)].append(A * B)

# Compute correlation values
print("\n=== Bell Correlation Results ===")
E = {}
for (a, b), vals in outcomes.items():
    E[(a, b)] = np.mean(vals)
    print(f"E({np.degrees(a):.1f}, {np.degrees(b):.1f}) = {E[(a, b)]:.3f}")

# CHSH combination
S = abs(E[(0, np.pi/8)] - E[(0, 3*np.pi/8)] +
        E[(np.pi/4, np.pi/8)] + E[(np.pi/4, 3*np.pi/8)])
print(f"\nCHSH S = {S:.3f} (Quantum max: ~2.828)")
if S > 2:
    print("Violation detected (Bell inequality broken)")
else:
    print("No violation (classical correlations only)")

# Save final field profiles to CSV
with open("phi_final.csv", "w", newline='') as f_phi, open("theta_final.csv", "w", newline='') as f_theta:
    writer_phi = csv.writer(f_phi)
    writer_theta = csv.writer(f_theta)
    for i in range(N):
        writer_phi.writerow(phi_t[i])
        writer_theta.writerow(theta_t[i])

# Plot final field profiles
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(phi_t, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='viridis')
axs[0].set_title("Final |phi(x, y)|")
axs[1].imshow(theta_t, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='twilight')
axs[1].set_title("Final twist theta(x, y)")
plt.tight_layout()
plt.savefig("field_profiles.png")
plt.show()