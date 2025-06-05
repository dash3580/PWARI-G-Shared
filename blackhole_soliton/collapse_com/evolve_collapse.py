import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

# Constants and grid
r_max = 20.0
nr = 1000
nt = 5000
dr = r_max / nr
dt = 0.005

tau = 1.0  # PWARI-G redshift relaxation timescale

r = np.linspace(0, r_max, nr)
eps = 1e-6

# Field variables
phi = np.zeros((nt, nr))
Pi = np.zeros((nt, nr))
Phi = np.zeros((nt, nr))
R = np.zeros((nt, nr))

# PWARI-G metric fields
alpha = np.ones((nt, nr))
a = np.ones((nt, nr))

# Soliton core tracking
core_location = np.zeros(nt)

# Conversion constant
mass_unit_km = 1.4766  # 1 M_sun = 1.4766 km (geometric units)

# Mapping amplitude to galaxy name
amplitude_to_object = {
    0.5: "Mrk335",
    1.0: "Mrk1501",
    1.5: "PG0026+129",
    2.0: "PG0052+251",
    2.5: "Fairall9"
}

# Observed log10(M/M_sun) values for reference matching
object_to_log_mass_obs = {
    "Mrk335": 7.230,
    "Mrk1501": 8.067,
    "PG0026+129": 8.487,
    "PG0052+251": 8.462,
    "Fairall9": 8.299
}

# Potential and its derivative
def V(phi): return 0.5 * phi**2
def dV_dphi(phi): return phi

def compute_mass_integral(n=-2):
    rho = 0.5 * Pi[n, :]**2 + 0.5 * Phi[n, :]**2 + V(phi[n, :])
    integrand = 4 * np.pi * r**2 * rho
    return np.trapz(integrand, r)

def initialize(amplitude=1.0):
    phi[...] = 0.0
    Pi[...] = 0.0
    Phi[...] = 0.0
    alpha[...] = 1.0
    R[...] = 0.0
    core_location[...] = 0.0
    phi0 = amplitude * np.exp(-((r - 5.0)**2) / (0.5**2))
    phi[0, :] = phi0

def apply_boundary_conditions(n):
    Phi[n, 0] = 0.0
    Phi[n, -1] = Phi[n, -2]
    Pi[n, 0] = Pi[n, 1]
    Pi[n, -1] = Pi[n, -2]
    alpha[n, 0] = alpha[n, 1]
    alpha[n, -1] = alpha[n, -2]

def update_alpha(n):
    for i in range(1, nr - 1):
        rho = 0.5 * Pi[n, i]**2 + 0.5 * Phi[n, i]**2 + V(phi[n, i])
        alpha[n + 1, i] = alpha[n, i] + dt * (-(alpha[n, i] - np.exp(-rho)) / tau)
        R[n, i] = Pi[n, i]**2 + Phi[n, i]**2 + 2 * V(phi[n, i])

def evolve():
    phi[1, :] = phi[0, :] + dt * alpha[0, :] * Pi[0, :]
    for n in range(1, nt - 1):
        apply_boundary_conditions(n)
        for i in range(1, nr - 1):
            dPhi_dr = (phi[n, i + 1] - phi[n, i - 1]) / (2 * dr)
            Phi[n, i] = dPhi_dr / a[n, i]
            term1 = r[i]**2 * alpha[n, i] * Phi[n, i]
            term0 = r[i - 1]**2 * alpha[n, i - 1] * Phi[n, i - 1]
            denom = dr * max(r[i]**2 + eps, eps)
            dPi_dt = (term1 - term0) / denom
            dPi_dt -= alpha[n, i] * a[n, i] * dV_dphi(phi[n, i])
            Pi[n + 1, i] = Pi[n - 1, i] + 2 * dt * dPi_dt
            phi[n + 1, i] = phi[n - 1, i] + 2 * dt * alpha[n, i] * Pi[n, i]
        update_alpha(n)
        core_location[n] = r[np.argmax(phi[n, :])]
        if n % 500 == 0:
            print(f"Step {n}/{nt} phi max: {np.max(phi[n, :])} Pi max: {np.max(Pi[n, :])} alpha min: {np.min(alpha[n, :])} core @ r= {core_location[n]}")

def run_sweep():
    amplitudes = [0.5, 1.0, 1.5, 2.0, 2.5]
    raw_masses = []
    log_obs_list = []
    labels = []

    # First pass: gather raw results
    for A in amplitudes:
        print(f"\n=== Running simulation for amplitude = {A} ===")
        initialize(amplitude=A)
        evolve()
        M_eff = compute_mass_integral(n=nt - 2)
        object_name = amplitude_to_object.get(A, f"A={A}")
        raw_masses.append(M_eff)
        log_obs = object_to_log_mass_obs.get(object_name, None)
        log_obs_list.append(log_obs)
        labels.append(object_name)

    # Fit scaling factor
    def scaling_model(m, scale):
        return np.log10(scale * np.array(m))

    mask = [m is not None for m in log_obs_list]
    m_fit = np.array(raw_masses)[mask]
    log_fit = np.array(log_obs_list)[mask]
    popt, _ = curve_fit(scaling_model, m_fit, log_fit, p0=[1.0])
    scale_factor = popt[0]
    print(f"Best-fit scale factor: {scale_factor:.3e}")

    # Save scaled results
    with open("output/pwari_mass_sweep.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Object", "PWARI_G_Meff"])
        for obj, raw in zip(labels, raw_masses):
            scaled = raw * scale_factor
            writer.writerow([obj, scaled])
            print(f"{obj}: M_eff_scaled = {scaled:.3e}")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    run_sweep()