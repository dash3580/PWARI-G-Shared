import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
R = np.zeros((nt, nr))  # Ricci scalar estimate

# PWARI-G metric fields
alpha = np.ones((nt, nr))
a = np.ones((nt, nr))  # kept constant for now

# To track soliton core location over time
core_location = np.zeros(nt)

# Potential and its derivative
def V(phi):
    return 0.5 * phi**2

def dV_dphi(phi):
    return phi

# Schwarzschild comparison function (safe version)
def schwarz_alpha(r, M):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(np.where(r > 2 * M, 1 - 2 * M / r, np.nan))

# Initial condition: Gaussian pulse
def initialize():
    phi0 = 1.0 * np.exp(-((r - 5.0)**2) / (0.5**2))
    phi[0, :] = phi0
    Pi[0, :] = 0.0
    return

# Boundary conditions
def apply_boundary_conditions(n):
    Phi[n, 0] = 0.0
    Phi[n, -1] = Phi[n, -2]
    Pi[n, 0] = Pi[n, 1]
    Pi[n, -1] = Pi[n, -2]
    alpha[n, 0] = alpha[n, 1]
    alpha[n, -1] = alpha[n, -2]

# PWARI-G redshift update from energy density
def update_alpha(n):
    for i in range(1, nr - 1):
        rho = 0.5 * Pi[n, i]**2 + 0.5 * Phi[n, i]**2 + V(phi[n, i])
        alpha[n + 1, i] = alpha[n, i] + dt * (-(alpha[n, i] - np.exp(-rho)) / tau)
        R[n, i] = Pi[n, i]**2 + Phi[n, i]**2 + 2 * V(phi[n, i])  # Ricci scalar estimate

# Time evolution using leapfrog scheme with PWARI-G redshift
def evolve():
    phi[1, :] = phi[0, :] + dt * alpha[0, :] * Pi[0, :]  # First step via Euler

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
            print(f"Step {n}/{nt}", "phi max:", np.max(phi[n, :]), "Pi max:", np.max(Pi[n, :]), "alpha min:", np.min(alpha[n, :]), "core @ r=", core_location[n])

# Run simulation
def main():
    os.makedirs("output", exist_ok=True)
    initialize()
    evolve()
    np.savez("output/collapse_fields.npz", phi=phi, Pi=Pi, Phi=Phi, R=R, r=r, alpha=alpha, a=a, core_location=core_location)

    # Final-time radial slices and Schwarzschild comparison
    fit_region = (r > 15)
    r_tail = r[fit_region]
    alpha_tail = alpha[nt - 2, :][fit_region]
    valid = r_tail > 2.0
    r_fit = r_tail[valid]
    alpha_fit = alpha_tail[valid]

    # Debug prints to check fitting region
    print('Final redshift profile (alpha[nt - 2, :]):', alpha[nt - 2, :])
    print('Fitting region r:', r[fit_region])
    print('Fitting values alpha_fit:', alpha[nt - 2, :][fit_region])

    try:
        popt, _ = curve_fit(schwarz_alpha, r_fit, alpha_fit, p0=[1.0])
        M_eff = popt[0]
        schwarz_fit = schwarz_alpha(r, M_eff)
    except:
        M_eff = np.nan
        schwarz_fit = np.full_like(r, np.nan)

    # Plot redshift comparison
    plt.figure()
    plt.plot(r, alpha[nt - 2, :], label='PWARI-G α(r)', color='orange')
    plt.plot(r, schwarz_fit, label=f'Schwarzschild α(r), M={M_eff:.2f}', linestyle='--', color='black')
    plt.xlabel("Radial coordinate r")
    plt.ylabel("Redshift α(r)")
    plt.title("PWARI-G vs Schwarzschild Redshift")
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig("output/alpha_vs_schwarzschild.png")
    plt.close()

    with open("output/effective_mass.txt", "w") as f:
        f.write(f"Estimated effective mass from redshift tail: M_eff = {M_eff:.6f}\n")

if __name__ == "__main__":
    main()

