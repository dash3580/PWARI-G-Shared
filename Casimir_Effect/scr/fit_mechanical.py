# fit_mechanical.py
# Fits a power law A/d^n to mechanical frequency shift data from 3-line text file

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load Data ---
print("Loading mechanical frequency shift data...")
lines = np.loadtxt("../data/Optimized_mech_freqs.txt")
d = lines[0]  # Distance in nm
f = lines[1]  # Frequency shift in Hz
err = lines[2]  # Measurement uncertainty in Hz (optional)

# Filter valid values
d_mask = (d > 0) & np.isfinite(d) & np.isfinite(f)
d = d[d_mask]
f = f[d_mask]
err = err[d_mask]

# --- Define model ---
def power_law(x, A, n):
    return A / x**n

# --- Fit model ---
popt, pcov = curve_fit(power_law, d, f, sigma=err, absolute_sigma=True)
A_fit, n_fit = popt

# --- Print result ---
print(f"Best-fit exponent: n = {n_fit:.4f}")
print(f"Best-fit amplitude: A = {A_fit:.4e}")

# --- Generate fit curve ---
d_fit = np.linspace(min(d), max(d), 500)
f_fit = power_law(d_fit, A_fit, n_fit)

# --- Plot ---
plt.figure(figsize=(6, 4))
plt.errorbar(d, f, yerr=err, fmt='o', label="Experimental Data", color="green", markersize=5)
plt.plot(d_fit, f_fit, label=f"Fit: A/d^{{{n_fit:.3f}}}", color="orange")
plt.xlabel("Distance (nm)")
plt.ylabel("Frequency Shift (Hz)")
plt.title("PWARI-G Fit to Mechanical Frequency Shift")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Save plot ---
plt.savefig("../plots/mechanical_shift_fit.png")
print("Saved plot to ../plots/mechanical_shift_fit.png")