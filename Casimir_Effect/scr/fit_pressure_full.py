# fit_pressure_full.py
# Enhanced pressure fitting script for PWARI-G Casimir analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load Data ---
print("Loading data from Casimir_pressure_dist_v4.h5")
df = pd.read_hdf("../data/Casimir_pressure_dist_v4.h5", key="data")

# Extract correct columns
x = df["Gap (nm)"].values
y = np.abs(df["P BCS (Pa)"].values)  # magnitude (pressure is negative)

# --- Filter ---
mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

# --- Fit Model ---
def power_law(d, A, n):
    return A / d**n

popt, pcov = curve_fit(power_law, x, y)
A_fit, n_fit = popt
perr = np.sqrt(np.diag(pcov))

# --- Residuals and Chi2 ---
y_fit = power_law(x, *popt)
residuals = y - y_fit
chi2 = np.sum((residuals)**2 / y_fit)

# --- Print Results ---
print(f"Best-fit exponent: n = {n_fit:.4f} ± {perr[1]:.4f}")
print(f"Best-fit amplitude: A = {A_fit:.4e} ± {perr[0]:.2e}")
print(f"Chi-squared: χ² = {chi2:.3e}")

# --- Plot Fit ---
plt.figure(figsize=(6.5, 4.5))
plt.scatter(x, y, label="|BCS Data|", color="blue", s=20)
x_fit = np.linspace(min(x), max(x), 500)
plt.plot(x_fit, power_law(x_fit, *popt), label=f"Fit: A/d$^{{{n_fit:.3f}}}$", color="red")
plt.xlabel("Gap (nm)")
plt.ylabel("Pressure Magnitude (Pa)")
plt.title("PWARI-G Fit to Casimir Pressure")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../plots/casimir_pressure_fit.png")
print("Saved fit plot to ../plots/casimir_pressure_fit.png")

# --- Plot Residuals ---
plt.figure(figsize=(6.5, 3.5))
plt.scatter(x, residuals, s=20, color="purple")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Gap (nm)")
plt.ylabel("Residuals (data - fit)")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/residuals_pressure_fit.png")
print("Saved residuals plot to ../plots/residuals_pressure_fit.png")
