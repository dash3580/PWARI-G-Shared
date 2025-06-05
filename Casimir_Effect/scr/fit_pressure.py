# fit_pressure.py
# Fits a power law A/d^n to BCS Casimir pressure data from dist_v4.h5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load Data ---
print("Loading data from Casimir_pressure_dist_v4.h5")
df = pd.read_hdf("../data/Casimir_pressure_dist_v4.h5", key="data")

# Extract columns
x = df['Gap (nm)'].values
y = np.abs(df['P BCS (Pa)'].values)  # Use magnitude since pressure is negative

# Filter valid data
mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

# --- Define model ---
def power_law(d, A, n):
    return A / d**n

# --- Fit model ---
popt, pcov = curve_fit(power_law, x, y)
A_fit, n_fit = popt

# --- Print result ---
print(f"Best-fit exponent: n = {n_fit:.4f}")
print(f"Best-fit amplitude: A = {A_fit:.4e}")

# --- Generate fitted curve ---
x_fit = np.linspace(min(x), max(x), 500)
y_fit = power_law(x_fit, A_fit, n_fit)

# --- Plot ---
plt.figure(figsize=(6, 4))
plt.scatter(x, y, label="|BCS Data|", color="blue", s=20)
plt.plot(x_fit, y_fit, label=f"Fit: A/d^{{{n_fit:.3f}}}", color="red")
plt.xlabel("Gap (nm)")
plt.ylabel("Pressure Magnitude (Pa)")
plt.title("PWARI-G Fit to BCS Casimir Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Save output ---
plt.savefig("../plots/casimir_pressure_fit.png")
print("Saved plot to ../plots/casimir_pressure_fit.png")