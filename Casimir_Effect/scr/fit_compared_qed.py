# fit_compare_qed.py
# Compares QED (1/d^4) and PWARI-G (fit) models against BCS Casimir pressure data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare

# --- Load Data ---
print("Loading BCS Casimir pressure data from HDF5 file...")
df = pd.read_hdf("../data/Casimir_pressure_dist_v4.h5", key="data")

d = df['Gap (nm)'].values
P = df['P BCS (Pa)'].values

# Filter valid data
d_mask = (d > 0) & np.isfinite(d) & np.isfinite(P)
d = d[d_mask]
P = np.abs(P[d_mask])  # Use magnitude for fitting

# --- Define models ---
def power_law(d, A, n):
    return A / d**n

def qed_model(d):
    A_qed = 1e-2  # Arbitrary normalization for visual comparison
    return A_qed / d**4

# --- Fit PWARI-G model ---
popt, _ = curve_fit(power_law, d, P)
A_pw, n_pw = popt
P_pw = power_law(d, A_pw, n_pw)

# --- Evaluate QED model ---
P_qed = qed_model(d)

# --- Normalize QED to same scale ---
scale = np.sum(P) / np.sum(P_qed)
P_qed_scaled = P_qed * scale

# --- Chi-squared ---
chi2_pw = np.sum((P - P_pw)**2 / P_pw)
chi2_qed = np.sum((P - P_qed_scaled)**2 / P_qed_scaled)

# --- Print results ---
print(f"PWARI-G fit: n = {n_pw:.4f}, Chi^2 = {chi2_pw:.4e}")
print(f"QED model: n = 4 (fixed), Chi^2 = {chi2_qed:.4e}")

# --- Plot ---
d_plot = np.linspace(min(d), max(d), 500)
P_pw_plot = power_law(d_plot, A_pw, n_pw)
P_qed_plot = qed_model(d_plot) * scale

plt.figure(figsize=(6, 5))
plt.scatter(d, P, label="BCS Data", color="blue", s=20)
plt.plot(d_plot, P_pw_plot, label=f"PWARI-G Fit: 1/d^{n_pw:.2f}", color="red")
plt.plot(d_plot, P_qed_plot, label="QED Model: 1/d⁴", color="green", linestyle="--")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Gap (nm)")
plt.ylabel("Pressure (Pa)")
plt.title("Casimir Pressure Fit Comparison (Log-Log)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("../plots/qed_comparison.png")


# --- Save Plot ---
plt.savefig("../plots/qed_comparison.png")
print("Saved plot to ../plots/qed_comparison.png")