import os
import re
import numpy as np
import pandas as pd

# --- Config ---
npy_dir = "./npy_cycle"  # Adjust this to match your path
dx = 0.2
grid_shape = (96, 96, 96)
bin_edges = np.arange(0, 50, 1.0)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
n1_range = (bin_centers >= 8.5) & (bin_centers <= 10.5)
n2_range = (bin_centers >= 4.5) & (bin_centers <= 6.5)
n3_range = (bin_centers >= 2.0) & (bin_centers <= 3.5)

# --- Find Steps ---
pattern = re.compile(r"phi_(\d+)\.npy")
steps = sorted(int(pattern.search(f).group(1)) for f in os.listdir(npy_dir) if pattern.match(f))

# --- Prepare storage ---
results = []

# --- Core loop ---
for step in steps:
    try:
        phi = np.load(os.path.join(npy_dir, f"phi_{step}.npy"))
        phi_dot = np.load(os.path.join(npy_dir, f"phi_dot_{step}.npy"))
        theta_dot = np.load(os.path.join(npy_dir, f"theta_dot_{step}.npy"))
        gravity = np.load(os.path.join(npy_dir, f"gravity_{step}.npy"))

        # Radial grid
        grid = np.indices(phi.shape)
        center = np.array(phi.shape)[:, None, None, None] / 2.0
        r = np.sqrt((grid[0] - center[0])**2 + (grid[1] - center[1])**2 + (grid[2] - center[2])**2)

        r_flat = r.flatten()
        phi_flat = phi.flatten()
        phi_dot_flat = phi_dot.flatten()
        theta_dot_flat = theta_dot.flatten()
        gravity_flat = gravity.flatten()

        shell_energy_density = (phi_flat**2) * (theta_dot_flat**2)
        radial_energy = np.zeros_like(bin_centers)
        for i in range(len(bin_centers)):
            in_bin = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i+1])
            radial_energy[i] = np.sum(shell_energy_density[in_bin])

        # Shell integrals
        E_n1 = np.sum(radial_energy[n1_range])
        E_n2 = np.sum(radial_energy[n2_range])
        E_n3 = np.sum(radial_energy[n3_range])
        E_total = E_n1 + E_n2 + E_n3

        # Global fields
        twist_total = np.sum(shell_energy_density) * dx**3
        soliton_energy = np.sum(phi_flat**2 + phi_dot_flat**2) * dx**3
        gravity_energy = np.sum((1 - gravity_flat)**2) * dx**3

        results.append([
            step, E_n1, E_n2, E_n3, E_total,
            E_n1 / E_total if E_total else 0,
            E_n2 / E_total if E_total else 0,
            E_n3 / E_total if E_total else 0,
            twist_total, soliton_energy, gravity_energy
        ])

    except Exception as e:
        print(f"Step {step} skipped: {e}")
        continue

# --- Save results ---
df = pd.DataFrame(results, columns=[
    "step", "E_n1", "E_n2", "E_n3", "E_total",
    "E_n1_frac", "E_n2_frac", "E_n3_frac",
    "twist_total", "soliton_energy", "gravity_energy"
])
df.sort_values("step").to_csv("pwari_metrics_summary.csv", index=False)
print("✅ Analysis complete. Saved to 'pwari_metrics_summary.csv'")
