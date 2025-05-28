import numpy as np
import os
import re
import csv

# Directory containing the .npy files
data_dir = "npy_cycle"  # Change if your folder is named differently
output_csv = "twist_energy_profile.csv"

# Find all twist_energy files
twist_files = sorted([
    f for f in os.listdir(data_dir) if re.match(r"twist_energy_\d+\.npy", f)
], key=lambda x: int(re.search(r"\d+", x).group()))

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["step", "total_twist_energy", "twist_wavefront_radius"])

    for file in twist_files:
        step = int(re.search(r"twist_energy_(\d+).npy", file).group(1))
        data = np.load(os.path.join(data_dir, file))

        # Total energy as before
        total_energy = np.sum(data)

        # --- NEW: Calculate wavefront radius from threshold
        threshold = 1e-11
    
        grid = np.indices(data.shape).astype(np.float32)
        center = np.array(data.shape)[:, None, None, None] / 2.0
        r2 = np.sum((grid - center) ** 2, axis=0)
        r_flat = np.sqrt(r2.flatten())
        data_flat = np.abs(data.flatten())

        mask = data_flat > threshold
        wavefront_radius = np.max(r_flat[mask]) if np.any(mask) else 0.0

        # Write all three values to CSV
        writer.writerow([step, total_energy, wavefront_radius])

print(f"✅ Extracted energy + wavefront profile saved to: {output_csv}")
