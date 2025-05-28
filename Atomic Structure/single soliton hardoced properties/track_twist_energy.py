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

# Open CSV and write header
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["step", "total_twist_energy"])

    for file in twist_files:
        step = int(re.search(r"twist_energy_(\d+).npy", file).group(1))
        data = np.load(os.path.join(data_dir, file))
        total_energy = np.sum(data)

        writer.writerow([step, total_energy])

print(f"✅ Extracted energy profile saved to: {output_csv}")
