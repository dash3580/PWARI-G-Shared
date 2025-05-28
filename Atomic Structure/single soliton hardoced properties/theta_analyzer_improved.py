
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import csv

# === CONFIG ===
input_folder = "slices"          
output_csv = "theta_shells_summary_improved.csv"
slice_prefix = "theta_slice_"    
gaussian_sigma = 1.0             
min_prominence = 1e-6            

def analyze_theta_slice(file_path):
    data = np.load(file_path)
    center = np.array(data.shape) // 2
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    max_radius = r.max()
    radial_theta = np.zeros(max_radius + 1)
    counts = np.zeros(max_radius + 1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ri = r[i, j]
            radial_theta[ri] += data[i, j]
            counts[ri] += 1

    radial_theta /= np.maximum(counts, 1)
    radial_smoothed = gaussian_filter(radial_theta, sigma=gaussian_sigma)

    # Find both positive and negative peaks
    pos_peaks, _ = find_peaks(radial_smoothed, prominence=min_prominence)
    neg_peaks, _ = find_peaks(-radial_smoothed, prominence=min_prominence)

    all_peaks = sorted(set(pos_peaks.tolist() + neg_peaks.tolist()))
    all_peak_values = radial_smoothed[all_peaks]

    return {
        "num_shells": len(all_peaks),
        "shell_radii": all_peaks,
        "theta_peak_values": all_peak_values.tolist()
    }

# === RUN SCRIPT ===
results = []
for fname in sorted(os.listdir(input_folder)):
    if fname.startswith(slice_prefix) and fname.endswith(".npy"):
        step_str = fname.replace(slice_prefix, "").replace(".npy", "")
        try:
            step = int(step_str)
        except ValueError:
            continue

        path = os.path.join(input_folder, fname)
        metrics = analyze_theta_slice(path)
        results.append({
            "step": step,
            "num_shells": metrics["num_shells"],
            "shell_radii": metrics["shell_radii"],
            "theta_peak_values": metrics["theta_peak_values"]
        })

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "num_shells", "shell_radii", "theta_peak_values"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Improved analysis complete. Saved to {output_csv}")
