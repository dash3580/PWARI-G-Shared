import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

folder = "npy_cycle"
phi_files = [f for f in os.listdir(folder) if f.startswith("phi_") and f.endswith(".npy")]

data = []
dx = 0.2  # grid spacing for physical units

for f in sorted(phi_files, key=lambda x: int(re.findall(r"\d+", x)[0])):
    step = int(re.findall(r"\d+", f)[0])
    path = os.path.join(folder, f)
    phi = np.load(path)

    # Build radial grid
    grid = np.indices(phi.shape)
    center = np.array(phi.shape)[:, None, None, None] / 2.0
    r = np.sqrt(np.sum((grid - center) ** 2, axis=0))

    # Bin by radius
    r_flat = r.flatten()
    phi_flat = phi.flatten()
    bins = np.arange(0, np.max(r_flat), 1.0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.zeros_like(bin_centers)

    for i in range(len(bin_centers)):
        in_bin = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
        if np.any(in_bin):
            profile[i] = np.mean(phi_flat[in_bin])

    # Effective radius where φ drops to 1/e of peak
    max_phi = np.max(profile)
    threshold = max_phi / np.e
    radius_index = np.argmax(profile < threshold)
    effective_radius = bin_centers[radius_index] * dx  # convert to physical units

    data.append((step, effective_radius))

# Save results
df = pd.DataFrame(data, columns=["step", "effective_radius"])
df.to_csv("soliton_radius_over_time.csv", index=False)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["effective_radius"], marker="o", linestyle="-")
plt.xlabel("Simulation Step")
plt.ylabel("Effective Radius (grid units × dx)")
plt.title("Soliton Radius Evolution Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("soliton_radius_trend.png")
plt.show()
