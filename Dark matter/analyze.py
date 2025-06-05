import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from scipy.ndimage import gaussian_filter

# --- CONFIGURATION ---
DX = 1.0
phi0 = 1.0
lam = 1.0
G = 1.0  # gravitational constant (natural units)
grid_size = 1024
center = np.array([grid_size // 2, grid_size // 2])

# --- PATH SETUP ---
output_dir = "dark_matter_output"
phi_files = sorted([
    f for f in glob(os.path.join(output_dir, "phi_*.npy"))
    if "final" not in f
])

csv_path = os.path.join(output_dir, "halo_profiles.csv")
image_path = os.path.join(output_dir, "final_density_map.png")

# --- RADIAL SETUP ---
r_max = grid_size // 2
radii = np.arange(1, r_max)
bin_counts = np.zeros_like(radii, dtype=np.float32)

# --- ENERGY DENSITY FUNCTION ---
def compute_density(phi):
    grad_x = np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)
    grad_y = np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)
    grad2 = (grad_x**2 + grad_y**2) / (4 * DX**2)
    potential = (lam / 4.0) * (phi**2 - phi0**2)**2
    return 0.5 * grad2 + potential

# --- RADIAL PROFILE FUNCTION ---
def radial_profile(density):
    y, x = np.indices(density.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.int32)
    mass_profile = np.zeros_like(radii, dtype=np.float32)
    for i, radius in enumerate(radii):
        mask = (r == radius)
        mass_profile[i] = np.sum(density[mask])
        bin_counts[i] += np.sum(mask)
    return mass_profile

# --- ANALYSIS LOOP ---
all_data = []

print(f"Processing {len(phi_files)} .npy files...")

for path in phi_files:
    phi = np.load(path)
    density = compute_density(phi)
    m_r = radial_profile(density)
    enclosed_mass = np.cumsum(m_r) * 2 * np.pi * DX
    velocity = np.sqrt(G * enclosed_mass / (radii + 1e-6))
    step = int(os.path.basename(path).split('_')[1].split('.')[0])
    for r, v, m in zip(radii, velocity, enclosed_mass):
        all_data.append({'step': step, 'radius': r, 'mass': m, 'velocity': v})

# --- SAVE CSV ---
df = pd.DataFrame(all_data)
df.to_csv(csv_path, index=False)
print(f"Saved CSV to: {csv_path}")

# --- VISUALIZE LAST FRAME ---
latest_phi = np.load(phi_files[-1])
density_final = compute_density(latest_phi)

plt.figure(figsize=(6, 5))
plt.imshow(gaussian_filter(density_final, sigma=2), cmap='inferno')
plt.colorbar(label="Energy Density")
plt.title(f"Final Soliton Field Energy Density\n{os.path.basename(phi_files[-1])}")
plt.savefig(image_path)
plt.close()
print(f"Saved final density map to: {image_path}")
