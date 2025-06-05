import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from glob import glob
from scipy.ndimage import gaussian_filter
import os

# --- CONFIGURATION ---
output_dir = "dark_matter_output"
phi_files = sorted([
    f for f in glob(os.path.join(output_dir, "phi_*.npy"))
    if "final" not in f
])

GRID_SIZE = 1024
phi0 = 1.0
lam = 1.0
seed = 42
smoothing_sigma = 2.0
fps = 20

# --- REBUILD SOLITON ORIGIN MAP ---
np.random.seed(seed)
soliton_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
num_solitons = 500
for _ in range(num_solitons):
    i = np.random.randint(0, GRID_SIZE)
    j = np.random.randint(0, GRID_SIZE)
    moving = np.random.rand() < 0.01
    if moving:
        soliton_mask[i, j] = 1  # mark moving (1), stationary remains 0

# --- ENERGY DENSITY FUNCTION ---
def compute_density(phi):
    grad_x = np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)
    grad_y = np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)
    grad2 = (grad_x**2 + grad_y**2) / 4.0
    potential = (lam / 4.0) * (phi**2 - phi0**2)**2
    return 0.5 * grad2 + potential

# --- SETUP PLOT ---
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap='bwr', vmin=-0.1, vmax=0.1)
ax.set_title("Dark Matter Evolution\nRed = Moving | Blue = Stationary")
plt.axis('off')

# --- ANIMATION FUNCTION ---
def update(frame_idx):
    phi = np.load(phi_files[frame_idx])
    density = compute_density(phi)
    smoothed = gaussian_filter(density, sigma=smoothing_sigma)

    # Encode blue (stationary) as negative, red (moving) as positive in 'bwr'
    color_layer = np.zeros_like(smoothed)
    color_layer += smoothed * (soliton_mask == 0)   # blue
    color_layer -= smoothed * (soliton_mask == 1)   # red

    im.set_data(color_layer)
    return [im]

# --- CREATE AND SAVE ANIMATION ---
ani = animation.FuncAnimation(fig, update, frames=len(phi_files), blit=True, interval=50)

gif_path = os.path.join(output_dir, "dark_matter_evolution.gif")
ani.save(gif_path, fps=fps, dpi=150)

plt.close()

print(f"✅ Animation saved to: {gif_path}")
