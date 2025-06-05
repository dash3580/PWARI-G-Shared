import numpy as np
import os

# --- Configuration ---
GRID_SIZE = 1024
DX = 1.0
DT = 0.1
STEPS = 100000
SAVE_EVERY = 500
phi0 = 1.0
lam = 1.0
V_INIT = 0.2
np.random.seed(42)

# --- Output ---
output_dir = "dark_matter_output"
os.makedirs(output_dir, exist_ok=True)

# --- Grid Initialization ---
phi = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
phi_dot = np.zeros_like(phi)
x = np.arange(GRID_SIZE)
X, Y = np.meshgrid(x, x, indexing='ij')

# --- Soliton Initialization ---
def add_soliton(x0, y0, moving=False):
    r2 = (X - x0)**2 + (Y - y0)**2
    bump = phi0 * np.exp(-r2 / 4.0)
    phi[:] += bump
    if moving:
        phi_dot[:] += V_INIT * (X - x0) * bump

for _ in range(500):
    i = np.random.randint(0, GRID_SIZE)
    j = np.random.randint(0, GRID_SIZE)
    moving = np.random.rand() < 0.01  # 1% moving
    add_soliton(i, j, moving)

# --- Evolution Function ---
def evolve(phi, phi_dot):
    laplacian = (
        -4 * phi +
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
    ) / DX**2
    acc = laplacian - lam * phi * (phi**2 - phi0**2)
    phi_dot += DT * acc
    phi += DT * phi_dot
    return phi, phi_dot

# --- Main Loop ---
for step in range(STEPS):
    phi, phi_dot = evolve(phi, phi_dot)
    if step % SAVE_EVERY == 0:
        np.save(os.path.join(output_dir, f"phi_{step:06d}.npy"), phi)
        print(f"Step {step} saved.")

# Save final state
np.save(os.path.join(output_dir, "phi_final.npy"), phi)
print("Simulation complete.")
