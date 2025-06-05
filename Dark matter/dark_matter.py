import numpy as np
import matplotlib.pyplot as plt
import os

# --- Simulation Parameters ---
GRID_SIZE = 256
TIME_STEPS = 5000
SAVE_INTERVAL = 500
NUM_SOLITONS = 500
VELOCITY_MAGNITUDE = 0.2

# --- Initialization ---
field = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
velocity_x = np.zeros_like(field)
velocity_y = np.zeros_like(field)

np.random.seed(42)
x_pos = np.random.randint(0, GRID_SIZE, NUM_SOLITONS)
y_pos = np.random.randint(0, GRID_SIZE, NUM_SOLITONS)

# --- Add solitons ---
for i in range(NUM_SOLITONS):
    dx = np.arange(GRID_SIZE) - x_pos[i]
    dy = np.arange(GRID_SIZE)[:, None] - y_pos[i]
    r2 = dx**2 + dy**2
    profile = np.exp(-r2 / 4.0)
    field += profile

    if i >= NUM_SOLITONS - 4:  # Last 4 solitons are moving
        theta = np.random.uniform(0, 2 * np.pi)
        velocity_x[y_pos[i], x_pos[i]] = VELOCITY_MAGNITUDE * np.cos(theta)
        velocity_y[y_pos[i], x_pos[i]] = VELOCITY_MAGNITUDE * np.sin(theta)

# --- Output directory ---
os.makedirs("galactic_output", exist_ok=True)

# --- Simulation Loop ---
for step in range(TIME_STEPS):
    grad_x = np.gradient(field, axis=1)
    grad_y = np.gradient(field, axis=0)

    field += -(velocity_x * grad_x + velocity_y * grad_y) * 0.1

    # Simple diffusion to avoid blow-up
    lap = np.gradient(np.gradient(field, axis=0), axis=0) + np.gradient(np.gradient(field, axis=1), axis=1)
    field += 0.01 * lap

    # Save field
    if step % SAVE_INTERVAL == 0:
        np.save(f"galactic_output/field_{step}.npy", field)
        print(f"Saved step {step}")

print("Simulation complete.")