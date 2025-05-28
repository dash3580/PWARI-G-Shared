import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd

# --- Configuration ---
folder = "npy_cycle"
save_images = False  # Set True to save images per step
dx = 0.2

# --- Collect available theta files ---
theta_files = sorted([f for f in os.listdir(folder) if f.startswith("theta_") and f.endswith(".npy")])
steps = []
for f in theta_files:
    match = re.search(r"theta_(\d+)\.npy", f)
    if match:
        steps.append(int(match.group(1)))

results = []

# --- Process each step ---
for step in steps:
    try:
        path = os.path.join(folder, f"theta_{step}.npy")
        theta = np.load(path)

        # Gradients
        grad_theta = np.gradient(theta, axis=(0, 1, 2))
        grad_mag = np.sqrt(sum(g**2 for g in grad_theta))

        # Grid
        x, y, z = np.indices(theta.shape)
        center = np.array(theta.shape)[:, None, None, None] / 2.0
        r_vec = np.stack((x - center[0], y - center[1], z - center[2]))
        r_hat = r_vec / (np.sqrt(np.sum(r_vec**2, axis=0)) + 1e-10)

        # Dot product ∇θ · r̂
        grad_dot_r = sum(grad_theta[i] * r_hat[i] for i in range(3))
        winding_scalar = np.sum(grad_dot_r) * (dx**3) / (2 * np.pi)

        # Optional plots
        if save_images:
            mid = theta.shape[2] // 2
            theta_slice = theta[:, :, mid]
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(theta_slice, cmap="coolwarm")
            plt.title(f"Theta Field (step {step})")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(grad_mag[:, :, mid], cmap="viridis")
            plt.title("|∇θ| (step {step})")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"theta_spin_scan_{step:05d}.png")
            plt.close()

        results.append([
            step,
            winding_scalar,
            float(np.mean(grad_mag)),
            float(np.max(grad_mag))
        ])

        print(f"✓ Step {step}: Winding = {winding_scalar:.4f}")

    except Exception as e:
        print(f"Skipping step {step} due to error: {e}")
        continue

# --- Save results ---
df = pd.DataFrame(results, columns=["step", "winding", "mean_grad_mag", "max_grad_mag"])
df.sort_values("step").to_csv("pwari_spin_diagnostics.csv", index=False)
print("\n✅ Scan complete. Saved to pwari_spin_diagnostics.csv")
