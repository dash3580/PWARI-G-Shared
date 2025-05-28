# soliton_self_twist_cycle.py (PWARI-G upgraded)
# Self-gravitating soliton with twist and redshift-based evolution

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Simulation parameters ---
GRID_SIZE = (96, 96, 96)
DX = 0.2
DT = 0.005
STEPS = 100000
GRAVITY_ALPHA = 1.0
SNAP_THRESHOLD = 1e-12
SNAP_ALPHA = 0.2

# --- Utilities ---
def laplacian(f):
    return sum(np.gradient(np.gradient(f, DX, axis=i), DX, axis=i) for i in range(3))

def make_damping_mask(shape, edge_width=10):
    mask = np.ones(shape)
    for axis in range(3):
        for i in range(edge_width):
            val = (i + 1) / edge_width
            front = [slice(None)] * 3
            back = [slice(None)] * 3
            front[axis] = i
            back[axis] = -1 - i
            mask[tuple(front)] *= val
            mask[tuple(back)] *= val
    return mask

def generate_twist_pulse(theta_dot, grad_theta, snap_zone):
    pulse = np.zeros_like(theta_dot)
    norm = np.sqrt(sum(g**2 for g in grad_theta)) + 1e-10
    for i in range(3):
        pulse += grad_theta[i] * theta_dot * snap_zone / norm
    pulse *= 0.05
    return pulse

# --- Initialization ---
def initialize_fields():
    global phi_init
    grid = np.indices(GRID_SIZE).astype(np.float32)
    center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
    r2 = np.sum((grid - center)**2, axis=0) * DX**2
    phi = np.exp(-r2 * 2.0)
    phi_dot = np.zeros_like(phi)
    theta = np.zeros_like(phi)
    theta_dot = np.zeros_like(phi)
    twist_wave = np.zeros_like(phi)

    grad_phi = np.gradient(phi, DX)
    grad_phi_sq = sum(g**2 for g in grad_phi)
    rho_init = 0.5 * phi_dot**2 + 0.5 * grad_phi_sq
    gravity = np.exp(-0.5 * rho_init)

    phi_init = np.copy(phi)
    return phi, phi_dot, theta, theta_dot, gravity, twist_wave

# --- Evolution Rules ---
def evolve_phi(phi, phi_dot, gravity, theta_dot, phi_lag=None, TAU=None):
    global phi_init
    ELASTIC_COEFF = 0.5
    mismatch = theta_dot**2
    base_phi_ddot = (1.0 / gravity) * (laplacian(phi) - phi**3 - phi * mismatch)
    if phi_lag is not None and TAU is not None:
        lag_term = (phi - phi_lag) / TAU
        elastic_pull = ELASTIC_COEFF * (phi_init - phi)
        phi_ddot = base_phi_ddot + elastic_pull - lag_term + elastic_pull
    else:
        phi_ddot = base_phi_ddot
    phi_dot += DT * phi_ddot
    phi += DT * phi_dot
    return phi, phi_dot

def evolve_twist_wave(twist_wave):
    wave_ddot = laplacian(twist_wave)
    twist_wave += DT * wave_ddot
    return twist_wave

def evolve_theta(phi, phi_dot, theta, theta_dot, gravity, twist_wave):
    grad_theta = np.gradient(theta, DX)
    lap_theta = laplacian(theta)
    grad_theta_sq = sum(g**2 for g in grad_theta)
    twist_strain = phi**2 * (grad_theta_sq + theta_dot**2)

    grid = np.indices(GRID_SIZE).astype(np.float32)
    center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
    r2 = np.sum((grid - center)**2, axis=0) * DX**2
    snap_bias = 1.0 / (1.0 + r2)
    snap_zone = (twist_strain * snap_bias) > SNAP_THRESHOLD
    snap_pressure = np.zeros_like(theta)
    snap_pressure[snap_zone] = theta_dot[snap_zone]

    theta_ddot = gravity * lap_theta + 0.01 * phi_dot - SNAP_ALPHA * snap_pressure
    discarded_energy = np.sum(twist_strain[snap_zone]) * DX**3

    if np.any(snap_zone):
        emission_pulse = generate_twist_pulse(theta_dot, grad_theta, snap_zone)
        twist_wave[snap_zone] += emission_pulse[snap_zone]
        theta_dot[snap_zone] = 0.0
        emitted_energy = np.sum((emission_pulse[snap_zone])**2) * DX**3
        print(f"Emitted twist energy from snap: {emitted_energy:.5e}")

    theta_dot *= 0.995
    theta_dot += DT * theta_ddot
    theta += DT * theta_dot
    return theta, theta_dot, discarded_energy, snap_zone, twist_wave

def evolve_gravity(gravity, phi, phi_dot, theta_dot, theta, snap_zone=None):
    global phi_init
    grid = np.indices(GRID_SIZE).astype(np.float32)
    center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
    r2 = sum((grid[i] - center[i])**2 for i in range(3)) * DX**2
    distance_weight = 1.0 / (1.0 + r2)

    grad_phi = np.gradient(phi, DX)
    grad_phi_sq = sum(g**2 for g in grad_phi)
    grad_theta = np.gradient(theta, DX)
    grad_theta_sq = sum(g**2 for g in grad_theta)
    rho_phi = 0.5 * phi_dot**2 + 0.5 * grad_phi_sq
    rho_twist = 0.5 * phi**2 * theta_dot**2 + 0.5 * phi**2 * grad_theta_sq
    rho_weighted = rho_phi + distance_weight * rho_twist

    GRAVITY_RELAX = 0.005
    gravity -= DT * GRAVITY_ALPHA * rho_weighted
    if snap_zone is not None:
        gravity[snap_zone] += GRAVITY_RELAX * (1.0 - gravity[snap_zone])
    phi_recoil_zone = (phi < phi_init) & (phi_dot < 0)
    gravity[phi_recoil_zone] += GRAVITY_RELAX * (1.0 - gravity[phi_recoil_zone])
    return gravity

# --- Main Simulation ---
def run():
    phi, phi_dot, theta, theta_dot, gravity, twist_wave = initialize_fields()
    damping_mask = make_damping_mask(GRID_SIZE)
    os.makedirs("frames_cycle", exist_ok=True)
    os.makedirs("npy_cycle", exist_ok=True)
    log = open("cycle_log.csv", "w")
    log.write("step,phi_max,twist_energy_max,soliton_energy,twist_energy,gravity_energy,discarded_energy\n")

    phi_buffer = []
    TAU_STEPS = 1

    for step in range(STEPS):
        phi_lag = phi_buffer[0] if len(phi_buffer) >= TAU_STEPS else None
        phi, phi_dot = evolve_phi(phi, phi_dot, gravity, theta_dot, phi_lag, TAU_STEPS * DT)
        twist_wave = evolve_twist_wave(twist_wave)
        theta, theta_dot, discarded_energy, snap_zone, twist_wave = evolve_theta(phi, phi_dot, theta, theta_dot, gravity, twist_wave)
        gravity = evolve_gravity(gravity, phi, phi_dot, theta_dot, theta, snap_zone)

        if snap_zone is not None and np.any(snap_zone):
            grid = np.indices(GRID_SIZE).astype(np.float32)
            center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
            r2 = np.sum((grid - center)**2, axis=0) * DX**2
            deletion_radius = 2.0
            wipe_zone = r2 > deletion_radius**2
            phi[wipe_zone] = 0.0
            phi_dot[wipe_zone] = 0.0

        for field in [phi, phi_dot, theta, theta_dot, twist_wave]:
            field *= damping_mask

        if step % 500 == 0:
            twist_energy = phi**2 * theta_dot**2
            soliton_energy = np.sum(phi**2 + phi_dot**2)
            twist_total = np.sum(twist_energy)
            gravity_energy = np.sum((1 - gravity)**2)
            twist_max = np.max(twist_energy)
            print(f"Step {step:4d} | φ_max: {np.max(phi):.5f} | Twist max: {twist_max:.5e} | Twist total: {twist_total:.5e} | Soliton E: {soliton_energy:.3e} | Gravity E: {gravity_energy:.3e} | Discarded E: {discarded_energy:.3e}")

            log.write(f"{step},{np.max(phi):.5f},{twist_max:.5f},{soliton_energy:.5f},{twist_total:.5f},{gravity_energy:.5f},{discarded_energy:.5f}\n")

            mid = GRID_SIZE[2] // 2
            plt.imshow(twist_wave[:, :, mid], cmap='inferno')
            plt.title(f"Detached Twist Wave (step {step})")
            plt.colorbar()
            plt.savefig(f"frames_cycle/wave_{step:04d}.png")
            plt.close()

            np.save(f"npy_cycle/phi_{step:04d}.npy", phi)
            np.save(f"npy_cycle/theta_{step:04d}.npy", theta)
            np.save(f"npy_cycle/gravity_{step:04d}.npy", gravity)
            np.save(f"npy_cycle/twist_energy_{step:04d}.npy", twist_energy)
            np.save(f"npy_cycle/phi_dot_{step:04d}.npy", phi_dot)
            np.save(f"npy_cycle/theta_dot_{step:04d}.npy", theta_dot)
            np.save(f"npy_cycle/twist_wave_{step:04d}.npy", twist_wave)

        phi_buffer.append(np.copy(phi))
        if len(phi_buffer) > TAU_STEPS:
            phi_buffer.pop(0)

    log.close()
    print("Simulation complete. Log saved as cycle_log.csv")

if __name__ == "__main__":
    run()