# PWARI-G Black Hole Collapse Simulations

This repository contains four scalar field collapse simulations using the PWARI-G (Photon Wave Absorption and Reshaping Interpretation with Gravity) framework. Each folder explores a variation in how output is generated or interpreted, while the core evolution physics remains consistent.

## 🔧 Core Theory

All simulations evolve a scalar breathing soliton field \( \phi(r, t) \) in 1+1D spacetime with dynamic redshift coupling:

- Lapse function \( \alpha(r, t) \) updates from energy density via a relaxation equation
- Spatial metric \( a(r, t) \) is fixed to 1 for simplicity
- Collapse proceeds from a Gaussian initial pulse
- No horizons or singularities form; redshift saturates smoothly

---

## 📁 Folder Overview

### `collapse/`
**Purpose**: Baseline simulation of soliton collapse.

- Outputs: `collapse_fields.npz`
- Tracks: \( \phi \), \( \Pi \), \( \Phi \), \( R \), \( \alpha \)
- No Schwarzschild comparison or observational matching

---

### `collapse_ric/`
**Purpose**: Adds Ricci scalar diagnostics and redshift tail fitting.

- Adds plot of Schwarzschild lapse fit: `alpha_vs_schwarzschild.png`
- Computes effective gravitational mass \( M_{\text{eff}} \) from redshift tail
- Saves results to `effective_mass.txt`

---

### `collapse_sch/`
**Purpose**: Includes detailed tail logging and overlays Schwarzschild curves with numerical debug output.

- Outputs core location, α minimum, and maximum field values every 500 steps
- Helps identify numerical issues (e.g., saturation, divergence)
- Focused on precision analysis of Schwarzschild matching

---

### `collapse_com/`
**Purpose**: Observational comparison with real black hole mass catalogs.

- Sweeps multiple amplitudes \( A \)
- Computes \( M_{\text{eff}} \) for each and fits a scaling factor
- Compares predicted vs observed black hole masses
- Outputs:
  - `pwari_mass_sweep.csv`
  - `pwari_comparison_results.csv`
  - `pwari_vs_observed_mass_comparison.png`

---

## 📈 How to Run

Each folder includes an `evolve_collapse.py`. To run:

```bash
cd collapse_ric  # or any other
python evolve_collapse.py
```

Ensure the following Python packages are installed:

```bash
pip install numpy matplotlib scipy
```

---

## 🧠 Interpretation

- Redshift dip \( \alpha \ll 1 \): mimics gravitational collapse
- Core tracking: shows how the breathing soliton stabilizes
- Mass fitting: compares numerically computed tail to Schwarzschild
- Observational residuals: show where PWARI-G diverges or agrees with AGN mass data

---

## 🔭 Future Work

- Add spinor and gauge fields for multi-mode feedback
- Migrate to 3+1D evolution
- Extend to gravitational wave emission and lensing analysis

---

## Author

Developed by **Darren Blair** using AI assistance for simulation, LaTeX reporting, and observational cross-validation.
