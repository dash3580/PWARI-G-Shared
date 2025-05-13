# PWARI-G Casimir Effect Study

This repository accompanies the research paper:

**"PWARI-G: Casimir Force Analysis from First Principles via Breathing Soliton Fields"**  
Author: Darren Blair

📄 [Read the full paper (PDF)](./2__PWARI_G__Casimir.pdf)

---

## 🔬 What is PWARI-G?

PWARI-G stands for **Photon Wave Absorption and Reshaping Interpretation with Gravity**.  
It is a fully deterministic, wave-only field framework that models particles as nonlinear breathing solitons. This repository explores how PWARI-G accounts for the **Casimir effect**—traditionally attributed to quantum vacuum fluctuations—using classical field dynamics only.

---

## 📈 Key Results

- **Power-law pressure fit:**  
  Experimental Casimir pressure data fits best to a PWARI-G predicted law of  
  \[
  P(d) = rac{A}{d^{3.193}}
  \]
  compared to the QED prediction of \(1/d^4\)

- **Residual Analysis:**  
  Residuals show systematic deviations at short range, offering insight into field confinement refinements

- **Dynamic test:**  
  Mechanical frequency shift in a nano-resonator matches a PWARI-G force exponent of \( n pprox 1.77 \)

---

## 📂 Repository Structure

```
PWARI-G-Shared/
├── Casimir_Effect/
│   ├── data/
│   │   ├── Casimir_pressure_dist_v4.h5
│   │   └── Optimized_mech_freqs.txt
│   ├── scr/
│   │   ├── fit_pressure.py
│   │   ├── fit_mechanical.py
│   │   ├── fit_compare_qed.py
│   │   └── residuals_plot.py
│   ├── figs/
│   │   ├── casimir_pressure_fit.png
│   │   ├── mechanical_shift_fit.png
│   │   ├── qed_comparison.png
│   │   └── residuals_pressure_fit.png
│   └── 2__PWARI_G__Casimir.pdf
├── README.md
```

---

## 🧪 Reproducing the Results

Clone the repository and install dependencies:

```bash
git clone https://github.com/dash3580/PWARI-G-Shared.git
cd PWARI-G-Shared/Casimir_Effect
pip install -r requirements.txt
```

Then run:

```bash
cd scr
python fit_pressure.py         # fits Casimir pressure vs distance
python fit_mechanical.py      # fits resonator frequency shift
python fit_compare_qed.py     # overlays QED vs PWARI-G
python residuals_plot.py      # generates residuals plot
```

---

## 📊 Dependencies

- Python 3.10+
- numpy
- pandas
- matplotlib
- scipy
- tables (`pip install tables` for .h5 reading)

---

## 📜 License & Credits

- Experimental data sourced from public BCS-corrected repositories
- All plots and analysis code authored by Darren Blair
- This project is released under the MIT License

---

## 📫 Contributions Welcome

If you're a physicist, coder, or curious mind and want to push the theory further, spot issues, or propose refinements—open an issue or contact me.
