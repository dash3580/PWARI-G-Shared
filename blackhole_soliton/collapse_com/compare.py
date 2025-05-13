import os
import pandas as pd
import matplotlib.pyplot as plt

# Print current working directory
print(f"Running from: {os.getcwd()}")

# === CONFIGURATION ===
catalog_path = "black_hole_catalog_sample.csv"
pwari_mass_sweep = "output/pwari_mass_sweep.csv"
output_plot_path = "output/pwari_vs_observed_mass_comparison.png"
output_csv_path = "output/pwari_comparison_results.csv"

# === LOAD DATA ===
try:
    catalog_df = pd.read_csv(catalog_path)
    pwari_df = pd.read_csv(pwari_mass_sweep)
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    exit(1)

# === Ensure column names match
pwari_df.rename(columns={"PWARI_G_Meff": "PWARI_G_Meff"}, inplace=True)

# === MERGE ON OBJECT NAME ===
merged = pd.merge(catalog_df, pwari_df, on="Object", how="inner")

# === CALCULATE RESIDUALS ===
merged['Residual'] = merged['PWARI_G_Meff'] - merged['log_MBH']

# === SAVE MERGED RESULTS ===
merged.to_csv(output_csv_path, index=False)

# === PLOT COMPARISON ===
plt.figure(figsize=(10, 6))
plt.scatter(merged['log_MBH'], merged['PWARI_G_Meff'], color='darkorange', label='PWARI-G Predicted')
plt.plot([merged['log_MBH'].min(), merged['log_MBH'].max()],
         [merged['log_MBH'].min(), merged['log_MBH'].max()],
         linestyle='--', color='gray', label='Ideal Match (y = x)')

plt.xlabel('Observed log(MBH) [M☉]')
plt.ylabel('PWARI-G Predicted Effective Mass [log M☉]')
plt.title('PWARI-G vs Observed Black Hole Masses')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot_path)
plt.close()

print("✅ Comparison complete!")
print(f"Results saved to: {output_csv_path}")
print(f"Plot saved to: {output_plot_path}")
