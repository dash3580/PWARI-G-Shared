
import numpy as np
import plotly.graph_objects as go

# === LOAD FILES FOR STEP 1000 ===
step = 160000
data_dir = "npy_cycle"  # Adjust if your npy files are elsewhere

fields = {
    "phi": np.load(f"{data_dir}/phi_{step}.npy"),
    "theta": np.load(f"{data_dir}/theta_{step}.npy"),
    "twist_energy": np.load(f"{data_dir}/twist_energy_{step}.npy")
}

# === VISUALIZATION ===
def plot_isosurface(volume, threshold, name, color, opacity):
    shape = volume.shape
    x, y, z = np.meshgrid(
        np.linspace(0, shape[0]-1, shape[0]),
        np.linspace(0, shape[1]-1, shape[1]),
        np.linspace(0, shape[2]-1, shape[2]),
        indexing="ij"
    )
    return go.Isosurface(
        value=volume.flatten(),
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        isomin=threshold,
        isomax=volume.max(),
        surface_count=3,
        colorscale=color,
        showscale=False,
        name=name,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=opacity
    )

fig = go.Figure(data=[
    plot_isosurface(fields["phi"], threshold=0.1, name="ϕ Field", color='Blues', opacity=0.5),
    plot_isosurface(fields["theta"], threshold=0.01, name="θ Field", color='Reds', opacity=0.4),
    plot_isosurface(fields["twist_energy"], threshold=1e-11, name="Twist Energy", color='Viridis', opacity=0.3)
])

fig.update_layout(
    scene=dict(aspectmode="cube"),
    title=f"PWARI-G Atom Visualization at Step {step}",
    margin=dict(l=0, r=0, b=0, t=30)
)

fig.show()
