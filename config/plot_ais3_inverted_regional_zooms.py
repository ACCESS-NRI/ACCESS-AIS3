"""Regional zoom-in diagnostics for the finalized AIS3_inverted.nc (Schoof, full mesh)."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyissm


ROOT = Path("/g/data/au88/jh7060/ACCESS-AIS3")
MODEL_DIR = ROOT / "models"
ISSM_DIR = (
    "/g/data/vk83/apps/spack/1.1/release/linux-x86_64/"
    "issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd"
)
OUTPUT = MODEL_DIR / "ais3_inverted_regional_zooms.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)

print("-- Loading AIS3_inverted.nc --", flush=True)
model = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_inverted.nc"))
solution = model.results.StressbalanceSolution
velocity = np.asarray(solution.Vel).ravel()
velocity_obs = np.asarray(model.inversion.vel_obs).ravel()
ice = np.asarray(model.mask.ice_levelset).ravel()

observed_ice = (ice < 0) & (velocity_obs > 0)
residual = velocity - velocity_obs

obs_plot = np.where(observed_ice, velocity_obs, np.nan)
model_plot = np.where(observed_ice, velocity, np.nan)
residual_plot = np.where(observed_ice, residual, np.nan)

# (name, xmin, xmax, ymin, ymax)
regions = [
    ("Pine Island / Thwaites (Amundsen Sea)", -1850000, -1250000, -500000, 100000),
    ("Ross Ice Shelf / Siple Coast", -700000, 500000, -1600000, -600000),
    ("Filchner-Ronne Ice Shelf", -1500000, 100000, 100000, 1500000),
    ("Totten Glacier (East Antarctica)", 1900000, 2500000, -1300000, -700000),
    ("Institute / Moller Ice Stream", -1000000, -200000, -800000, 0),
]

fig, axes = plt.subplots(len(regions), 3, figsize=(18, 6 * len(regions)), constrained_layout=True)

for row, (name, xmin, xmax, ymin, ymax) in enumerate(regions):
    for col, (field, cmap, label, kwargs) in enumerate([
        (obs_plot, "PuOr", "observed speed (m/yr, log)", {"log": True}),
        (model_plot, "PuOr", "modelled speed (m/yr, log)", {"log": True}),
        (residual_plot, "RdBu_r", "model - observation (m/yr)", {"vmin": -100, "vmax": 100}),
    ]):
        ax = axes[row, col]
        pyissm.plot.plot_model_field(
            model,
            field,
            show_cbar=True,
            cmap=cmap,
            ax=ax,
            cbar_kwargs={"label": label},
            **kwargs,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        if col == 0:
            ax.set_title(f"{name}\nObserved speed")
        elif col == 1:
            ax.set_title("Modelled speed")
        else:
            mx = np.asarray(model.mesh.x).ravel()
            my = np.asarray(model.mesh.y).ravel()
            sub = (mx >= xmin) & (mx <= xmax) & (my >= ymin) & (my <= ymax) & observed_ice
            rmse = np.sqrt(np.nanmean(residual[sub] ** 2)) if sub.any() else float("nan")
            ax.set_title(f"Residual; RMSE={rmse:.1f} m/yr")

fig.suptitle(
    "AIS3_inverted.nc -- Schoof control inversion, regional zooms",
    fontsize=16,
)
fig.savefig(OUTPUT, dpi=110)
plt.close(fig)
print(f"SAVED {OUTPUT}", flush=True)
