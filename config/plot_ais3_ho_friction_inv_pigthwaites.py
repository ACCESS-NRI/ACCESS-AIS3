"""Plot diagnostics for the PIG/Thwaites HO friction re-inversion validation test
(test_ho_friction_inv_pigthwaites.nc)."""

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
OUTPUT = MODEL_DIR / "ais3_ho_friction_inv_pigthwaites_diagnostics.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)

print("-- Loading test_ho_friction_inv_pigthwaites.nc --", flush=True)
model = pyissm.model.io.load_model(str(MODEL_DIR / "test_ho_friction_inv_pigthwaites.nc"))

vel = np.asarray(model.results.StressbalanceSolution.Vel).ravel()
vel_obs = np.asarray(model.inversion.vel_obs).ravel()
il = np.asarray(model.mask.ice_levelset).ravel()
ol = np.asarray(model.mask.ocean_levelset).ravel()
on_base = np.asarray(model.mesh.vertexonbase).astype(bool).ravel()
on_surface = np.asarray(model.mesh.vertexonsurface).astype(bool).ravel()
C = np.asarray(model.friction.C).ravel()

grounded_surface = (il < 0) & (ol > 0) & on_surface
residual = vel - vel_obs
rmse = np.sqrt(np.nanmean(residual[grounded_surface] ** 2))

obs_plot = np.where(grounded_surface, vel_obs, np.nan)
mod_plot = np.where(grounded_surface, vel, np.nan)
resid_plot = np.where(grounded_surface, residual, np.nan)
C_plot = np.where((il < 0) & on_base, C, np.nan)

fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)

pyissm.plot.plot_model_field(
    model, obs_plot, show_cbar=True, cmap="PuOr", ax=axes[0, 0], log=True,
    cbar_kwargs={"label": "observed surface speed (m/yr, log)"},
)
axes[0, 0].set_title("Observed surface speed")

pyissm.plot.plot_model_field(
    model, mod_plot, show_cbar=True, cmap="PuOr", ax=axes[0, 1], log=True,
    cbar_kwargs={"label": "modelled surface speed (m/yr, log)"},
)
axes[0, 1].set_title(f"Modelled (HO) surface speed; max={np.nanmax(mod_plot):.0f} m/yr")

pyissm.plot.plot_model_field(
    model, resid_plot, show_cbar=True, cmap="RdBu_r", ax=axes[1, 0],
    vmin=-500, vmax=500, cbar_kwargs={"label": "model - observation (m/yr)"},
)
axes[1, 0].set_title(f"Surface residual; RMSE={rmse:.1f} m/yr")

pyissm.plot.plot_model_field(
    model, C_plot, layer=1, show_cbar=True, cmap="viridis", ax=axes[1, 1],
    cbar_kwargs={"label": "FrictionC (basal)"},
)
axes[1, 1].set_title(f"Inverted basal friction (Schoof C); mean={np.nanmean(C[on_base]):.1f}")

fig.suptitle(
    "test_ho_friction_inv_pigthwaites.nc -- HO friction re-inversion validation "
    "(PIG/Thwaites subdomain)",
    fontsize=14,
)
plt.savefig(OUTPUT, dpi=100)
print(f"saved {OUTPUT}")
