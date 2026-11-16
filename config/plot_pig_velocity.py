"""Plot modelled vs. observed surface speed (and residual) zoomed to the PIG/Thwaites
drainage basin, for the current AIS3_ho_friction_inv.nc (chunk-2) checkpoint."""
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
OUTPUT = MODEL_DIR / "ais3_ho_friction_inv_pig_zoom.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)

print("-- Loading AIS3_ho_friction_inv.nc --", flush=True)
model = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_ho_friction_inv.nc"))

vel = np.asarray(model.results.StressbalanceSolution.Vel).ravel()
vel_obs = np.asarray(model.inversion.vel_obs).ravel()
il = np.asarray(model.mask.ice_levelset).ravel()
ol = np.asarray(model.mask.ocean_levelset).ravel()
on_surface = np.asarray(model.mesh.vertexonsurface).astype(bool).ravel()
x = np.asarray(model.mesh.x).ravel()
y = np.asarray(model.mesh.y).ravel()

print("-- Flagging PIG/Thwaites elements via PIG_Outline.exp --", flush=True)
flag_elem = pyissm.model.mesh.flag_elements(model, "assets/PIG_Outline.exp")
elx = np.asarray(model.mesh.elements).astype(int) - 1
pig_vertex = np.zeros(model.mesh.numberofvertices, dtype=bool)
pig_vertex[elx[flag_elem].ravel()] = True

grounded_surface = (il < 0) & (ol > 0) & on_surface
mask = grounded_surface & pig_vertex

residual = vel - vel_obs
rmse_pig = np.sqrt(np.nanmean(residual[mask] ** 2))

# zoom box: PIG vertex extent + 15% padding
xs, ys = x[pig_vertex], y[pig_vertex]
xpad = 0.15 * (xs.max() - xs.min())
ypad = 0.15 * (ys.max() - ys.min())
xlim = (xs.min() - xpad, xs.max() + xpad)
ylim = (ys.min() - ypad, ys.max() + ypad)

obs_plot = np.where(grounded_surface, vel_obs, np.nan)
mod_plot = np.where(grounded_surface, vel, np.nan)
resid_plot = np.where(grounded_surface, residual, np.nan)

# shared color range for obs/model panels (PIG-masked), so the two are directly comparable
vmin_shared = np.nanmin([np.nanmin(obs_plot[mask]), np.nanmin(mod_plot[mask])])
vmax_shared = np.nanmax([np.nanmax(obs_plot[mask]), np.nanmax(mod_plot[mask])])
print(f"shared speed color range (PIG-masked): {vmin_shared:.3f} - {vmax_shared:.1f} m/yr")

fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

# BUGFIX: pyissm.plot.plot_model_field's log=True path always builds a bare
# matplotlib.colors.LogNorm() internally and separately forwards vmin/vmax via **kwargs to
# ax.tripcolor -- this matplotlib version rejects passing a Normalize instance together with
# vmin/vmax ("Passing a Normalize instance simultaneously with vmin/vmax is not supported").
# Bypass the helper for these two panels: replicate its mesh handling, build the LogNorm with
# vmin/vmax baked in ourselves, and call tripcolor directly.
shared_norm = matplotlib.colors.LogNorm(vmin=vmin_shared, vmax=vmax_shared)
mesh, mesh_x, mesh_y, mesh_elements, is3d = pyissm.model.mesh.process_mesh(model)

trip0 = axes[0].tripcolor(mesh, obs_plot[model.mesh.vertexonsurface == 1] if is3d else obs_plot,
                           edgecolors="face", norm=shared_norm, cmap="PuOr", shading="gouraud")
fig.colorbar(trip0, ax=axes[0], label="observed surface speed (m/yr, log)")
axes[0].set_xlim(xlim); axes[0].set_ylim(ylim)
axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
axes[0].set_title("Observed surface speed (PIG/Thwaites)")

trip1 = axes[1].tripcolor(mesh, mod_plot[model.mesh.vertexonsurface == 1] if is3d else mod_plot,
                           edgecolors="face", norm=shared_norm, cmap="PuOr", shading="gouraud")
fig.colorbar(trip1, ax=axes[1], label="modelled surface speed (m/yr, log)")
axes[1].set_xlim(xlim); axes[1].set_ylim(ylim)
axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")
axes[1].set_title(f"Modelled (HO) surface speed (PIG/Thwaites); max={np.nanmax(mod_plot[mask]):.0f} m/yr")

pyissm.plot.plot_model_field(
    model, resid_plot, show_cbar=True, cmap="RdBu_r", ax=axes[2],
    vmin=-500, vmax=500, cbar_kwargs={"label": "model - observation (m/yr)"},
)
axes[2].set_xlim(xlim); axes[2].set_ylim(ylim)
axes[2].set_title(f"Residual (PIG/Thwaites); RMSE={rmse_pig:.1f} m/yr")

fig.suptitle(
    "AIS3_ho_friction_inv.nc -- PIG/Thwaites zoom, chunk 2 checkpoint",
    fontsize=14,
)
plt.savefig(OUTPUT, dpi=110)
print(f"saved {OUTPUT}")
print(f"PIG RMSE: {rmse_pig:.4f}")
print("DONE")
