"""Plot diagnostics for the full-continental HO friction re-inversion (AIS3_ho_friction_inv.nc),
current checkpoint (chunk 2, chunked warm-restart -- see docs/post_inversion_pipeline_worklog.md
section 3)."""

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
OUTPUT = MODEL_DIR / "ais3_ho_friction_inv_diagnostics.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)

print("-- Loading AIS3_ho_friction_inv.nc --", flush=True)
model = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_ho_friction_inv.nc"))

vel = np.asarray(model.results.StressbalanceSolution.Vel).ravel()
vel_obs = np.asarray(model.inversion.vel_obs).ravel()
il = np.asarray(model.mask.ice_levelset).ravel()
ol = np.asarray(model.mask.ocean_levelset).ravel()
on_base = np.asarray(model.mesh.vertexonbase).astype(bool).ravel()
on_surface = np.asarray(model.mesh.vertexonsurface).astype(bool).ravel()
C = np.asarray(model.results.StressbalanceSolution.FrictionC).ravel()

grounded_surface = (il < 0) & (ol > 0) & on_surface
residual = vel - vel_obs
rmse = np.sqrt(np.nanmean(residual[grounded_surface] ** 2))

obs_plot = np.where(grounded_surface, vel_obs, np.nan)
mod_plot = np.where(grounded_surface, vel, np.nan)
resid_plot = np.where(grounded_surface, residual, np.nan)
C_plot = np.where((il < 0) & on_base, C, np.nan)

# shared color range for obs/model panels, so the two are directly comparable
vmin_shared = np.nanmin([np.nanmin(obs_plot[grounded_surface]), np.nanmin(mod_plot[grounded_surface])])
vmax_shared = np.nanmax([np.nanmax(obs_plot[grounded_surface]), np.nanmax(mod_plot[grounded_surface])])
print(f"shared speed color range: {vmin_shared:.3f} - {vmax_shared:.1f} m/yr")

fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)

# BUGFIX: pyissm.plot.plot_model_field's log=True path always builds a bare
# matplotlib.colors.LogNorm() internally and separately forwards vmin/vmax via **kwargs to
# ax.tripcolor -- this matplotlib version rejects passing a Normalize instance together with
# vmin/vmax. Bypass the helper for these two panels: replicate its mesh handling, build the
# LogNorm with vmin/vmax baked in ourselves, and call tripcolor directly.
shared_norm = matplotlib.colors.LogNorm(vmin=vmin_shared, vmax=vmax_shared)
mesh, mesh_x, mesh_y, mesh_elements, is3d = pyissm.model.mesh.process_mesh(model)

trip0 = axes[0, 0].tripcolor(mesh, obs_plot[model.mesh.vertexonsurface == 1] if is3d else obs_plot,
                              edgecolors="face", norm=shared_norm, cmap="PuOr", shading="gouraud")
fig.colorbar(trip0, ax=axes[0, 0], label="observed surface speed (m/yr, log)")
axes[0, 0].set_xlabel("X (m)"); axes[0, 0].set_ylabel("Y (m)")
axes[0, 0].set_title("Observed surface speed")

trip1 = axes[0, 1].tripcolor(mesh, mod_plot[model.mesh.vertexonsurface == 1] if is3d else mod_plot,
                              edgecolors="face", norm=shared_norm, cmap="PuOr", shading="gouraud")
fig.colorbar(trip1, ax=axes[0, 1], label="modelled surface speed (m/yr, log)")
axes[0, 1].set_xlabel("X (m)"); axes[0, 1].set_ylabel("Y (m)")
axes[0, 1].set_title(f"Modelled (HO) surface speed; max={np.nanmax(mod_plot):.0f} m/yr")

# Smaller residual range (+-100, down from +-500) to resolve structure in the bulk of the
# domain -- most residuals are near-zero (RMSE=97), a handful of localised hotspots (e.g.
# PIG's trunk, see ais3_ho_friction_inv_pig_zoom.png) will simply saturate at the color limits.
pyissm.plot.plot_model_field(
    model, resid_plot, show_cbar=True, cmap="RdBu_r", ax=axes[1, 0],
    vmin=-100, vmax=100, cbar_kwargs={"label": "model - observation (m/yr)"},
)
axes[1, 0].set_title(f"Surface residual; RMSE={rmse:.1f} m/yr")

pyissm.plot.plot_model_field(
    model, C_plot, layer=1, show_cbar=True, cmap="viridis", ax=axes[1, 1],
    cbar_kwargs={"label": "FrictionC (basal)"},
)
axes[1, 1].set_title(f"Inverted basal friction (Schoof C); mean={np.nanmean(C[(il < 0) & on_base]):.1f}")

fig.suptitle(
    "AIS3_ho_friction_inv.nc -- full-continental HO friction re-inversion, "
    "chunk 2 checkpoint (RMSE={:.2f} m/yr)".format(rmse),
    fontsize=14,
)
plt.savefig(OUTPUT, dpi=100)
print(f"saved {OUTPUT}")
print(f"RMSE (script's own computation, sanity check): {rmse:.4f}")
