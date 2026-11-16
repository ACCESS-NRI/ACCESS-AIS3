"""Plot diagnostics for the finalized AIS3_inverted.nc (Schoof, full continental mesh)."""

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
OUTPUT = MODEL_DIR / "ais3_inverted_schoof_diagnostics.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)

print("-- Loading AIS3_inverted.nc --", flush=True)
model = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_inverted.nc"))
solution = model.results.StressbalanceSolution
velocity = np.asarray(solution.Vel).ravel()
friction = np.asarray(model.friction.C).ravel()
velocity_obs = np.asarray(model.inversion.vel_obs).ravel()
ice = np.asarray(model.mask.ice_levelset).ravel()
ocean = np.asarray(model.mask.ocean_levelset).ravel()
spcvx = np.asarray(model.stressbalance.spcvx).ravel()
pinned = np.isfinite(spcvx)
grounded = (ice < 0) & (ocean >= 0)
floating = (ice < 0) & (ocean < 0)
observed_grounded = grounded & (velocity_obs > 0)

residual = velocity - velocity_obs
log_ratio = np.full_like(velocity, np.nan, dtype=float)
valid_log = observed_grounded & (velocity > 0)
log_ratio[valid_log] = np.log10(velocity[valid_log] / velocity_obs[valid_log])

grounded_rmse = np.sqrt(np.nanmean(residual[observed_grounded] ** 2))
fast_grounded = grounded & (velocity_obs > 100)
fast_rmse = np.sqrt(np.nanmean(residual[fast_grounded] ** 2))
shelf_obs = floating & (velocity_obs > 0)
shelf_rmse = np.sqrt(np.nanmean(residual[shelf_obs] ** 2))

grounded_obs_plot = np.where(observed_grounded, velocity_obs, np.nan)
grounded_model_plot = np.where(observed_grounded, velocity, np.nan)
grounded_residual_plot = np.where(observed_grounded, residual, np.nan)
grounded_friction_plot = np.where(grounded, friction, np.nan)
pinned_plot = np.where(grounded | floating, pinned.astype(float), np.nan)

floating_obs_plot = np.where(shelf_obs, velocity_obs, np.nan)
floating_model_plot = np.where(shelf_obs, velocity, np.nan)
floating_residual_plot = np.where(shelf_obs, residual, np.nan)
floating_max = np.nanmax(velocity[floating]) if floating.any() else float("nan")

fig, axes = plt.subplots(3, 3, figsize=(22, 19), constrained_layout=True)

pyissm.plot.plot_model_field(
    model,
    grounded_obs_plot,
    show_cbar=True,
    cmap="PuOr",
    ax=axes[0, 0],
    log=True,
    cbar_kwargs={"label": "observed speed (m/yr, log)"},
)
axes[0, 0].set_title("Observed grounded speed")

pyissm.plot.plot_model_field(
    model,
    grounded_model_plot,
    show_cbar=True,
    cmap="PuOr",
    ax=axes[0, 1],
    log=True,
    cbar_kwargs={"label": "modelled speed (m/yr, log)"},
)
axes[0, 1].set_title(f"Modelled grounded speed; max={np.nanmax(velocity):.0f} m/yr")

pyissm.plot.plot_model_field(
    model,
    grounded_residual_plot,
    show_cbar=True,
    cmap="RdBu_r",
    ax=axes[0, 2],
    vmin=-100,
    vmax=100,
    cbar_kwargs={"label": "model - observation (m/yr)"},
)
axes[0, 2].set_title(
    f"Grounded residual; RMSE={grounded_rmse:.1f}, fast RMSE={fast_rmse:.1f}"
)

pyissm.plot.plot_model_field(
    model,
    floating_obs_plot,
    show_cbar=True,
    cmap="PuOr",
    ax=axes[1, 0],
    log=True,
    cbar_kwargs={"label": "observed speed (m/yr, log)"},
)
axes[1, 0].set_title("Observed floating (shelf) speed")

pyissm.plot.plot_model_field(
    model,
    floating_model_plot,
    show_cbar=True,
    cmap="PuOr",
    ax=axes[1, 1],
    log=True,
    cbar_kwargs={"label": "modelled speed (m/yr, log)"},
)
axes[1, 1].set_title(f"Modelled floating speed; max={floating_max:.0f} m/yr")

pyissm.plot.plot_model_field(
    model,
    floating_residual_plot,
    show_cbar=True,
    cmap="RdBu_r",
    ax=axes[1, 2],
    vmin=-200,
    vmax=200,
    cbar_kwargs={"label": "model - observation (m/yr)"},
)
axes[1, 2].set_title(f"Floating (shelf) residual; RMSE={shelf_rmse:.1f}")

pyissm.plot.plot_model_field(
    model,
    log_ratio,
    show_cbar=True,
    cmap="RdBu_r",
    ax=axes[2, 0],
    vmin=-1,
    vmax=1,
    cbar_kwargs={"label": "log10(model/observation)"},
)
axes[2, 0].set_title("Grounded log-speed ratio")

pyissm.plot.plot_model_field(
    model,
    grounded_friction_plot,
    show_cbar=True,
    cmap="viridis",
    ax=axes[2, 1],
    log=True,
    cbar_kwargs={"label": "Schoof FrictionC (log)"},
)
axes[2, 1].set_title(
    f"Final C; median={np.nanmedian(friction[grounded]):.1f}, "
    f"bounds={100*np.mean(friction[grounded] <= 50.01):.1f}% / "
    f"{100*np.mean(friction[grounded] >= 7999.99):.1f}%"
)

pyissm.plot.plot_model_field(
    model,
    pinned_plot,
    show_cbar=True,
    cmap="RdBu_r",
    ax=axes[2, 2],
    vmin=0,
    vmax=1,
    cbar_kwargs={"label": "1 = Dirichlet-pinned to observation"},
)
axes[2, 2].set_title(
    f"Pinned vertices (3 waves: thin-ice / lower-bound-C / non-ice); n={int(pinned.sum())}"
)

fig.suptitle(
    "AIS3_inverted.nc -- Schoof control inversion (FrictionC), Cmax=2.0, full continental mesh\n"
    f"grounded RMSE={grounded_rmse:.1f} m/yr; shelf diagnostic RMSE={shelf_rmse:.1f} m/yr; "
    f"max modelled velocity={np.nanmax(velocity):.0f} m/yr",
    fontsize=15,
)
fig.savefig(OUTPUT, dpi=120)
plt.close(fig)

grounded_residual = residual[observed_grounded]
print(f"SAVED {OUTPUT}", flush=True)
print(
    f"grounded RMSE={grounded_rmse:.2f}; fast grounded RMSE={fast_rmse:.2f}; "
    f"shelf RMSE={shelf_rmse:.2f}",
    flush=True,
)
print(
    f"residual median={np.nanmedian(grounded_residual):+.2f}; "
    f"mean={np.nanmean(grounded_residual):+.2f}; "
    f"|residual|<50={100*np.mean(np.abs(grounded_residual) < 50):.1f}%",
    flush=True,
)
print(
    f"C grounded min/median/max={np.nanmin(friction[grounded]):.4g}/"
    f"{np.nanmedian(friction[grounded]):.4g}/{np.nanmax(friction[grounded]):.4g}",
    flush=True,
)
