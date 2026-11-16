"""Plot diagnostics for the 20-year SSA relaxation (AIS3_relaxed.nc, full continental mesh)."""

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
OUTPUT = MODEL_DIR / "ais3_relaxed_diagnostics.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)

print("-- Loading AIS3_relaxed.nc --", flush=True)
model = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_relaxed.nc"))
ts = model.results.TransientSolution

vel = np.asarray(ts.Vel)          # (nsteps, nv)
thick = np.asarray(ts.Thickness)  # (nsteps, nv)
ocean = np.asarray(ts.MaskOceanLevelset)  # (nsteps, nv)
nsteps = vel.shape[0]
years = model.timestepping.start_time + np.arange(nsteps) * model.timestepping.time_step

ice = np.asarray(model.mask.ice_levelset).ravel()
ice_present = ice < 0

ocean0 = ocean[0]
oceanN = ocean[-1]
grounded0 = ice_present & (ocean0 >= 0)
groundedN = ice_present & (oceanN >= 0)
floating0 = ice_present & (ocean0 < 0)
floatingN = ice_present & (oceanN < 0)

vel0, velN = vel[0], vel[-1]
thick0, thickN = thick[0], thick[-1]
dvel = velN - vel0
dthick = thickN - thick0

vel0_plot = np.where(ice_present, vel0, np.nan)
velN_plot = np.where(ice_present, velN, np.nan)
dvel_plot = np.where(ice_present, dvel, np.nan)
thick0_plot = np.where(ice_present, thick0, np.nan)
thickN_plot = np.where(ice_present, thickN, np.nan)
dthick_plot = np.where(ice_present, dthick, np.nan)

# floating-only thickness change (melt/SMB were both held at zero this pass, so any
# shelf dH is purely dynamic thinning/thickening from the relaxing velocity field)
floating_either = floating0 | floatingN
dthick_floating_plot = np.where(floating_either, dthick, np.nan)

# grounding line migration: cells that flipped grounded<->floating over the 20 years
newly_floating = grounded0 & floatingN
newly_grounded = floating0 & groundedN
migration_plot = np.full(ice.shape, np.nan)
migration_plot[newly_floating] = -1
migration_plot[newly_grounded] = 1

# time series diagnostics (grounded-only, observed-ice mask not needed here)
mean_speed_ts = np.nanmean(np.where(grounded0, vel, np.nan), axis=1)
max_speed_ts = np.nanmax(np.where(ice_present, vel, np.nan), axis=1)
total_thickness_ts = np.nansum(np.where(ice_present, thick, 0.0), axis=1)
volume_anomaly_ts = total_thickness_ts - total_thickness_ts[0]

print(f"nsteps={nsteps}  years 0..{years[-1]:.2f}")
print(f"max|dVel| over relaxation: {np.nanmax(np.abs(dvel[ice_present])):.2f} m/yr")
print(f"mean|dVel| over relaxation: {np.nanmean(np.abs(dvel[ice_present])):.2f} m/yr")
print(f"max|dThickness| over relaxation: {np.nanmax(np.abs(dthick[ice_present])):.2f} m")
print(f"mean|dThickness| over relaxation: {np.nanmean(np.abs(dthick[ice_present])):.2f} m")
print(f"vertices newly floating: {newly_floating.sum()}  newly grounded: {newly_grounded.sum()}")

fig, axes = plt.subplots(3, 3, figsize=(22, 19), constrained_layout=True)

pyissm.plot.plot_model_field(
    model, vel0_plot, show_cbar=True, cmap="PuOr", ax=axes[0, 0], log=True,
    cbar_kwargs={"label": "speed (m/yr, log)"},
)
axes[0, 0].set_title("Speed at t=0 (post-inversion)")

pyissm.plot.plot_model_field(
    model, velN_plot, show_cbar=True, cmap="PuOr", ax=axes[0, 1], log=True,
    cbar_kwargs={"label": "speed (m/yr, log)"},
)
axes[0, 1].set_title(f"Speed at t={years[-1]:.0f}yr; max={np.nanmax(velN_plot):.0f} m/yr")

pyissm.plot.plot_model_field(
    model, dvel_plot, show_cbar=True, cmap="RdBu_r", ax=axes[0, 2],
    vmin=-100, vmax=100, cbar_kwargs={"label": "dVel (m/yr)"},
)
axes[0, 2].set_title(f"Speed change (20yr); mean|dVel|={np.nanmean(np.abs(dvel[ice_present])):.1f}")

pyissm.plot.plot_model_field(
    model, thick0_plot, show_cbar=True, cmap="viridis", ax=axes[1, 0],
    cbar_kwargs={"label": "thickness (m)"},
)
axes[1, 0].set_title("Thickness at t=0")

pyissm.plot.plot_model_field(
    model, thickN_plot, show_cbar=True, cmap="viridis", ax=axes[1, 1],
    cbar_kwargs={"label": "thickness (m)"},
)
axes[1, 1].set_title(f"Thickness at t={years[-1]:.0f}yr")

pyissm.plot.plot_model_field(
    model, dthick_plot, show_cbar=True, cmap="RdBu_r", ax=axes[1, 2],
    vmin=-50, vmax=50, cbar_kwargs={"label": "dThickness (m)"},
)
axes[1, 2].set_title(f"Thickness change (20yr); mean|dH|={np.nanmean(np.abs(dthick[ice_present])):.2f} m")

pyissm.plot.plot_model_field(
    model, dthick_floating_plot, show_cbar=True, cmap="RdBu_r", ax=axes[2, 0],
    vmin=-50, vmax=50, cbar_kwargs={"label": "dThickness (m), floating only"},
)
axes[2, 0].set_title(
    f"Floating ice thickness change\n(zero SMB/melt this pass -- dynamic thinning/thickening only)"
)

axes[2, 1].plot(years, mean_speed_ts, color="tab:blue", label="mean grounded speed")
axes[2, 1].set_xlabel("time (yr)")
axes[2, 1].set_ylabel("mean grounded speed (m/yr)", color="tab:blue")
axes[2, 1].tick_params(axis="y", labelcolor="tab:blue")
ax2b = axes[2, 1].twinx()
ax2b.plot(years, max_speed_ts, color="tab:red", label="max speed (all ice)")
ax2b.set_ylabel("max speed, all ice (m/yr)", color="tab:red")
ax2b.tick_params(axis="y", labelcolor="tab:red")
axes[2, 1].set_title("Speed evolution over relaxation")

axes[2, 2].plot(years, volume_anomaly_ts / 1e9, color="tab:green")
axes[2, 2].set_xlabel("time (yr)")
axes[2, 2].set_ylabel("total ice thickness anomaly (x1e9 m, area-unweighted)")
axes[2, 2].set_title(
    f"Volume-proxy drift over relaxation\n"
    f"newly floating={newly_floating.sum()}  newly grounded={newly_grounded.sum()}"
)
axes[2, 2].axhline(0, color="gray", lw=0.5)

fig.suptitle(
    "AIS3_relaxed.nc -- 20yr SSA relaxation (thermal off, zero SMB/melt placeholder)",
    fontsize=15,
)
plt.savefig(OUTPUT, dpi=100)
print(f"saved {OUTPUT}")
