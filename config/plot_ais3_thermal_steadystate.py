"""Plot diagnostics for AIS3_thermal_steadystate.nc (full continental HO velocity +
single-shot thermal solve, non-ice-cleaned temperature)."""

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
OUTPUT = MODEL_DIR / "ais3_thermal_steadystate_diagnostics.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)

print("-- Loading AIS3_thermal_steadystate.nc --", flush=True)
model = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_thermal_steadystate.nc"))

T = np.asarray(model.initialization.temperature).ravel()
Vel = np.asarray(model.initialization.vel).ravel()
ice = np.asarray(model.mask.ice_levelset).ravel()
ice_present = ice < 0

nv2d = model.mesh.numberofvertices2d
T_ice = np.where(ice_present, T, np.nan)
Vel_ice = np.where(ice_present, Vel, np.nan)

print(f"nv total={T.size}  nv2d={nv2d}  numberoflayers={model.mesh.numberoflayers}")
print(f"basal T range (ice only): [{np.nanmin(T_ice[:nv2d]):.2f}, {np.nanmax(T_ice[:nv2d]):.2f}]")

fig, axes = plt.subplots(2, 3, figsize=(22, 13), constrained_layout=True)

pyissm.plot.plot_model_field(
    model, T_ice, layer=1, show_cbar=True, cmap="viridis", ax=axes[0, 0],
    vmin=200, vmax=273.15, cbar_kwargs={"label": "basal temperature (K)"},
)
axes[0, 0].set_title("Basal temperature")

pyissm.plot.plot_model_field(
    model, T_ice, show_cbar=True, cmap="viridis", ax=axes[0, 1],
    vmin=220, vmax=273.15, cbar_kwargs={"label": "surface temperature (K)"},
)
axes[0, 1].set_title("Surface temperature")

dT_base_surf = T_ice[:nv2d] - T_ice[-nv2d:]
pyissm.plot.plot_model_field(
    model, np.where(ice_present[:nv2d], dT_base_surf, np.nan), layer=1,
    show_cbar=True, cmap="RdBu_r", ax=axes[0, 2], vmin=-30, vmax=30,
    cbar_kwargs={"label": "basal - surface T (K)"},
)
axes[0, 2].set_title("Vertical temperature gradient (base - surface)")

pyissm.plot.plot_model_field(
    model, Vel_ice, layer=1, show_cbar=True, cmap="PuOr", ax=axes[1, 0], log=True,
    cbar_kwargs={"label": "basal speed (m/yr, log)"},
)
axes[1, 0].set_title(f"Basal speed; max={np.nanmax(Vel_ice):.0f} m/yr")

axes[1, 1].hist(T_ice[ice_present], bins=100, color="tab:blue")
axes[1, 1].set_xlabel("temperature (K)")
axes[1, 1].set_ylabel("node count")
axes[1, 1].set_title("Temperature distribution (all ice nodes, all layers)")
axes[1, 1].axvline(273.15, color="red", lw=1, label="pressure melting (0 depth)")
axes[1, 1].legend()

near_melting_base = ice_present[:nv2d] & (T_ice[:nv2d] > 272.0)
frac_near_melting = np.sum(near_melting_base) / max(np.sum(ice_present[:nv2d]), 1)
axes[1, 2].axis("off")
summary = (
    f"Full continental HO thermal steady state\n\n"
    f"Total nodes: {T.size:,}\n"
    f"2D columns: {nv2d:,}\n"
    f"Layers: {model.mesh.numberoflayers}\n\n"
    f"Basal T range: [{np.nanmin(T_ice[:nv2d]):.1f}, {np.nanmax(T_ice[:nv2d]):.1f}] K\n"
    f"Mean basal T: {np.nanmean(T_ice[:nv2d]):.1f} K\n"
    f"Fraction of grounded ice near basal melting\n"
    f"(T > 272K): {100*frac_near_melting:.1f}%\n\n"
    f"Max velocity: {np.nanmax(Vel_ice):.0f} m/yr\n\n"
    f"Non-ice nodes overridden to 250K: 1,897,275\n"
    f"Nodes needing 200K floor clip: 1,606,647\n"
    f"Genuine-cold nodes preserved (200-220K): 4,255\n"
    f"(all fixed post-solve, see ais_0.1.py comments)"
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                 fontsize=12, verticalalignment="top", family="monospace")

fig.suptitle(
    "AIS3_thermal_steadystate.nc -- full continental HO velocity + single-shot thermal solve",
    fontsize=15,
)
plt.savefig(OUTPUT, dpi=100)
print(f"saved {OUTPUT}")
