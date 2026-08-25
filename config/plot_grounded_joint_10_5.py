"""Plot diagnostics for the completed grounded-only joint 10/5 inversion."""

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
RUN_NAME = "run_001_10_5_1e-08_1e-17"
OUTPUT = MODEL_DIR / "grounded_joint_10_5_diagnostics.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)


print("-- Reconstructing the inversion model --", flush=True)
md = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_param.nc"))
shelf_source = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_param.nc"))
shelf_source.mask.ice_levelset = pyissm.model.param.kill_icebergs(shelf_source)
shelf = shelf_source.extract(
    (shelf_source.mask.ocean_levelset < 0)
    & (shelf_source.mask.ice_levelset < 0)
)

load_cluster = pyissm.model.classes.cluster.gadi()
load_cluster.codepath = ISSM_DIR + "/bin"
load_cluster.executionpath = str(ROOT / "execution_newB_rheology")
load_cluster.login = "jh7060"
load_cluster.project = "au88"
load_cluster.storage = "gdata/au88"
shelf.cluster = load_cluster
shelf.settings.waitonlock = 0
shelf.inversion.iscontrol = 0
shelf.miscellaneous.name = "run_001_1_10_1e-17"
shelf_result = pyissm.model.execute.solve(
    shelf,
    "Stressbalance",
    load_only=True,
    runtime_name=False,
    check_consistency=False,
)
shelf_vertices = np.asarray(shelf_result.mesh.extractedvertices).ravel() - 1
md.materials.rheology_B[shelf_vertices] = np.asarray(
    shelf_result.results.StressbalanceSolution.MaterialsRheologyBbar
).ravel()

md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)
rho_i = md.materials.rho_ice
rho_w = md.materials.rho_water
thickness = np.asarray(md.geometry.thickness).ravel().copy()
bed = np.asarray(md.geometry.bed).ravel()
ocean_full = np.asarray(md.mask.ocean_levelset).ravel()
thickness = np.maximum(thickness, 100.0)
floating_full = ocean_full < 0
base = np.empty_like(thickness)
surface = np.empty_like(thickness)
base[floating_full] = -thickness[floating_full] * rho_i / rho_w
surface[floating_full] = thickness[floating_full] * (1.0 - rho_i / rho_w)
base[~floating_full] = bed[~floating_full]
surface[~floating_full] = bed[~floating_full] + thickness[~floating_full]
md.geometry.thickness = thickness
md.geometry.base = base
md.geometry.surface = surface

effective_pressure = md.friction.effective_pressure.copy()
effective_pressure[effective_pressure < 0] = 0
md.friction.effective_pressure = effective_pressure
md.friction.effective_pressure_limit = 0.07

ice_elements = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset)
mds = md.extract(ice_elements < 1)
boundary = mds.mesh.vertexonboundary.astype(bool)
ocean = np.asarray(mds.mask.ocean_levelset).ravel()
shelf_front = boundary & (ocean < 0)
mds.stressbalance.spcvx[shelf_front] = np.nan
mds.stressbalance.spcvy[shelf_front] = np.nan
mds.stressbalance.spcvz[shelf_front] = np.nan
mds.mask.ice_levelset[shelf_front] = 0

mds.friction = pyissm.model.classes.friction.default(mds.friction)
mds.friction.p = np.full(mds.mesh.numberofelements, 3.0)
mds.friction.q = np.full(mds.mesh.numberofelements, 3.0)
mds.friction.coupling = 3
effective_pressure = mds.friction.effective_pressure.copy()
effective_pressure[effective_pressure < 0] = 0
mds.friction.effective_pressure = effective_pressure
mds.friction.effective_pressure_limit = 0.07

b_prior = np.asarray(mds.materials.rheology_B).ravel().copy()

result_cluster = pyissm.model.classes.cluster.gadi()
result_cluster.codepath = ISSM_DIR + "/bin"
result_cluster.executionpath = str(ROOT / "execution_grounded_suite_joint")
result_cluster.login = "jh7060"
result_cluster.project = "au88"
result_cluster.storage = "gdata/au88"
mds.cluster = result_cluster
mds.settings.waitonlock = 0
mds.inversion.iscontrol = 0
mds.miscellaneous.name = RUN_NAME

print(f"-- Loading {RUN_NAME} --", flush=True)
model = pyissm.model.execute.solve(
    mds,
    "Stressbalance",
    load_only=True,
    runtime_name=False,
    check_consistency=False,
)
solution = model.results.StressbalanceSolution
velocity = np.asarray(solution.Vel).ravel()
friction = np.asarray(solution.FrictionCoefficient).ravel()
b_inverted = np.asarray(solution.MaterialsRheologyBbar).ravel()
velocity_obs = np.asarray(mds.inversion.vel_obs).ravel()
ice = np.asarray(mds.mask.ice_levelset).ravel()
grounded = (ice < 0) & (ocean > 0)
floating = (ice < 0) & (ocean < 0)
observed_grounded = grounded & (velocity_obs > 0)

residual = velocity - velocity_obs
log_ratio = np.full_like(velocity, np.nan, dtype=float)
valid_log = observed_grounded & (velocity > 0)
log_ratio[valid_log] = np.log10(velocity[valid_log] / velocity_obs[valid_log])
b_change = 100.0 * (b_inverted / np.maximum(b_prior, 1e-30) - 1.0)

grounded_rmse = np.sqrt(np.nanmean(residual[observed_grounded] ** 2))
fast_grounded = grounded & (velocity_obs > 100)
fast_rmse = np.sqrt(np.nanmean(residual[fast_grounded] ** 2))
shelf_obs = floating & (velocity_obs > 0)
shelf_rmse = np.sqrt(np.nanmean(residual[shelf_obs] ** 2))

grounded_obs_plot = np.where(observed_grounded, velocity_obs, np.nan)
grounded_model_plot = np.where(observed_grounded, velocity, np.nan)
grounded_residual_plot = np.where(observed_grounded, residual, np.nan)
grounded_friction_plot = np.where(grounded, friction, np.nan)
grounded_b_change_plot = np.where(grounded, b_change, np.nan)

fig, axes = plt.subplots(2, 3, figsize=(22, 13), constrained_layout=True)

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
    log_ratio,
    show_cbar=True,
    cmap="RdBu_r",
    ax=axes[1, 0],
    vmin=-1,
    vmax=1,
    cbar_kwargs={"label": "log10(model/observation)"},
)
axes[1, 0].set_title("Grounded log-speed ratio")

pyissm.plot.plot_model_field(
    model,
    grounded_friction_plot,
    show_cbar=True,
    cmap="viridis",
    ax=axes[1, 1],
    log=True,
    cbar_kwargs={"label": "friction coefficient C (log)"},
)
axes[1, 1].set_title(
    f"Inverted C; median={np.nanmedian(friction[grounded]):.2f}, "
    f"bounds={100*np.mean(friction[grounded] <= 0.1001):.1f}% / "
    f"{100*np.mean(friction[grounded] >= 9.999):.1f}%"
)

pyissm.plot.plot_model_field(
    model,
    grounded_b_change_plot,
    show_cbar=True,
    cmap="RdBu_r",
    ax=axes[1, 2],
    vmin=-50,
    vmax=50,
    cbar_kwargs={"label": "B change from prior (%)"},
)
axes[1, 2].set_title(
    f"Signed B change; median={np.nanmedian(b_change[grounded]):+.1f}%, "
    f">10%={100*np.mean(np.abs(b_change[grounded]) > 10):.1f}%"
)

fig.suptitle(
    "Grounded-only JOINT C+B inversion, cf101=10 / cf103=5\n"
    f"grounded RMSE={grounded_rmse:.1f} m/yr; shelf diagnostic RMSE={shelf_rmse:.1f} m/yr; "
    "stopped on dxmin at iteration 104",
    fontsize=16,
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
    f"{np.nanmedian(friction[grounded]):.4g}/{np.nanmax(friction[grounded]):.4g}; "
    f"B |change|>10%={100*np.mean(np.abs(b_change[grounded]) > 10):.1f}%",
    flush=True,
)
