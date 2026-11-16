"""Plot diagnostics for the tight-restol (0.001) Schoof m1qn3 control inversion, Cmax=2.0."""

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
OUTPUT = MODEL_DIR / "schoof_tightrestol_diagnostics.png"

os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)


def load(cluster_execpath, run_name, md_template):
    m = md_template
    m.cluster = pyissm.model.classes.cluster.gadi()
    m.cluster.codepath = ISSM_DIR + "/bin"
    m.cluster.executionpath = cluster_execpath
    m.cluster.login = "jh7060"
    m.cluster.project = "au88"
    m.cluster.storage = "gdata/au88"
    m.miscellaneous.name = run_name
    return pyissm.model.execute.solve(
        m, "Stressbalance", load_only=True, runtime_name=False, check_consistency=False
    )


print("-- Building base mesh --", flush=True)
md = pyissm.model.io.load_model(str(MODEL_DIR / "AIS3_param.nc"))
md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)
ice_elements = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset)
mds = md.extract(ice_elements < 1)
boundary = mds.mesh.vertexonboundary.astype(bool)
ocean = np.asarray(mds.mask.ocean_levelset).ravel()
shelf_front = boundary & (ocean < 0)
mds.stressbalance.spcvx[shelf_front] = np.nan
mds.stressbalance.spcvy[shelf_front] = np.nan
mds.stressbalance.spcvz[shelf_front] = np.nan
mds.mask.ice_levelset[shelf_front] = 0
mds.settings.waitonlock = 0
mds.inversion.iscontrol = 0

print("-- Loading warm-start (continuation step1) --", flush=True)
warm_model = load(
    str(ROOT / "execution_schoof_continuation_step1"), "schoof_cont_step1_cmax2.0", mds
)
C_warmstart = np.asarray(warm_model.results.StressbalanceSolution.FrictionC).ravel().copy()

print("-- Loading final (tightrestol) result --", flush=True)
model = load(
    str(ROOT / "execution_schoof_m1qn3_tightrestol_test"),
    "schoof_m1qn3_tightrestol_cmax2.0",
    mds,
)
solution = model.results.StressbalanceSolution
velocity = np.asarray(solution.Vel).ravel()
friction = np.asarray(solution.FrictionC).ravel()
velocity_obs = np.asarray(mds.inversion.vel_obs).ravel()
ice = np.asarray(mds.mask.ice_levelset).ravel()
grounded = (ice < 0) & (ocean > 0)
floating = (ice < 0) & (ocean < 0)
observed_grounded = grounded & (velocity_obs > 0)

residual = velocity - velocity_obs
log_ratio = np.full_like(velocity, np.nan, dtype=float)
valid_log = observed_grounded & (velocity > 0)
log_ratio[valid_log] = np.log10(velocity[valid_log] / velocity_obs[valid_log])
c_change = friction - C_warmstart

grounded_rmse = np.sqrt(np.nanmean(residual[observed_grounded] ** 2))
fast_grounded = grounded & (velocity_obs > 100)
fast_rmse = np.sqrt(np.nanmean(residual[fast_grounded] ** 2))
shelf_obs = floating & (velocity_obs > 0)
shelf_rmse = np.sqrt(np.nanmean(residual[shelf_obs] ** 2))

grounded_obs_plot = np.where(observed_grounded, velocity_obs, np.nan)
grounded_model_plot = np.where(observed_grounded, velocity, np.nan)
grounded_residual_plot = np.where(observed_grounded, residual, np.nan)
grounded_friction_plot = np.where(grounded, friction, np.nan)
grounded_cchange_plot = np.where(grounded, c_change, np.nan)

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
    cbar_kwargs={"label": "Schoof FrictionC (log)"},
)
axes[1, 1].set_title(
    f"Final C; median={np.nanmedian(friction[grounded]):.1f}, "
    f"bounds={100*np.mean(friction[grounded] <= 50.01):.1f}% / "
    f"{100*np.mean(friction[grounded] >= 7999.99):.1f}%"
)

pyissm.plot.plot_model_field(
    model,
    grounded_cchange_plot,
    show_cbar=True,
    cmap="RdBu_r",
    ax=axes[1, 2],
    vmin=-500,
    vmax=500,
    cbar_kwargs={"label": "C final - C warmstart"},
)
axes[1, 2].set_title(
    f"C movement from warm-start; median|change|={np.nanmedian(np.abs(c_change[grounded])):.1f}, "
    f">10={100*np.mean(np.abs(c_change[grounded]) > 10):.1f}%"
)

fig.suptitle(
    "Schoof control inversion (FrictionC), Cmax=2.0, restol=0.001, m1qn3\n"
    f"grounded RMSE={grounded_rmse:.1f} m/yr (warmstart was 188.6); shelf diagnostic RMSE={shelf_rmse:.1f} m/yr; "
    "stopped: max simulator calls reached (nsim=170), still improving",
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
