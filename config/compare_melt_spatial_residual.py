"""Spatial cross-check between the two disagreeing melt-gamma_0 calibration results:
the crude per-vertex-RMSE sweep (ais_0.1.py's melt_gamma_tuning, best gamma_0=553.77,
models/AIS3_melt_gamma_tuning.nc) vs. the ISMIP7-informed J1/J2/J3 Monte Carlo calibration
(melt_ismip7_calibration.py, median/95th percentile gamma_0=5537.7, execution/
AIS3_melt_ismip7_g4). The RMSE metric monotonically preferred lower gamma_0 all the way to
the bottom of its tested range -- flagged in docs/post_inversion_pipeline_worklog.md as a
possible sign the metric is being gamed by turning melt off almost everywhere rather than
reflecting a real optimum. This script checks that directly: compares each candidate's
simulated melt field against the ITS_LIVE melt_mean observational target at every floating
mesh vertex, computes RMSE (as a consistency check against the documented values), the
fraction of floating area with near-zero simulated melt, and produces a spatial residual
map for visual inspection.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm
import ccdtools

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'


def cluster_config():
    cl = pyissm.model.classes.cluster.gadi()
    cl.codepath = os.environ['ISSM_DIR'] + '/bin'
    cl.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
    cl.storage = 'gdata/au88+gdata/vk83+gdata/av17'
    cl.moduleuse = ['/g/data/vk83/modules/']
    cl.moduleload = ['access-issm_ad/2026.05.0']
    cl.login = 'jh7060'
    cl.project = 'au88'
    return cl


print('-- Loading RMSE-best model (gamma_0=553.77, AIS3_melt_gamma_tuning.nc)...', flush=True)
md_rmse = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_melt_gamma_tuning.nc')
gamma_rmse = float(md_rmse.basalforcings.gamma_0)
print(f'   confirmed gamma_0 = {gamma_rmse:.2f}')

print('-- Reloading ISMIP7 ensemble member g4 (gamma_0=5537.7, Zhou tf)...', flush=True)
md_ismip7 = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
md_ismip7.miscellaneous.name = 'AIS3_melt_ismip7_g4'
md_ismip7.cluster = cluster_config()
md_ismip7 = pyissm.model.execute.solve(md_ismip7, 'Transient', load_only=True, runtime_name=False)

# Both models share the identical mesh (both derive from AIS3_ho_friction_inv.nc without
# re-meshing), so 2D coordinates/connectivity/on_base indexing are directly comparable.
on_base = np.asarray(md_rmse.mesh.vertexonbase).astype(bool).ravel()
x2d = np.asarray(md_rmse.mesh.x2d).ravel()
y2d = np.asarray(md_rmse.mesh.y2d).ravel()
ol2d = np.asarray(md_rmse.mask.ocean_levelset).ravel()[on_base]
floating = ol2d < 0
print(f'   {floating.sum()} floating 2D vertices out of {on_base.sum()} total base vertices')

melt_rmse = np.asarray(md_rmse.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()[on_base]
melt_ismip7 = np.asarray(md_ismip7.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()[on_base]

print('-- Loading ITS_LIVE observed melt_mean and interpolating onto the 2D mesh...', flush=True)
catalog = ccdtools.catalog.DataCatalog()
melt_obs_ds = catalog.load_dataset('measures_its_live_antarctic_quarterly_ice_shelf_height_change')
melt_obs_2d = pyissm.data.interp.xr_to_mesh(melt_obs_ds, 'melt_mean', x2d, y2d)

# --- Headline stats ---
def report(label, sim):
    resid = sim[floating] - melt_obs_2d[floating]
    rmse = np.sqrt(np.nanmean(resid ** 2))
    bias = np.nanmean(resid)
    near_zero_frac = np.mean(np.abs(sim[floating]) < 0.1)  # < 0.1 m/yr treated as ~off
    print(f'{label}: RMSE={rmse:.2f} m/yr  bias={bias:+.2f} m/yr  '
          f'frac(|melt|<0.1 m/yr)={near_zero_frac:.1%}  '
          f'mean_sim={np.nanmean(sim[floating]):.2f}  mean_obs={np.nanmean(melt_obs_2d[floating]):.2f}')
    return resid, rmse, near_zero_frac

print('\n-- Spatial comparison --', flush=True)
resid_rmse, rmse_val_rmse, nz_rmse = report(f'gamma_0={gamma_rmse:.1f} (RMSE-best)', melt_rmse)
resid_ismip7, rmse_val_ismip7, nz_ismip7 = report('gamma_0=5537.7 (ISMIP7-best)', melt_ismip7)

# --- Plot: obs, both sims, both residuals ---
print('\n-- Plotting spatial residual maps...', flush=True)
fx, fy = x2d[floating], y2d[floating]
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

def scatter_panel(ax, vals, title, vmin, vmax, cmap='viridis'):
    sc = ax.scatter(fx, fy, c=vals, s=1, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.04)

obs_f = melt_obs_2d[floating]
vmax_melt = np.nanpercentile(np.abs(np.concatenate([obs_f, melt_rmse[floating], melt_ismip7[floating]])), 98)
scatter_panel(axes[0, 0], obs_f, 'ITS_LIVE observed melt_mean (m/yr)', 0, vmax_melt)
scatter_panel(axes[0, 1], melt_rmse[floating], f'Sim melt, gamma_0={gamma_rmse:.1f} (RMSE-best)', 0, vmax_melt)
scatter_panel(axes[0, 2], melt_ismip7[floating], 'Sim melt, gamma_0=5537.7 (ISMIP7-best)', 0, vmax_melt)

vmax_resid = np.nanpercentile(np.abs(np.concatenate([resid_rmse, resid_ismip7])), 98)
axes[1, 0].axis('off')
scatter_panel(axes[1, 1], resid_rmse, f'Residual (sim-obs), gamma_0={gamma_rmse:.1f}\n'
              f'RMSE={rmse_val_rmse:.2f}, near-zero frac={nz_rmse:.1%}', -vmax_resid, vmax_resid, cmap='RdBu_r')
scatter_panel(axes[1, 2], resid_ismip7, f'Residual (sim-obs), gamma_0=5537.7\n'
              f'RMSE={rmse_val_ismip7:.2f}, near-zero frac={nz_ismip7:.1%}', -vmax_resid, vmax_resid, cmap='RdBu_r')

plt.tight_layout()
out_png = f'{MODEL_DIR}/melt_spatial_residual_comparison.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
print('DONE')
