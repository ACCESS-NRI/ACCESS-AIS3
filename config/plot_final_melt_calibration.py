"""Spatial plot of the final melt calibration (gamma_0=300, per-basin-refit deltaT_basin,
models/AIS3_melt_gamma_tuning.nc) -- sim melt, ITS_LIVE observed melt_mean for reference, and
their residual, mirroring compare_melt_spatial_residual.py's panel layout.
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

print('-- Loading final melt calibration (gamma_0=300, refit deltaT_basin)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_melt_gamma_tuning.nc')

on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
x2d = np.asarray(md.mesh.x2d).ravel()
y2d = np.asarray(md.mesh.y2d).ravel()
ol2d = np.asarray(md.mask.ocean_levelset).ravel()[on_base]
floating = ol2d < 0
print(f'   {floating.sum()} floating 2D vertices', flush=True)

melt = np.asarray(md.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()[on_base]

print('-- Loading ITS_LIVE observed melt_mean for reference...', flush=True)
catalog = ccdtools.catalog.DataCatalog()
melt_obs_ds = catalog.load_dataset('measures_its_live_antarctic_quarterly_ice_shelf_height_change')
melt_obs_2d = pyissm.data.interp.xr_to_mesh(melt_obs_ds, 'melt_mean', x2d, y2d)

resid = melt[floating] - melt_obs_2d[floating]
rmse = np.sqrt(np.nanmean(resid ** 2))
bias = np.nanmean(resid)
print(f'   RMSE vs ITS_LIVE = {rmse:.2f} m/yr, bias = {bias:+.2f} m/yr '
      f'(mean_sim={np.nanmean(melt[floating]):.2f}, mean_obs={np.nanmean(melt_obs_2d[floating]):.2f})')

print('-- Plotting...', flush=True)
fx, fy = x2d[floating], y2d[floating]
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))


def scatter_panel(ax, vals, title, vmin, vmax, cmap='viridis'):
    sc = ax.scatter(fx, fy, c=vals, s=1, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.04)


obs_f = melt_obs_2d[floating]
sim_f = melt[floating]
vmax_melt = np.nanpercentile(np.abs(np.concatenate([obs_f, sim_f])), 98)
scatter_panel(axes[0], obs_f, 'ITS_LIVE observed melt_mean (m/yr)', 0, vmax_melt)
scatter_panel(axes[1], sim_f, 'Final calibration: gamma_0=300, refit deltaT_basin (m/yr)', 0, vmax_melt)
vmax_resid = np.nanpercentile(np.abs(resid), 98)
scatter_panel(axes[2], resid, f'Residual (sim-obs)\nRMSE={rmse:.2f} m/yr, bias={bias:+.2f} m/yr',
              -vmax_resid, vmax_resid, cmap='RdBu_r')

plt.tight_layout()
out_png = f'{MODEL_DIR}/melt_final_calibration_spatial.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
print('DONE')
