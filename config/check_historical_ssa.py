"""Sanity check of AIS3_historical_1995_2019_SSA.nc (the completed, single-shot 1995-2019
SSA historical run): NaN/Inf/extreme values, dH distribution over the full 24 years,
grounding-line migration activity (if MaskOceanLevelset was saved), and a spatial dH plot.
Mirrors check_historical_chunk1.py's structure/conventions, adapted for SSA's 2D mesh (no
vertical layers, so no nv3d/on_base reshaping needed) and single-shot (not chunked) run.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'

print('-- Loading AIS3_historical_1995_2019_SSA.nc...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')

start_t = float(md.timestepping.start_time)
final_t = float(md.timestepping.final_time)
years_elapsed = final_t - start_t
print(f'   Run spans t={start_t} -> t={final_t} ({years_elapsed:.1f} years)')

ts = md.results.TransientSolution
nv = md.mesh.numberofvertices
if isinstance(ts, list):
    print(f'   {len(ts)} timesteps saved')
    first, last = ts[0], ts[-1]
else:
    raw = np.asarray(ts.Thickness).ravel()
    n_steps = raw.size // nv
    print(f'   collapsed object holds {n_steps} timesteps flattened (size={raw.size}, nv={nv})')
    class _Step:
        def __init__(self, idx):
            self.idx = idx
        def __getattr__(self, name):
            arr = np.asarray(getattr(ts, name)).ravel()
            return arr.reshape(n_steps, nv)[self.idx]
    first, last = _Step(0), _Step(-1)

H0 = np.asarray(first.Thickness).ravel()
H1 = np.asarray(last.Thickness).ravel()

print('\n=== Final-timestep field sanity ===')
fields_to_check = ['Thickness']
for fname in ['Vel', 'Surface', 'Base']:
    try:
        arr = np.asarray(getattr(last, fname)).ravel()
        fields_to_check.append(fname)
    except AttributeError:
        print(f'   {fname}: NOT SAVED (requested_outputs did not include it) -- skipping')

for name in fields_to_check:
    arr = H1 if name == 'Thickness' else np.asarray(getattr(last, name)).ravel()
    print(f'   {name}: NaN={np.isnan(arr).sum()} Inf={np.isinf(arr).sum()} '
          f'min={np.nanmin(arr):.3g} max={np.nanmax(arr):.3g} mean={np.nanmean(arr):.3g}')

neg_thickness = (H1 < 0).sum()
print(f'\n   Negative thickness count: {neg_thickness}')

il = np.asarray(md.mask.ice_levelset).ravel()
ol_final = np.asarray(md.mask.ocean_levelset).ravel()  # current (final) state on the model
try:
    ol0 = np.asarray(first.MaskOceanLevelset).ravel()
    ol1 = np.asarray(last.MaskOceanLevelset).ravel()
    grounded0 = ol0 >= 0
    grounded1 = ol1 >= 0
    newly_floating = grounded0 & ~grounded1
    newly_grounded = ~grounded0 & grounded1
    print(f'\n=== Grounding-line migration over the full {years_elapsed:.0f} years ===')
    print(f'   Newly floating (grounded->floating): {int(newly_floating.sum())} vertices')
    print(f'   Newly grounded (floating->grounded): {int(newly_grounded.sum())} vertices')
except AttributeError:
    print('\n=== Grounding-line migration: MaskOceanLevelset not saved per-timestep, skipping ===')
    ol1 = ol_final

dH = H1 - H0
dhdt_sim = dH / years_elapsed
print(f'\n=== dH distribution over the full {years_elapsed:.0f}-year run ===')
print(f'   NaN={np.isnan(dH).sum()}')
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f'   p{p:5.1f}: {np.nanpercentile(dH, p):.2f} m')

print('\n-- Recomputing dH/dt vs MIPKIT dhdt_cpom for reference (production script already '
      'reported RMSE=1.402 m/yr)...', flush=True)
mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
dhdt_obs_grid = mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=-1)
dhdt_obs_mesh = pyissm.data.interp.xr_to_mesh(
    dhdt_obs_grid, 'dhdt_cpom', md.mesh.x, md.mesh.y, x_var='x1km', y_var='y1km')
gr = (il < 0) & (ol_final > 0)
mismatch_rmse = np.sqrt(np.nanmean((dhdt_sim[gr] - dhdt_obs_mesh[gr]) ** 2))
bias = np.nanmean(dhdt_sim[gr] - dhdt_obs_mesh[gr])
print(f'   Grounded dH/dt mismatch RMSE = {mismatch_rmse:.3f} m/yr, bias (sim-obs) = {bias:.3f} m/yr')

print('\n-- Plotting spatial dH...', flush=True)
mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()
fig, ax = plt.subplots(figsize=(9, 9))
vmax = np.nanpercentile(np.abs(dH), 98)
sc = ax.scatter(mesh_x, mesh_y, c=dH, s=1, cmap='RdBu_r', vmin=-vmax, vmax=vmax, rasterized=True)
ax.set_title(f'AIS3_historical_1995_2019_SSA: dH over t={start_t:.0f}-{final_t:.0f} (m)\n'
             f'mean|dH|={np.nanmean(np.abs(dH)):.2f}m, RMSE vs dhdt_cpom={mismatch_rmse:.3f} m/yr')
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(sc, ax=ax, fraction=0.04)
plt.tight_layout()
out_png = f'{MODEL_DIR}/historical_ssa_dH_spatial.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
print('DONE')
