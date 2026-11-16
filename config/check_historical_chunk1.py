"""Sanity check of AIS3_historical_1995_2019.nc after chunk 1 (t=1995->2001, 6 years) of the
chunked warm-restart historical run: NaN/Inf/extreme values, dH/dt over the completed span vs
MIPKIT's dhdt_cpom (an EARLY signal only -- the official comparison is against the full
1995-2019 span, this is 6 of 24 years), grounding-line migration activity, and a spatial dH
plot. Mirrors check_ho_relaxed.py's structure/conventions.
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

print('-- Loading AIS3_historical_1995_2019.nc (checkpoint after chunk 1)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019.nc')

start_t = float(md.timestepping.start_time)
final_t = float(md.timestepping.final_time)
print(f'   Checkpoint spans t={start_t} -> t={final_t} ({final_t - start_t:.1f} years)')

ts = md.results.TransientSolution
nv = md.mesh.numberofvertices
# Same TransientSolution round-trip quirk found in check_ho_relaxed.py / historical_dhdt_tuning.
if isinstance(ts, list):
    print(f'   {len(ts)} timesteps saved')
    final = ts[-1]
    Vel = np.asarray(final.Vel).ravel()
    H1 = np.asarray(final.Thickness).ravel()
    Surf1 = np.asarray(final.Surface).ravel()
    Base1 = np.asarray(final.Base).ravel()
    ol1 = np.asarray(final.MaskOceanLevelset).ravel()
else:
    raw_vel = np.asarray(ts.Vel).ravel()
    n_steps = raw_vel.size // nv
    print(f'   collapsed object holds {n_steps} timesteps flattened (size={raw_vel.size}, nv={nv})')

    def last_step(field_name):
        arr = np.asarray(getattr(ts, field_name)).ravel()
        return arr.reshape(n_steps, nv)[-1]

    Vel = last_step('Vel')
    H1 = last_step('Thickness')
    Surf1 = last_step('Surface')
    Base1 = last_step('Base')
    ol1 = last_step('MaskOceanLevelset')

print('\n-- Loading AIS3_ho_relaxed.nc for the true 1995 starting geometry (H0)...', flush=True)
md_ref = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_relaxed.nc')
H0 = np.asarray(md_ref.geometry.thickness).ravel()
ol0 = np.asarray(md_ref.mask.ocean_levelset).ravel()

print('\n=== Chunk-1 final-timestep field sanity ===')
for name, arr in [('Vel', Vel), ('Thickness', H1), ('Surface', Surf1), ('Base', Base1)]:
    print(f'   {name}: NaN={np.isnan(arr).sum()} Inf={np.isinf(arr).sum()} '
          f'min={np.nanmin(arr):.3g} max={np.nanmax(arr):.3g} mean={np.nanmean(arr):.3g}')

neg_thickness = (H1 < 0).sum()
print(f'\n   Negative thickness count: {neg_thickness}')

on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
on_surface = np.asarray(md.mesh.vertexonsurface).astype(bool).ravel()
il = np.asarray(md.mask.ice_levelset).ravel()

# Grounding-line migration activity: vertices that changed grounded/floating regime.
grounded0 = ol0 >= 0
grounded1 = ol1 >= 0
newly_floating = grounded0 & ~grounded1
newly_grounded = ~grounded0 & grounded1
print(f'\n=== Grounding-line migration over the 6-year chunk ===')
print(f'   Newly floating (grounded->floating): {int(newly_floating.sum())} vertices')
print(f'   Newly grounded (floating->grounded): {int(newly_grounded.sum())} vertices')

dH = H1 - H0
years_elapsed = final_t - start_t
dhdt_sim_chunk1 = dH / years_elapsed
print(f'\n=== dH distribution over chunk 1 ({years_elapsed:.1f} years) ===')
print(f'   NaN={np.isnan(dH).sum()}')
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f'   p{p:5.1f}: {np.nanpercentile(dH, p):.2f} m')

print('\n-- Comparing chunk-1 dH/dt against MIPKIT dhdt_cpom (EARLY signal, 6 of 24 years)...', flush=True)
mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
# BUGFIX (2026-09-21, found on this exact run): xr_to_mesh requires a 2D variable on a
# rectilinear grid, but mipkit's own dhdt_cpom is 3D (carries cpom_dhdt_time) -- passing the
# full dataset+var_name through crashes with "must be 2D on a rectilinear grid". Same fix
# applied to ais_0.1.py's and ais_0.1_SSA.py's identical pattern.
dhdt_obs_grid = mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=-1)
dhdt_obs_mesh = pyissm.data.interp.xr_to_mesh(
    dhdt_obs_grid, 'dhdt_cpom', md.mesh.x, md.mesh.y, x_var='x1km', y_var='y1km')

gr = (il < 0) & (ol0 > 0) & (ol1 > 0) & on_surface  # stay-grounded-throughout, matches obs coverage
mismatch_rmse = np.sqrt(np.nanmean((dhdt_sim_chunk1[gr] - dhdt_obs_mesh[gr]) ** 2))
bias = np.nanmean(dhdt_sim_chunk1[gr] - dhdt_obs_mesh[gr])
print(f'   Grounded dH/dt mismatch RMSE (chunk 1 only) vs dhdt_cpom = {mismatch_rmse:.3f} m/yr')
print(f'   Mean bias (sim - obs) = {bias:.3f} m/yr')
print('   NOTE: this is a 6-year rate compared against dhdt_cpom\'s own multi-year rate -- a')
print('   directional/magnitude sanity check only. The official comparison happens once the')
print('   full 1995-2019 (24-year) run completes.')

print('\n-- Plotting spatial dH (chunk 1, 6 years)...', flush=True)
x2d = np.asarray(md.mesh.x2d).ravel() if hasattr(md.mesh, 'x2d') else np.asarray(md.mesh.x).ravel()
y2d = np.asarray(md.mesh.y2d).ravel() if hasattr(md.mesh, 'y2d') else np.asarray(md.mesh.y).ravel()
dH_2d = dH[on_base] if hasattr(md.mesh, 'x2d') else dH
fig, ax = plt.subplots(figsize=(9, 9))
vmax = np.nanpercentile(np.abs(dH_2d), 98)
sc = ax.scatter(x2d, y2d, c=dH_2d, s=1, cmap='RdBu_r', vmin=-vmax, vmax=vmax, rasterized=True)
ax.set_title(f'AIS3_historical_1995_2019 chunk 1: dH over t={start_t:.0f}-{final_t:.0f} (m)\n'
             f'mean|dH|={np.nanmean(np.abs(dH)):.2f}m, RMSE vs dhdt_cpom*{years_elapsed:.0f}yr={mismatch_rmse*years_elapsed:.2f}m')
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(sc, ax=ax, fraction=0.04)
plt.tight_layout()
out_png = f'{MODEL_DIR}/historical_chunk1_dH_spatial.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
print('DONE')
