"""Deeper sanity check of the salvaged AIS3_ho_relaxed.nc: NaN/Inf/extreme values in the
final timestep's Vel/Thickness/Surface/Base fields, grounded RMSE vs observed velocity (as a
consistency check that relaxation didn't degrade the friction-inverted fit too badly), and a
spatial plot of thickness change over the relaxation.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'

print('-- Loading AIS3_ho_relaxed.nc...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_relaxed.nc')

ts = md.results.TransientSolution
nv = md.mesh.numberofvertices
# BUGFIX (round 2): the "collapsed to a bare object" fix wasn't quite right either -- the
# bare object actually holds the FULL transient history as flattened multi-timestep arrays
# (final.Vel.size == 50*nv exactly, confirmed by the IndexError this crashed with), not just
# the final step. Reshape and take the last row to get the genuine final-timestep field.
if isinstance(ts, list):
    print(f'   {len(ts)} timesteps saved')
    final = ts[-1]
    Vel = np.asarray(final.Vel).ravel()
    H1 = np.asarray(final.Thickness).ravel()
    Surf1 = np.asarray(final.Surface).ravel()
    Base1 = np.asarray(final.Base).ravel()
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
H0 = np.asarray(md.geometry.thickness).ravel()

print('\n=== Final timestep field sanity ===')
for name, arr in [('Vel', Vel), ('Thickness', H1), ('Surface', Surf1), ('Base', Base1)]:
    print(f'   {name}: NaN={np.isnan(arr).sum()} Inf={np.isinf(arr).sum()} '
          f'min={np.nanmin(arr):.3g} max={np.nanmax(arr):.3g} mean={np.nanmean(arr):.3g}')

neg_thickness = (H1 < 0).sum()
print(f'\n   Negative thickness count: {neg_thickness}')

on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
vel_obs = np.asarray(md.inversion.vel_obs).ravel() if hasattr(md.inversion, 'vel_obs') else None
il = np.asarray(md.mask.ice_levelset).ravel()
ol = np.asarray(md.mask.ocean_levelset).ravel()
on_surface = np.asarray(md.mesh.vertexonsurface).astype(bool).ravel()

if vel_obs is not None:
    gr_surf = (il < 0) & (ol >= 0) & (vel_obs > 0) & on_surface
    rmse = np.sqrt(np.nanmean((Vel[gr_surf] - vel_obs[gr_surf]) ** 2))
    print(f'\n=== Grounded surface RMSE vs obs (post-relaxation): {rmse:.2f} m/yr ===')
    print('   (compare against the melt-calibrated/pre-relaxation state -- some drift is')
    print('   expected from a real 1yr forward transient, large drift would be concerning)')

dH = H1 - H0
print(f'\n=== dH distribution over relaxation ===')
print(f'   NaN={np.isnan(dH).sum()}')
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f'   p{p:5.1f}: {np.nanpercentile(dH, p):.2f} m')

print('\n-- Plotting spatial dH...', flush=True)
x2d = np.asarray(md.mesh.x2d).ravel() if hasattr(md.mesh, 'x2d') else np.asarray(md.mesh.x).ravel()
y2d = np.asarray(md.mesh.y2d).ravel() if hasattr(md.mesh, 'y2d') else np.asarray(md.mesh.y).ravel()
dH_2d = dH[on_base] if hasattr(md.mesh, 'x2d') else dH
fig, ax = plt.subplots(figsize=(9, 9))
vmax = np.nanpercentile(np.abs(dH_2d), 98)
sc = ax.scatter(x2d, y2d, c=dH_2d, s=1, cmap='RdBu_r', vmin=-vmax, vmax=vmax, rasterized=True)
ax.set_title(f'AIS3_ho_relaxed: dH over 1yr relaxation (m)\n'
             f'max|dH|=982.93m (spatial max), mean|dH|=13.00m')
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(sc, ax=ax, fraction=0.04)
plt.tight_layout()
out_png = f'{MODEL_DIR}/ho_relaxed_dH_spatial.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
print('DONE')
