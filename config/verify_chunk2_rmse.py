"""Independent verification of chunk-2 HO friction re-inversion RMSE, run as its own PBS
job (the login node OOMs trying to load this full 3D-extruded ~23.8M-node model)."""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pyissm

print('loading AIS3_ho_friction_inv.nc...', flush=True)
md = pyissm.model.io.load_model('models/AIS3_ho_friction_inv.nc')
print('loaded', flush=True)

vel = np.asarray(md.results.StressbalanceSolution.Vel).ravel()
vo = np.asarray(md.inversion.vel_obs).ravel()
il = np.asarray(md.mask.ice_levelset).ravel()
ol = np.asarray(md.mask.ocean_levelset).ravel()
on_surface = np.asarray(md.mesh.vertexonsurface).astype(bool).ravel()
gr = (il < 0) & (ol > 0) & on_surface

rmse = np.sqrt(np.nanmean((vel[gr] - vo[gr]) ** 2))
print(f'Grounded (surface) RMSE, independent recompute: {rmse}')
print(f'n grounded surface points: {int(gr.sum())}')
print(f'vel[gr] stats: min={np.nanmin(vel[gr]):.2f} max={np.nanmax(vel[gr]):.2f} mean={np.nanmean(vel[gr]):.2f}')
print(f'vo[gr] stats: min={np.nanmin(vo[gr]):.2f} max={np.nanmax(vo[gr]):.2f} mean={np.nanmean(vo[gr]):.2f}')
print('DONE')
