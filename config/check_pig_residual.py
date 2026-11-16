"""Check the magnitude of the model-observation velocity residual specifically within the
PIG/Thwaites drainage basin, for the current AIS3_ho_friction_inv.nc (chunk-2) checkpoint."""
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
gr_surf = (il < 0) & (ol > 0) & on_surface

print('-- Flagging PIG/Thwaites elements via PIG_Outline.exp --', flush=True)
flag_elem = pyissm.model.mesh.flag_elements(md, 'assets/PIG_Outline.exp')
elx = np.asarray(md.mesh.elements).astype(int) - 1
pig_vertex = np.zeros(md.mesh.numberofvertices, dtype=bool)
pig_vertex[elx[flag_elem].ravel()] = True

mask = gr_surf & pig_vertex
residual = vel - vo

print(f'n vertices in PIG mask (grounded, surface, in outline): {int(mask.sum())}')
print(f'residual (model-obs) at PIG: mean={np.nanmean(residual[mask]):.2f} '
      f'median={np.nanmedian(residual[mask]):.2f} std={np.nanstd(residual[mask]):.2f}')
print(f'RMSE at PIG: {np.sqrt(np.nanmean(residual[mask]**2)):.2f} m/yr')
print(f'|residual| at PIG: mean={np.nanmean(np.abs(residual[mask])):.2f} '
      f'max={np.nanmax(np.abs(residual[mask])):.2f}')
print(f'vel[mask] stats: min={np.nanmin(vel[mask]):.2f} max={np.nanmax(vel[mask]):.2f} mean={np.nanmean(vel[mask]):.2f}')
print(f'vo[mask] stats: min={np.nanmin(vo[mask]):.2f} max={np.nanmax(vo[mask]):.2f} mean={np.nanmean(vo[mask]):.2f}')

print('-- Full continental RMSE for comparison --', flush=True)
print(f'full continental RMSE: {np.sqrt(np.nanmean(residual[gr_surf]**2)):.2f} m/yr')

worst = np.argsort(np.where(mask, np.abs(residual), -1))[-10:]
x = np.asarray(md.mesh.x).ravel(); y = np.asarray(md.mesh.y).ravel()
print('-- worst 10 PIG residuals --')
for i in worst[::-1]:
    print(f'  vel={vel[i]:.1f} vel_obs={vo[i]:.1f} resid={residual[i]:.1f}  x={x[i]:.0f} y={y[i]:.0f}')

print('DONE')
