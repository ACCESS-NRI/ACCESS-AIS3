"""Quick screening check before committing to the full SSA cascade redo (patch
AIS3_SSA_inverted.nc -> re-run ssa_relaxation_budd (~9.5h) -> redo melt_gamma_tuning_ssa +
the full deltaT_basin refit chain). Basal melt (basalforcings.floatingice_melting_rate) only
applies to FLOATING ice -- if most of the artificially-100m-floored vertices are actually
grounded, the thickness bug wouldn't have meaningfully affected the melt calibration chain
(gamma_0=300) at all, regardless of the geometry bug being real. If a large fraction ARE
floating, estimate the actual melt sensitivity: floating-ice draft depth (~thickness*rho_ice/
rho_water) determines which Zhou TF depth layer gets sampled, so a large thickness correction
changes draft depth, which can shift melt at those specific vertices (thermal forcing profiles
are typically depth-dependent) -- report both fractions and a rough magnitude estimate.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'

print('-- Loading AIS3_SSA_inverted.nc (pre-relaxation, where the 100m floor originates)...', flush=True)
md0 = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_inverted.nc')
H0 = np.asarray(md0.geometry.thickness).ravel()
ol0 = np.asarray(md0.mask.ocean_levelset).ravel()
floating0 = ol0 < 0
floored = np.abs(H0 - 100.0) < 0.5

print(f'\n=== Floored-vertex regime breakdown ===')
print(f'   {floored.sum()} vertices ({100*floored.mean():.2f}%) at the artificial 100m floor')
print(f'   Of those: {int((floored & floating0).sum())} ({100*(floored & floating0).sum()/max(floored.sum(),1):.1f}%) are FLOATING')
print(f'             {int((floored & ~floating0).sum())} ({100*(floored & ~floating0).sum()/max(floored.sum(),1):.1f}%) are GROUNDED')

print('\n-- Loading AIS3_param.nc for real thickness (same source restore_floored_thickness.py used)...', flush=True)
md_param = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_param.nc')
H_real = np.asarray(md_param.geometry.thickness).ravel()
from scipy.interpolate import NearestNDInterpolator
x_param = np.asarray(md_param.mesh.x).ravel()
y_param = np.asarray(md_param.mesh.y).ravel()
real_lookup = NearestNDInterpolator(np.column_stack([x_param, y_param]), H_real)
mesh_x0 = np.asarray(md0.mesh.x).ravel()
mesh_y0 = np.asarray(md0.mesh.y).ravel()
H_real_on_mesh = real_lookup(np.column_stack([mesh_x0, mesh_y0]))

floored_floating = floored & floating0
n_ff = int(floored_floating.sum())
print(f'\n=== Draft-depth shift for the {n_ff} floored-AND-floating vertices ===')
ri = float(md0.materials.rho_ice)
rw = float(md0.materials.rho_water)
draft_before = H0[floored_floating] * ri / rw  # positive depth below sea level
draft_after = H_real_on_mesh[floored_floating] * ri / rw
draft_shift = draft_before - draft_after  # positive = draft got shallower (less negative depth)
print(f'   Draft before (floored H=100m): {draft_before.mean():.1f} m (uniform, since H was uniform)')
print(f'   Draft after (real H): mean={draft_after.mean():.1f}m min={draft_after.min():.1f}m max={draft_after.max():.1f}m')
print(f'   Mean draft shoaling (shallower by): {draft_shift.mean():.1f} m')

print(f'\n=== Context: how much floating area does this group represent? ===')
elx = np.asarray(md0.mesh.elements).astype(int) - 1
tri_x, tri_y = mesh_x0[elx], mesh_y0[elx]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
vertex_area = np.zeros(md0.mesh.numberofvertices)
for c in range(3):
    np.add.at(vertex_area, elx[:, c], elem_area / 3.0)
total_floating_area = vertex_area[floating0].sum()
affected_floating_area = vertex_area[floored_floating].sum()
print(f'   Total floating area: {total_floating_area/1e6:.1f} km^2')
print(f'   Affected (floored) floating area: {affected_floating_area/1e6:.1f} km^2 '
      f'({100*affected_floating_area/total_floating_area:.2f}% of all floating area)')
print('DONE')
