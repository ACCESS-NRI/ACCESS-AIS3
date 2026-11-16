"""SSA-mesh version of restore_floored_thickness.py -- restores real observed thickness for
the 198,230 vertices (12.48%) carrying the artificial 100m numerical-stability floor in
AIS3_SSA_inverted.nc, BEFORE relaxation runs (unlike HO, where the fix was applied after
friction re-inversion/melt calibration but before relaxation -- SSA's own stage order puts
relaxation first, so the fix has to go in at this earlier point).

check_ssa_thickness_fix_melt_impact.py found this is NOT a minor edge case: 62.1% of the
floored vertices (123,196) are floating -- 59.74% of ALL floating area in the domain -- with
a mean draft-depth shift from 89.6m (artificial) to 12.2m (real), a 77.5m shoaling. Since
melt calibration is driven by ocean thermal forcing sampled at draft depth, this plausibly
affected the majority of the ice-shelf melt calculation, motivating the full cascade redo:
this fix -> re-run ssa_relaxation_budd -> redo melt_gamma_tuning_ssa -> redo the deltaT_basin
refit chain.

Same technique as the HO fix: real thickness recovered from AIS3_param.nc (the only place it
still exists) via (x,y) nearest-neighbour, surface/base recomputed with the same hydrostatic-
floating/bed-grounded formula ssa_inverted_solve_budd itself used. Backs up before overwriting.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import shutil
import numpy as np
import pyissm
from scipy.interpolate import NearestNDInterpolator

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
TARGET = f'{MODEL_DIR}/AIS3_SSA_inverted.nc'
BACKUP = f'{MODEL_DIR}/AIS3_SSA_inverted_prethicknessfix_backup.nc'
FLOOR_TOLERANCE = 0.5

if not os.path.exists(BACKUP):
    print(f'-- Backing up {TARGET} -> {BACKUP} before any modification...', flush=True)
    shutil.copy2(TARGET, BACKUP)
else:
    print(f'-- Backup already exists at {BACKUP}, not overwriting it.', flush=True)

print('-- Loading original parameterized model (AIS3_param.nc) for real thickness...', flush=True)
md_orig = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_param.nc')
H_real_orig = np.asarray(md_orig.geometry.thickness).ravel()
x_orig = np.asarray(md_orig.mesh.x).ravel()
y_orig = np.asarray(md_orig.mesh.y).ravel()
real_lookup = NearestNDInterpolator(np.column_stack([x_orig, y_orig]), H_real_orig)

print('-- Loading target model (AIS3_SSA_inverted.nc)...', flush=True)
md = pyissm.model.io.load_model(TARGET)
nv = md.mesh.numberofvertices
print(f'   {nv} vertices')

mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()
H_real_on_target_mesh = real_lookup(np.column_stack([mesh_x, mesh_y]))

H_current = np.asarray(md.geometry.thickness).ravel().copy()
floored = np.abs(H_current - 100.0) < FLOOR_TOLERANCE
print(f'\n-- {int(floored.sum())} vertices ({100*floored.sum()/nv:.2f}%) identified as sitting '
      f'at the artificial 100m floor...', flush=True)
print(f'   Real thickness for these vertices: min={H_real_on_target_mesh[floored].min():.2f} '
      f'max={H_real_on_target_mesh[floored].max():.2f} mean={H_real_on_target_mesh[floored].mean():.2f}')

H_new = H_current.copy()
H_new[floored] = H_real_on_target_mesh[floored]
H_new = np.maximum(H_new, 1.0)

print('-- Recomputing surface/base for restored vertices...', flush=True)
ri = float(md.materials.rho_ice)
rw = float(md.materials.rho_water)
ol = np.asarray(md.mask.ocean_levelset).ravel()
bed = np.asarray(md.geometry.bed).ravel()
surf_current = np.asarray(md.geometry.surface).ravel().copy()
base_current = np.asarray(md.geometry.base).ravel().copy()

flt = (ol < 0) & floored
grd = (ol >= 0) & floored

base_new = base_current.copy()
surf_new = surf_current.copy()
base_new[flt] = np.maximum(-H_new[flt] * ri / rw, bed[flt])
surf_new[flt] = base_new[flt] + H_new[flt]
base_new[grd] = bed[grd]
surf_new[grd] = bed[grd] + H_new[grd]

md.geometry.thickness = H_new
md.geometry.base = base_new
md.geometry.surface = surf_new

neg_or_zero = (H_new <= 0).sum()
print(f'\n-- Sanity: {neg_or_zero} vertices with thickness <= 0 after restoration (expect 0)')
print(f'   Restored thickness distribution at previously-floored vertices: '
      f'min={H_new[floored].min():.2f} max={H_new[floored].max():.2f} '
      f'mean={H_new[floored].mean():.2f} (was uniformly 100.0 before)')

print(f'\n-- Saving patched model to {TARGET} (overwriting, original preserved at {BACKUP})...', flush=True)
pyissm.model.io.save_model(md, TARGET)
print('DONE')
