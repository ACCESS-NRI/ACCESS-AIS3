"""Restore real observed thickness for the ~12.48% of the domain that has been carrying an
artificial 100m numerical-stability floor (from ssa_inverted_solve, ais_0.1.py:1398-1409)
through every downstream step since -- SSA relaxation, HO thermal steady-state, HO friction
re-inversion, melt calibration, and now HO relaxation, where check_ho_relaxed_coastal_ring.py
confirmed 57.2% of those artificially-floored vertices collapsed to ISSM's own 1m
masstransport.min_thickness within a single year once masstransport was finally enabled --
not a physically real signal, a numerical artifact of the mismatch between the artificial
100m starting value and whatever velocity/flux these cells actually have.

The floor was legitimate for the friction INVERSION itself (kept driving stress consistent
with what ssa_friction_inv_reg_lcurve actually solved against), but nothing ever restored
real thickness afterward for forward/transient use. The pre-floor value was never saved
anywhere downstream (it was a local variable, _H_orig, in ssa_inverted_solve, discarded once
that step finished) -- so this goes back to AIS3_param.nc (the original parameterized model,
before ANY flooring was ever applied) to recover real observed thickness.

Patches AIS3_melt_gamma_tuning.nc (ho_relaxation's actual input) in place, recomputing
surface/base for the restored vertices with the same hydrostatic-floating / bed-grounded
formula ssa_inverted_solve itself used (just with real H instead of the floored value). Does
NOT clamp restored thickness to any minimum beyond what's already real -- ISSM's own
masstransport.min_thickness=1.0 is the appropriate floor during the actual transient, letting
genuinely thin cells be genuinely thin is more physically honest than an artificial mask.

Backs up the pre-patch file first. ho_relaxation must be re-run against this corrected input
to get a genuinely fixed AIS3_ho_relaxed.nc -- this script only fixes the input.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import shutil
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
TARGET = f'{MODEL_DIR}/AIS3_melt_gamma_tuning.nc'
BACKUP = f'{MODEL_DIR}/AIS3_melt_gamma_tuning_prethicknessfix_backup.nc'
FLOOR_TOLERANCE = 0.5  # m, matches check_ho_relaxed_coastal_ring.py's own criterion

if not os.path.exists(BACKUP):
    print(f'-- Backing up {TARGET} -> {BACKUP} before any modification...', flush=True)
    shutil.copy2(TARGET, BACKUP)
else:
    print(f'-- Backup already exists at {BACKUP}, not overwriting it.', flush=True)

print('-- Loading original parameterized model (AIS3_param.nc) for real thickness...', flush=True)
md_orig = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_param.nc')
H_real_orig = np.asarray(md_orig.geometry.thickness).ravel()
print(f'   AIS3_param.nc has {md_orig.mesh.numberofvertices} vertices')

print('-- Loading target model (AIS3_melt_gamma_tuning.nc)...', flush=True)
md = pyissm.model.io.load_model(TARGET)
nv = md.mesh.numberofvertices
print(f'   {nv} vertices')

# AIS3_param.nc and AIS3_melt_gamma_tuning.nc are on DIFFERENT meshes (the latter is extruded
# HO, and possibly a different 2D vertex count/ordering after everything this pipeline did
# upstream -- extraction, remeshing checks, etc.) -- cannot assume vertex[i] in one file is
# the same physical location as vertex[i] in the other. Interpolate real thickness from
# AIS3_param.nc's mesh onto md's mesh by (x,y) nearest-neighbour, matching the pattern used
# throughout this project for cross-mesh data transfer.
from scipy.interpolate import NearestNDInterpolator
x_orig = np.asarray(md_orig.mesh.x).ravel()
y_orig = np.asarray(md_orig.mesh.y).ravel()
# AIS3_param.nc is presumably the 2D (pre-extrusion) SSA-era mesh -- use its own x/y directly.
real_lookup = NearestNDInterpolator(np.column_stack([x_orig, y_orig]), H_real_orig)

mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()
# For an extruded 3D mesh, x/y repeat per layer -- but the (x,y) lookup is layer-independent,
# so this is safe (every layer's vertex gets the same real-thickness value for its (x,y)
# column, correctly matching how ice thickness is a purely 2D/columnar quantity).
H_real_on_target_mesh = real_lookup(np.column_stack([mesh_x, mesh_y]))

H_current = np.asarray(md.geometry.thickness).ravel().copy()
floored = np.abs(H_current - 100.0) < FLOOR_TOLERANCE
print(f'\n-- {int(floored.sum())} vertices ({100*floored.sum()/nv:.2f}%) identified as sitting '
      f'at the artificial 100m floor...', flush=True)

print(f'   Real thickness for these vertices (from AIS3_param.nc, nearest-neighbour): '
      f'min={H_real_on_target_mesh[floored].min():.2f} max={H_real_on_target_mesh[floored].max():.2f} '
      f'mean={H_real_on_target_mesh[floored].mean():.2f}')

H_new = H_current.copy()
H_new[floored] = H_real_on_target_mesh[floored]
# Ice cannot have zero/negative thickness -- ISSM's own transient min_thickness (1.0) is the
# appropriate physical floor from here on, not an arbitrary reintroduced mask, but avoid
# literal <=0 values causing an immediate geometry inconsistency at save/consistency-check time.
H_new = np.maximum(H_new, 1.0)

print('-- Recomputing surface/base for restored vertices (same hydrostatic/bed formula '
      'ssa_inverted_solve itself used)...', flush=True)
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
