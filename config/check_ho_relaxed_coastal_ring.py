"""Investigate the coastal ring of ~-99m dH in AIS3_ho_relaxed.nc. Hypothesis: this project
has repeatedly floored thin ice at exactly 100m upstream (np.maximum(H, 100.0), a numerical-
stability measure, not a real physical thickness -- used in ssa_inverted_solve,
finalize_p1q1_reg_lcurve.py, etc.). If the actual transient's masstransport.min_thickness is
much lower, cells artificially inflated to 100m would rapidly relax down toward that lower
floor once masstransport is enabled -- producing a near-uniform ~-99m adjustment exactly
where the artificial flooring was applied (thin/coastal ice), not a real physical signal.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'

print('-- Loading pre-relaxation state (AIS3_melt_gamma_tuning.nc, ho_relaxation\'s actual input)...', flush=True)
md0 = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_melt_gamma_tuning.nc')
H0 = np.asarray(md0.geometry.thickness).ravel()
print(f'   masstransport.min_thickness = {md0.masstransport.min_thickness}')
near_100 = np.abs(H0 - 100.0) < 0.5
print(f'   {near_100.sum()} vertices ({100*near_100.mean():.2f}%) sit within 0.5m of exactly 100m thickness')

print('\n-- Loading post-relaxation state (AIS3_ho_relaxed.nc)...', flush=True)
md1 = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_relaxed.nc')
ts = md1.results.TransientSolution
nv = md1.mesh.numberofvertices
raw = np.asarray(ts.Thickness).ravel()
n_steps = raw.size // nv
H1 = raw.reshape(n_steps, nv)[-1]

dH = H1 - H0
print(f'   dH stats: min={dH.min():.2f} max={dH.max():.2f}')

near_neg99 = np.abs(dH - (-99.0)) < 2.0
print(f'\n=== Cross-check ===')
print(f'   {near_neg99.sum()} vertices with dH within 2m of -99m')
overlap = near_100 & near_neg99
print(f'   Of those, {overlap.sum()} ({100*overlap.sum()/max(near_neg99.sum(),1):.1f}%) were '
      f'ALSO within 0.5m of the 100m flooring value pre-relaxation')
print(f'   Of the {near_100.sum()} vertices at the 100m floor, '
      f'{(near_100 & near_neg99).sum()} ({100*(near_100 & near_neg99).sum()/max(near_100.sum(),1):.1f}%) '
      f'ended up with dH~-99m')

print(f'\n=== What did H1 (post-relaxation) actually settle to for the near-100m-floor group? ===')
H1_floor_group = H1[near_100]
for p in [0, 5, 25, 50, 75, 95, 100]:
    print(f'   p{p:5.1f}: {np.nanpercentile(H1_floor_group, p):.3f} m')

print('DONE')
