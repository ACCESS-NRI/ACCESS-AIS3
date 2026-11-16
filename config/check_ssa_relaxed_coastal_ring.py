"""SSA-mesh version of check_ho_relaxed_coastal_ring.py -- check whether AIS3_SSA_relaxed.nc
suffers the same artificial-100m-floor collapse found in AIS3_ho_relaxed.nc. ssa_inverted_
solve_budd (ais_0.1_SSA.py) uses the identical np.maximum(H, 100.0) flooring pattern, and
ssa_relaxation_budd was the FIRST step to enable masstransport on that geometry -- same
mechanism risk. This matters more here: SSA's stage order is inverted_solve -> relaxation ->
melt_gamma_tuning -> deltaT_basin refit, so if relaxation already corrupted thickness, every
downstream SSA melt-calibration result (including the gamma_0=300 answer) inherited it.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'

print('-- Loading pre-relaxation state (AIS3_SSA_inverted.nc)...', flush=True)
md0 = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_inverted.nc')
H0 = np.asarray(md0.geometry.thickness).ravel()
print(f'   masstransport.min_thickness = {md0.masstransport.min_thickness if hasattr(md0.masstransport, "min_thickness") else "N/A"}')
near_100 = np.abs(H0 - 100.0) < 0.5
print(f'   {near_100.sum()} vertices ({100*near_100.mean():.2f}%) sit within 0.5m of exactly 100m thickness')

print('\n-- Loading post-relaxation state (AIS3_SSA_relaxed.nc)...', flush=True)
md1 = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_relaxed.nc')
ts = md1.results.TransientSolution
nv = md1.mesh.numberofvertices
if isinstance(ts, list):
    H1 = np.asarray(ts[-1].Thickness).ravel()
else:
    raw = np.asarray(ts.Thickness).ravel()
    n_steps = raw.size // nv
    print(f'   collapsed object holds {n_steps} timesteps flattened')
    H1 = raw.reshape(n_steps, nv)[-1]

dH = H1 - H0
print(f'   dH stats: min={dH.min():.2f} max={dH.max():.2f}')
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f'   p{p:5.1f}: {np.nanpercentile(dH, p):.2f} m')

near_neg99 = np.abs(dH - (-99.0)) < 2.0
print(f'\n=== Cross-check ===')
print(f'   {near_neg99.sum()} vertices with dH within 2m of -99m')
overlap = near_100 & near_neg99
print(f'   Of those, {overlap.sum()} ({100*overlap.sum()/max(near_neg99.sum(),1):.1f}%) were '
      f'ALSO within 0.5m of the 100m flooring value pre-relaxation')
print(f'   Of the {near_100.sum()} vertices at the 100m floor, '
      f'{(near_100 & near_neg99).sum()} ({100*(near_100 & near_neg99).sum()/max(near_100.sum(),1):.1f}%) '
      f'ended up with dH~-99m')
print('DONE')
