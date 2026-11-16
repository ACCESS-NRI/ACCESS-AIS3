"""Did the SSA relaxation's end state ever reach downstream steps?
Compares md.geometry.thickness across the SSA chain and against the relaxation's own final state.
If AIS3_SSA_relaxed.nc's geometry equals AIS3_SSA_inverted.nc's, relaxation was never propagated."""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import gc
import numpy as np
import pyissm

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
H = {}
relax_end = None
for name in ['AIS3_SSA_inverted', 'AIS3_SSA_relaxed', 'AIS3_melt_final_ssa', 'AIS3_historical_1995_2019_SSA']:
    md = pyissm.model.io.load_model(f'{M}/{name}.nc')
    H[name] = np.asarray(md.geometry.thickness).ravel().copy()
    if name == 'AIS3_SSA_relaxed':
        ts = md.results.TransientSolution
        nv = md.mesh.numberofvertices
        if isinstance(ts, list) and len(ts) > 0:
            relax_end = np.asarray(ts[-1].Thickness).ravel().copy()
        elif not isinstance(ts, list):
            raw = np.asarray(ts.Thickness).ravel()
            relax_end = raw.reshape(raw.size // nv, nv)[-1].copy()
        print(f'{name}: relaxation span {md.timestepping.start_time}->{md.timestepping.final_time}')
    del md
    gc.collect()
    print(f'loaded {name}: mean H {H[name].mean():.3f} m')

ref = H['AIS3_SSA_inverted']
for k, v in H.items():
    d = v - ref
    print(f'{k:32s} geometry - inverted geometry: max|d| {np.abs(d).max():.4f} m, mean|d| {np.abs(d).mean():.4f} m')
if relax_end is not None:
    for k, v in H.items():
        d = v - relax_end
        print(f'{k:32s} geometry - RELAXATION END STATE: max|d| {np.abs(d).max():.2f} m, mean|d| {np.abs(d).mean():.3f} m')
print('DONE')
