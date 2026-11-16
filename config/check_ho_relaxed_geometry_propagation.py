"""HO counterpart of check_relaxed_geometry_propagation.py (worklog 4y): did ho_relaxation's end
state ever reach the historical run?

Compares md.geometry.thickness (full 3D) of AIS3_melt_gamma_tuning.nc (relaxation input),
AIS3_ho_relaxed.nc, and AIS3_historical_1995_2019_chunk1rerun.nc (the historical run's 1995
start, fresh-started from AIS3_ho_relaxed.nc) against each other and against the relaxation's own
end state (AIS3_ho_relaxed.nc TransientSolution, last step). Bug confirmed for HO if the relaxed
and historical geometries equal the melt_gamma_tuning geometry and differ from the end state.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import gc
import numpy as np
import pyissm

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
H = {}
OL = {}
relax_end = None
relax_end_ol = None
for name in ['AIS3_melt_gamma_tuning', 'AIS3_ho_relaxed', 'AIS3_historical_1995_2019_chunk1rerun']:
    print(f'-- loading {name}...', flush=True)
    md = pyissm.model.io.load_model(f'{M}/{name}.nc')
    H[name] = np.asarray(md.geometry.thickness).ravel().copy()
    OL[name] = np.asarray(md.mask.ocean_levelset).ravel().copy()
    print(f'   timestepping {md.timestepping.start_time}->{md.timestepping.final_time}, '
          f'mean H {H[name].mean():.3f} m', flush=True)
    if name == 'AIS3_ho_relaxed':
        ts = md.results.TransientSolution
        nv = md.mesh.numberofvertices

        def last(field):
            if isinstance(ts, list):
                return np.asarray(getattr(ts[-1], field)).ravel().copy()
            raw = np.asarray(getattr(ts, field)).ravel()
            return raw.reshape(raw.size // nv, nv)[-1].copy()

        relax_end = last('Thickness')
        try:
            relax_end_ol = last('MaskOceanLevelset')
        except AttributeError:
            relax_end_ol = None
        n = len(ts) if isinstance(ts, list) else np.asarray(ts.Thickness).size // nv
        print(f'   relaxation history: {n} saved steps', flush=True)
    del md
    gc.collect()

ref = H['AIS3_melt_gamma_tuning']
print('\n=== geometry.thickness vs relaxation INPUT (AIS3_melt_gamma_tuning) ===')
for k, v in H.items():
    d = v - ref
    print(f'{k:40s} max|d| {np.abs(d).max():.4f} m  mean|d| {np.abs(d).mean():.4f} m')
print('\n=== geometry.thickness vs relaxation END STATE (AIS3_ho_relaxed TransientSolution[-1]) ===')
for k, v in H.items():
    d = v - relax_end
    print(f'{k:40s} max|d| {np.abs(d).max():.2f} m  mean|d| {np.abs(d).mean():.3f} m')
print('\n=== mask.ocean_levelset: fraction of vertices with the same grounded/floating state ===')
for k, v in OL.items():
    s = f'vs input {100 * np.mean((v >= 0) == (OL["AIS3_melt_gamma_tuning"] >= 0)):.3f}%'
    if relax_end_ol is not None:
        s += f'   vs relaxation end {100 * np.mean((v >= 0) == (relax_end_ol >= 0)):.3f}%'
    print(f'{k:40s} {s}')
print('DONE')
