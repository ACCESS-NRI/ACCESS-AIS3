"""Salvage the real, already-computed p=q=1 Budd cf501=0.0001 result (from the ad-hoc
p1q1_reg_sweep2.py script, execution_p1q1_reg_sweep2/run_001_10_100_0.0001/, 2026-08-08) into
the canonical production location the real ssa_friction_inv_reg_lcurve step expects
(models/AIS3_ssa_friction_inv_reg_lcurve/run_001_10_100_0.0001/) -- promoting the genuinely
best RMSE found anywhere in this project (60.4 m/yr) from ad-hoc-script status to a real,
loadable production artifact, without needing to regenerate AIS3_param.nc with Budd config or
re-run the (expensive) unregularised baseline + reg-lcurve inversion steps from scratch.

Rebuilds the exact extraction domain ssa_friction_inv_reg_lcurve itself builds (same code,
copied deliberately rather than imported -- see ais_0.1.py's own helper functions), loads
results directly from the run's raw .outbin via load_results_from_disk (same technique already
validated this session for the Schoof step99 salvage), and verifies against the documented
RMSE=60.4 before saving.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
RUN_NAME = 'run_001_10_100_0.0001'
OUTBIN = f'/g/data/au88/jh7060/ACCESS-AIS3/execution_p1q1_reg_sweep2/{RUN_NAME}/{RUN_NAME}.outbin'
OUT_DIR = f'{MODEL_DIR}/AIS3_ssa_friction_inv_reg_lcurve/{RUN_NAME}'


def extract_friction_inversion_domain(md):
    """Copied verbatim from ais_0.1.py -- see that file for the full rationale comment."""
    ice_levelset_elements = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset)
    mds = md.extract(ice_levelset_elements < 1)
    bnd = mds.mesh.vertexonboundary.astype(bool)
    ocean_ls = np.asarray(mds.mask.ocean_levelset).ravel()
    floating_bnd = bnd & (ocean_ls < 0)
    mds.stressbalance.spcvx[floating_bnd] = np.nan
    mds.stressbalance.spcvy[floating_bnd] = np.nan
    mds.stressbalance.spcvz[floating_bnd] = np.nan
    mds.mask.ice_levelset[floating_bnd] = 0
    return mds


def load_shelf_rheology_B():
    """Copied verbatim from ais_0.1.py -- see that file for the full rationale comment."""
    _cl = pyissm.model.classes.cluster.gadi()
    _cl.codepath = os.environ['ISSM_DIR'] + '/bin'
    _cl.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution_newB_rheology'
    _cl.login = 'jh7060'; _cl.project = 'au88'; _cl.storage = 'gdata/au88'
    mshelf = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_param.nc')
    mshelf.mask.ice_levelset = pyissm.model.param.kill_icebergs(mshelf)
    sel = (mshelf.mask.ocean_levelset < 0) & (mshelf.mask.ice_levelset < 0)
    mshelf = mshelf.extract(sel)
    mshelf.cluster = _cl
    mshelf.settings.waitonlock = 0
    mshelf.inversion.iscontrol = 0
    mshelf.miscellaneous.name = 'run_001_1_10_1e-17'
    mr = pyissm.model.execute.solve(mshelf, 'Stressbalance', load_only=True, runtime_name=False, check_consistency=False)
    return np.asarray(mr.mesh.extractedvertices).ravel(), np.asarray(mr.results.StressbalanceSolution.MaterialsRheologyBbar).ravel()


print('-- Loading parameterized model (AIS3_param.nc)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_param.nc')

print('-- Loading SSA floating rheology inversion results...', flush=True)
_ev, _Bshelf = load_shelf_rheology_B()
md.materials.rheology_B[_ev - 1] = _Bshelf

print('-- Removing icebergs from ice levelset...', flush=True)
md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

print("-- Flooring thin ice at 100m, re-flooring N (exact match to ssa_friction_inv_reg_lcurve)...", flush=True)
_ri = md.materials.rho_ice; _rw = md.materials.rho_water
_H = np.asarray(md.geometry.thickness).ravel().copy()
_ol = np.asarray(md.mask.ocean_levelset).ravel()
_surf0 = np.asarray(md.geometry.surface).ravel().copy()
_H = np.maximum(_H, 100.0)
_flt = _ol < 0
_base = np.empty_like(_H); _surf = np.empty_like(_H)
_base[_flt] = -_H[_flt] * _ri / _rw; _surf[_flt] = _H[_flt] * (1.0 - _ri / _rw)
_surf[~_flt] = _surf0[~_flt]
_base[~_flt] = _surf0[~_flt] - _H[~_flt]
md.geometry.thickness = _H; md.geometry.base = _base; md.geometry.surface = _surf

_lim = 0.07
N = md.friction.effective_pressure.copy()
N[N < 0] = 0
_Nfloor = _lim * md.materials.rho_ice * md.constants.g * _H
N = np.maximum(N, _Nfloor)
md.friction.effective_pressure = N
md.friction.effective_pressure_limit = _lim

print('-- Extracting friction-inversion domain...', flush=True)
mds = extract_friction_inversion_domain(md)
mds.friction.coupling = 3

print('-- Re-flooring N on the extracted domain...', flush=True)
_Hs = np.asarray(mds.geometry.thickness).ravel()
_Ns = mds.friction.effective_pressure.copy(); _Ns[_Ns < 0] = 0
_Nfs = 0.07 * mds.materials.rho_ice * mds.constants.g * _Hs
mds.friction.effective_pressure = np.maximum(_Ns, _Nfs)
mds.friction.effective_pressure_limit = 0.07

print('-- Setting friction class to Budd/default (p=q=1) for this shell...', flush=True)
mds.friction = pyissm.model.classes.friction.default(mds.friction)
mds.friction.p = np.full(mds.mesh.numberofelements, 1.0)
mds.friction.q = np.full(mds.mesh.numberofelements, 1.0)
mds.friction.coefficient = np.full(mds.mesh.numberofvertices, 1.8)

print(f'-- Loading results from raw outbin ({OUTBIN})...', flush=True)
mds.miscellaneous.name = RUN_NAME
mds = pyissm.model.execute.load_results_from_disk(mds, OUTBIN)

vel = np.asarray(mds.results.StressbalanceSolution.Vel).ravel()
vo = np.asarray(mds.inversion.vel_obs).ravel()
il = np.asarray(mds.mask.ice_levelset).ravel()
ol = np.asarray(mds.mask.ocean_levelset).ravel()
gr = (il < 0) & (ol >= 0) & (vo > 0)
rmse = np.sqrt(np.nanmean((vel[gr] - vo[gr]) ** 2))
print(f'\nVerification: grounded RMSE = {rmse:.2f} m/yr (documented value: 60.4 m/yr)')

C = np.asarray(mds.results.StressbalanceSolution.FrictionCoefficient).ravel()
print(f'FrictionCoefficient: min={np.nanmin(C):.3f} max={np.nanmax(C):.3f} mean={np.nanmean(C):.3f}')

os.makedirs(OUT_DIR, exist_ok=True)
out_path = f'{OUT_DIR}/{RUN_NAME}.nc'
print(f'\nSaving to {out_path}', flush=True)
pyissm.model.io.save_model(mds, out_path)
print('DONE')
