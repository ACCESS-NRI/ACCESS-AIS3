"""Build the FINAL, fully self-consistent SSA melt-calibrated model, AIS3_melt_final_ssa.nc.

Found necessary (2026-09-21) because historical_dhdt_tuning_ssa loads
AIS3_melt_gamma_tuning_ssa.nc, which only carries melt_gamma_tuning_ssa's own COARSE 9-point
sweep pick (gamma_0=5537.7, published/unrefit deltaT_basin) -- that file was never updated
after the deltaT_basin secant refit (ssa_melt_deltaT_basin_refit.py) converged on the actual
accepted answer, gamma_0=300 with a per-basin REFIT deltaT_basin (confirmed via
ssa_melt_deltaT_refit_recalibrate.py, job 179305728: best by J2 = gamma_0=300, i=1).

Neither refit script ever saved a full model file -- they only tracked numeric state
(deltaT_refit_state_ssa.json) and pulled raw melt-rate results via load_only=True against
each candidate/round's own execution folder. This script reconstructs the winning
candidate's full ismip6 basalforcings config (mirroring setup_base_model() in
ssa_melt_deltaT_basin_refit.py exactly), pulls its already-completed final-round solve
(AIS3_ssa_deltaT_refit_g1_r5) via the same load_only=True technique, and saves the result as
a genuinely complete, self-consistent model -- so downstream steps can trust what they load
instead of each having to reconstruct basalforcings from scratch themselves.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import json
import numpy as np
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
STATE_FILE = f'{MODEL_DIR}/deltaT_refit_state_ssa.json'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                 'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
TIME_SENTINEL = 1e9
WINNING_CANDIDATE = 1  # gamma_0=300, confirmed best-by-J2 in ssa_melt_deltaT_refit_recalibrate.py

cluster = pyissm.model.classes.cluster.gadi()
cluster.codepath = os.environ['ISSM_DIR'] + '/bin'
cluster.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
cluster.storage = 'gdata/au88+gdata/vk83+gdata/av17'
cluster.moduleuse = ['/g/data/vk83/modules/']
cluster.moduleload = ['access-issm_ad/2026.05.0']
cluster.np = 48
cluster.memory = 190
cluster.time = 60 * 48
cluster.login = 'jh7060'
cluster.project = 'au88'

print(f'-- Loading refit state ({STATE_FILE})...', flush=True)
with open(STATE_FILE) as f:
    state = json.load(f)
cand = state['candidates'][str(WINNING_CANDIDATE)]
gamma_0 = float(cand['gamma_0'])
final_round = len(cand['history']) - 1
delta_t_refit = np.asarray(cand['history'][-1]['delta_t'], dtype=float)
run_name = f'AIS3_ssa_deltaT_refit_g{WINNING_CANDIDATE}_r{final_round}'
print(f'   Winning candidate i={WINNING_CANDIDATE}: gamma_0={gamma_0:.1f}, '
      f'final_round={final_round}, converged={cand["converged"]}, run={run_name}')

print('-- Loading relaxed SSA model and reconstructing ismip6 basalforcings '
      '(same setup as ssa_melt_deltaT_basin_refit.py)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_relaxed.nc')
md.results.TransientSolution = []

md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)
ds_basin = xr.open_dataset(BASIN_FILE)
basin_grid = ds_basin['basinNumber'].values
bx, by = ds_basin['x'].values, ds_basin['y'].values
num_basins = int(np.nanmax(basin_grid)) + 1
assert delta_t_refit.size == num_basins, (
    f'refit delta_t size {delta_t_refit.size} != num_basins {num_basins}')

elx = np.asarray(md.mesh.elements).astype(int) - 1
mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()
ecx = mesh_x[elx[:, :3]].mean(axis=1)
ecy = mesh_y[elx[:, :3]].mean(axis=1)
bxx, byy = np.meshgrid(bx, by)
basin_lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
basin_id_elements = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)
md.basalforcings.basin_id = basin_id_elements
md.basalforcings.num_basins = num_basins

md.basalforcings.delta_t = delta_t_refit
md.basalforcings.islocal = 1
md.basalforcings.gamma_0 = gamma_0

print('-- Interpolating Zhou ocean thermal-forcing climatology onto the mesh...', flush=True)
ds_tf = xr.open_dataset(ZHOU_TF_FILE)
tfx, tfy, tfz = ds_tf['x'].values, ds_tf['y'].values, ds_tf['z'].values
tf_full = ds_tf['tf'].values
tf_list = []
for k in range(tfz.size):
    interp_k = RegularGridInterpolator((tfy, tfx), tf_full[k], method='linear',
                                        bounds_error=False, fill_value=np.nan)
    vals = interp_k(np.column_stack([mesh_y, mesh_x]))
    vals = np.nan_to_num(vals, nan=0.0)
    col = np.append(vals, TIME_SENTINEL)
    tf_list.append(col.reshape(-1, 1))
md.basalforcings.tf = tf_list
md.basalforcings.tf_depths = tfz.copy()

md.inversion.iscontrol = 0
md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
md.transient.ismasstransport = 1
md.masstransport.spcthickness = np.full(md.mesh.numberofvertices, np.nan)
md.smb.mass_balance = np.zeros(md.mesh.numberofvertices)
md.timestepping.start_time = 0
md.timestepping.final_time = 0.01
md.timestepping.time_step = 0.01
md.transient.requested_outputs = ['default', 'BasalforcingsFloatingiceMeltingRate']
md.cluster = cluster
md.miscellaneous.name = run_name
md.settings.waitonlock = 0

print(f'-- Pulling already-completed results from {run_name}...', flush=True)
md = pyissm.model.execute.solve(md, 'Transient', load_only=True, runtime_name=False)

melt = np.asarray(md.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
print(f'\n=== Sanity check ===')
print(f'   gamma_0={md.basalforcings.gamma_0}, delta_t range=[{delta_t_refit.min():.3f}, '
      f'{delta_t_refit.max():.3f}] degC')
print(f'   Floating melt rate: min={np.nanmin(melt):.3f} max={np.nanmax(melt):.3f} '
      f'mean={np.nanmean(melt):.3f} m/yr')

print(f'\nSaving to {MODEL_DIR}/AIS3_melt_final_ssa.nc')
pyissm.model.io.save_model(md, f'{MODEL_DIR}/AIS3_melt_final_ssa.nc')
print('DONE')
