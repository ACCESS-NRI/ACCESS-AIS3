"""Wire the FINAL, deltaT_basin-refit ISMIP7-informed melt calibration result into the real
production file models/AIS3_melt_gamma_tuning.nc, superseding the earlier fixed-deltaT_basin
gamma_0=5537.7 version this same file held (see docs/post_inversion_pipeline_worklog.md SS4a/
4b/4c).

Full story: the original ISMIP7 J1+J2+J3 calibration (fixed deltaT_basin at the published
prior) picked gamma_0=5537.7. Investigating the deltaT_basin-fixed gap
(melt_deltaT_basin_refit.py: per-basin secant root-finding against the Paolo/Adusumilli
target, 8/9 candidates fully converged) showed J1 becomes trivially ~0 for every candidate
once deltaT_basin is properly fit per basin -- it stops discriminating. The real signal moves
to J2 (within-basin spatial/buttressing pattern, which a spatially-uniform-per-basin
deltaT_basin cannot fix), and J2 shows a sharp, clean minimum at gamma_0=300 (J2=12.4 vs.
45-350 for neighbouring candidates). melt_deltaT_refit_recalibrate.py's J1+J2-only Monte Carlo
(the methodologically clean version, no stale-J3 mixed in) is unanimous: 5th/50th/95th
percentile gamma_0 = 300/300/300. J3 was left stale (not recomputed under refit deltaT_basin,
would need 18 more solves) -- accepted as the working value given how sharp and unanimous the
J1+J2 signal already is (2026-09-02 decision).

No new solve needed here -- g1's own refit run (AIS3_deltaT_refit_g1_r5, gamma_0=300, its own
per-basin-refit deltaT_basin, already fully converged and solved) is salvaged directly into
the production location. The shell model must be reconfigured with the EXACT same
basalforcings setup melt_deltaT_basin_refit.py's setup_base_model() used (official IMBIE2
basins, Zhou tf, gamma_0/delta_t overridden to g1's values) before load_only=True pulls in
the actual solved results -- a fresh AIS3_ho_friction_inv.nc shell on its own carries none of
that config, only mesh/geometry (same lesson as every other salvage script this session).
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
STATE_FILE = f'{MODEL_DIR}/deltaT_refit_state.json'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                 'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
TIME_SENTINEL = 1e9
RUN_NAME = 'AIS3_deltaT_refit_g1_r5'
FINAL_GAMMA0 = 300.0


def interp_tf_onto_mesh(tf_path, mesh_x, mesh_y, var='tf'):
    ds = xr.open_dataset(tf_path)
    gx, gy, gz = ds['x'].values, ds['y'].values, ds['z'].values
    tf_full = ds[var].values
    tf_list = []
    for k in range(gz.size):
        interp_k = RegularGridInterpolator((gy, gx), tf_full[k], method='linear',
                                            bounds_error=False, fill_value=np.nan)
        vals = interp_k(np.column_stack([mesh_y, mesh_x]))
        vals = np.nan_to_num(vals, nan=0.0)
        col = np.append(vals, TIME_SENTINEL)
        tf_list.append(col.reshape(-1, 1))
    return tf_list, gz.copy()


with open(STATE_FILE) as f:
    state = json.load(f)
cand = state['candidates']['1']
assert cand['gamma_0'] == FINAL_GAMMA0 and cand['converged'], 'g1 state mismatch -- check deltaT_refit_state.json'
refit_delta_t = np.array(cand['proposal'])
print(f'-- g1 refit deltaT_basin (16 basins): {refit_delta_t}', flush=True)

print('-- Loading HO friction-inverted model (shell)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')

print('-- Configuring ISMIP6 basal melt parameterisation (OFFICIAL IMBIE2 basins)...', flush=True)
md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)
ds_basin = xr.open_dataset(BASIN_FILE)
basin_grid = ds_basin['basinNumber'].values
bx, by = ds_basin['x'].values, ds_basin['y'].values
num_basins = int(np.nanmax(basin_grid)) + 1
elx = np.asarray(md.mesh.elements2d).astype(int) - 1
vx2d = np.asarray(md.mesh.x2d).ravel()
vy2d = np.asarray(md.mesh.y2d).ravel()
ecx = vx2d[elx[:, :3]].mean(axis=1)
ecy = vy2d[elx[:, :3]].mean(axis=1)
bxx, byy = np.meshgrid(bx, by)
basin_lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
basin_id_2d = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)
md.basalforcings.basin_id = pyissm.model.mesh._project_3d(md, vector=basin_id_2d, type='element', layer=0)
md.basalforcings.num_basins = num_basins
md.basalforcings.islocal = 1
md.basalforcings.delta_t = refit_delta_t
md.basalforcings.gamma_0 = FINAL_GAMMA0

print('-- Interpolating Zhou ocean thermal-forcing climatology onto the mesh...', flush=True)
mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()
tf_list, tf_depths = interp_tf_onto_mesh(ZHOU_TF_FILE, mesh_x, mesh_y)
md.basalforcings.tf = tf_list
md.basalforcings.tf_depths = tf_depths

md.inversion.iscontrol = 0
md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
md.transient.ismasstransport = 1
md.masstransport.spcthickness = np.full(md.mesh.numberofvertices, np.nan)
md.smb.mass_balance = np.zeros(md.mesh.numberofvertices)
md.timestepping.start_time = 0
md.timestepping.final_time = 0.01
md.timestepping.time_step = 0.01
md.transient.requested_outputs = ['default', 'BasalforcingsFloatingiceMeltingRate']

print('-- Assigning cluster and updating settings...', flush=True)
md.cluster = pyissm.model.classes.cluster.gadi()
md.cluster.codepath = os.environ['ISSM_DIR'] + '/bin'
md.cluster.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
md.cluster.storage = 'gdata/au88+gdata/vk83+gdata/av17'
md.cluster.moduleuse = ['/g/data/vk83/modules/']
md.cluster.moduleload = ['access-issm_ad/2026.05.0']
md.cluster.login = 'jh7060'
md.cluster.project = 'au88'
md.cluster.queue = 'hugemem'
md.cluster.np = 96
md.cluster.memory = 1450 * 2
md.cluster.time = 60 * 24
md.settings.waitonlock = 0

print(f'-- Loading g1\'s converged refit run ({RUN_NAME})...', flush=True)
md.miscellaneous.name = RUN_NAME
md = pyissm.model.execute.solve(md, 'Transient', load_only=True, runtime_name=False)

melt = np.asarray(md.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
ol = np.asarray(md.mask.ocean_levelset).ravel()
floating = ol < 0
print(f'   melt field loaded: mean={np.nanmean(melt[floating]):.3f} m/yr over '
      f'{floating.sum()} floating vertices, gamma_0={md.basalforcings.gamma_0}', flush=True)

out_path = f'{MODEL_DIR}/AIS3_melt_gamma_tuning.nc'
print(f'-- Saving to {out_path} (supersedes the earlier fixed-deltaT gamma_0=5537.7 version)...', flush=True)
pyissm.model.io.save_model(md, out_path)
print('DONE')
