"""SSA gamma_0 ensemble: one short forward melt solve per candidate gamma_0, all at the
PUBLISHED deltaT_basin. This is the starting data for the SSA melt calibration chain:

  melt_gamma_sweep_ssa.py            -> execution_SSA/AIS3_melt_gamma_tuning_ssa_g{0..8}
  melt_deltaT_sensitivity_test_ssa.py  (+1 degC run; compared against g6)
  ssa_melt_deltaT_basin_refit.py       (round 0 bootstraps from g0-g8 and g6)
  ssa_melt_deltaT_refit_recalibrate.py (picks gamma_0 by J2)
  -> copy the winning gamma_0/deltaT_basin into melt_gamma_tuning_ssa in ais_0.1_SSA.py

Moved out of ais_0.1_SSA.py's melt_gamma_tuning_ssa step (2026-09-29), which now only applies
the validated result. Run names are unchanged, so the downstream scripts read them as before.
The step's old "best by J1" pick is not saved any more: with deltaT_basin held at the published
prior it is superseded by the refit (worklog 4c, 4n). The analyze phase only prints it.

PHASE (env var): 'submit' submits the 9 solves (waitonlock=0, returns once submitted) and
refuses if any g{i} execution dir already exists -- archive the old ones first (mv -n);
'analyze' prints per-candidate basin melt totals and J1 against Paolo/Adusumilli.
    qsub -v PHASE=submit launch_melt_gamma_sweep_ssa.pbs   (then PHASE=analyze)
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import copy
import numpy as np
import pandas as pd
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

PHASE = os.environ.get('PHASE', 'analyze')

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXECUTION_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                 'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
GAMMA0_FILE = '/g/data/au88/ismip6/2300/forcings/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc'
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'
TIME_SENTINEL = 1e9
# Same grid as the HO ensemble (melt_ismip7_calibration.py); index 6 is the published 11075.45,
# which the sensitivity test and refit use as their reference run (g6).
GAMMA0_VALUES = [100.0, 300.0, 1000.0, 3000.0, 5537.7, 8306.6, 11075.45, 15000.0, 25000.0]

cluster = pyissm.model.classes.cluster.gadi()
cluster.codepath = os.environ['ISSM_DIR'] + '/bin'
cluster.executionpath = EXECUTION_DIR
cluster.storage = 'gdata/au88+gdata/vk83+gdata/av17'
cluster.moduleuse = ['/g/data/vk83/modules/']
cluster.moduleload = ['access-issm_ad/2026.05.0']
cluster.np = 48
cluster.memory = 190
cluster.time = 60 * 48
cluster.login = 'jh7060'
cluster.project = 'au88'


def run_name(i):
    return f'AIS3_melt_gamma_tuning_ssa_g{i}'


def setup_base_model():
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_relaxed.nc')
    md.results.TransientSolution = []  # memory fix, see melt_gamma_tuning_ssa

    md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)
    ds_basin = xr.open_dataset(BASIN_FILE)
    basin_grid = ds_basin['basinNumber'].values
    bx, by = ds_basin['x'].values, ds_basin['y'].values
    num_basins = int(np.nanmax(basin_grid)) + 1

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

    ds_gamma = xr.open_dataset(GAMMA0_FILE)
    deltaT_grid = ds_gamma['deltaT_basin'].values
    gx, gy = ds_gamma['x'].values, ds_gamma['y'].values
    gxx, gyy = np.meshgrid(gx, gy)
    dT_lookup = NearestNDInterpolator(np.column_stack([gxx.ravel(), gyy.ravel()]), deltaT_grid.ravel())
    dT_at_basin_grid = dT_lookup(np.column_stack([bxx.ravel(), byy.ravel()])).reshape(basin_grid.shape)
    delta_t_per_basin = np.array([
        np.nanmean(dT_at_basin_grid[basin_grid == b]) if np.any(basin_grid == b) else 0.0
        for b in range(num_basins)
    ])
    md.basalforcings.delta_t = np.nan_to_num(delta_t_per_basin, nan=0.0)
    md.basalforcings.islocal = 1

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
    md.settings.waitonlock = 0

    return md, num_basins, delta_t_per_basin


def basin_geometry(md, num_basins):
    mesh_x = np.asarray(md.mesh.x).ravel()
    mesh_y = np.asarray(md.mesh.y).ravel()
    elx = np.asarray(md.mesh.elements).astype(int) - 1
    tri_x, tri_y = mesh_x[elx], mesh_y[elx]
    elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                              - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
    vertex_area = np.zeros(md.mesh.numberofvertices)
    for c in range(3):
        np.add.at(vertex_area, elx[:, c], elem_area / 3.0)
    ds_basin = xr.open_dataset(BASIN_FILE)
    basin_grid = ds_basin['basinNumber'].values
    bx, by = ds_basin['x'].values, ds_basin['y'].values
    bxx, byy = np.meshgrid(bx, by)
    basin_lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
    ecx = mesh_x[elx[:, :3]].mean(axis=1)
    ecy = mesh_y[elx[:, :3]].mean(axis=1)
    basin_id_elements = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)
    counts = np.zeros(md.mesh.numberofvertices)
    basin_id_vtx = np.zeros(md.mesh.numberofvertices)
    for c in range(3):
        np.add.at(basin_id_vtx, elx[:, c], basin_id_elements - 1)
        np.add.at(counts, elx[:, c], 1)
    basin_id_vtx = np.round(basin_id_vtx / np.maximum(counts, 1)).astype(int)
    ol = np.asarray(md.mask.ocean_levelset).ravel()
    floating = ol < 0
    rho_ice = float(md.materials.rho_ice)

    def basin_bmb(melt_myr):
        mass_flux = melt_myr * rho_ice * vertex_area / 1e12
        bmb = np.zeros(num_basins)
        for b in range(num_basins):
            mask = (basin_id_vtx == b) & floating
            bmb[b] = np.nansum(mass_flux[mask])
        return bmb

    return basin_bmb


def load_melt(name):
    mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_relaxed.nc')
    mdi.results.TransientSolution = []
    mdi.miscellaneous.name = name
    mdi.cluster = cluster
    mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
    return np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()


print(f'PHASE={PHASE}', flush=True)
if PHASE == 'submit':
    existing = [run_name(i) for i in range(len(GAMMA0_VALUES))
                if os.path.exists(f'{EXECUTION_DIR}/{run_name(i)}')]
    if existing:
        raise FileExistsError(f'execution dirs already exist, archive them first (mv -n): {existing}')
    md, num_basins, published_delta_t = setup_base_model()
    print(f'-- Submitting gamma_0 ensemble ({len(GAMMA0_VALUES)} runs, published deltaT_basin)...', flush=True)
    for i, g0 in enumerate(GAMMA0_VALUES):
        mdi = copy.deepcopy(md)
        mdi.basalforcings.gamma_0 = float(g0)
        mdi.miscellaneous.name = run_name(i)
        print(f'   g{i}: gamma_0={g0:.1f}', flush=True)
        pyissm.model.execute.solve(mdi, 'Transient', load_only=False, runtime_name=False)
    print("DONE submitting. Rerun with PHASE=analyze once the jobs finish.")

elif PHASE == 'analyze':
    md, num_basins, published_delta_t = setup_base_model()
    basin_bmb = basin_geometry(md, num_basins)
    bmb_obs = pd.read_csv(MELT_OBS_CSV)['BMR (Gt/yr)'].values
    print(f'-- Basin melt vs Paolo/Adusumilli (target total {bmb_obs.sum():.1f} Gt/yr)...', flush=True)
    for i, g0 in enumerate(GAMMA0_VALUES):
        try:
            bmb = basin_bmb(load_melt(run_name(i)))
        except Exception as e:
            print(f'   g{i} (gamma_0={g0:.1f}): FAILED to load ({e})')
            continue
        print(f'   g{i} (gamma_0={g0:.1f}): total bmb={bmb.sum():.1f} Gt/yr, '
              f'J1={np.nanmean(np.abs(bmb - bmb_obs)):.3f}')

else:
    raise ValueError(f"PHASE must be 'submit' or 'analyze', got {PHASE!r}")
