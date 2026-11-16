"""SSA-mesh version of melt_deltaT_sensitivity_test.py -- bootstrap per-basin slope estimate
(+1 degC uniform deltaT_basin shift at the published gamma_0=11075.45) needed to seed the
first Newton step of ssa_melt_deltaT_basin_refit.py's secant loop. Mirrors the HO version's
method and sign-convention findings (increasing deltaT_basin increases melt -- confirmed on
HO, expected to hold here too since it's the same underlying ISSM melt formula, but verified
independently rather than assumed).

Plain `cluster` config (48 cores/190GB/normal) -- SSA mesh, no hugemem needed. Includes the
TransientSolution-stripping memory fix found necessary for melt_gamma_tuning_ssa.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pandas as pd
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

PHASE = 'analyze'  # 'submit' or 'analyze' -- hand-toggle before each qsub

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                 'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
GAMMA0_FILE = '/g/data/au88/ismip6/2300/forcings/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc'
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'
TIME_SENTINEL = 1e9
PUBLISHED_GAMMA0 = 11075.45
DELTA_SHIFT = 1.0

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


if PHASE == 'submit':
    md, num_basins, published_delta_t = setup_base_model()
    md.basalforcings.gamma_0 = PUBLISHED_GAMMA0
    md.basalforcings.delta_t = published_delta_t + DELTA_SHIFT
    md.miscellaneous.name = 'AIS3_melt_deltaT_test_plus1_ssa'
    print(f'-- Submitting SSA sensitivity-test run: gamma_0={PUBLISHED_GAMMA0}, '
          f'deltaT_basin shifted +{DELTA_SHIFT} degC uniformly...', flush=True)
    pyissm.model.execute.solve(md, 'Transient', load_only=False, runtime_name=False)
    print('DONE submitting. Set PHASE=\'analyze\' and rerun once the job finishes.')

elif PHASE == 'analyze':
    md, num_basins, published_delta_t = setup_base_model()
    basin_bmb = basin_geometry(md, num_basins)
    obs_df = pd.read_csv(MELT_OBS_CSV)
    bmb_obs = obs_df['BMR (Gt/yr)'].values

    print('-- Loading baseline (g6, gamma_0=11075.45, published deltaT_basin)...', flush=True)
    melt_base = load_melt('AIS3_melt_gamma_tuning_ssa_g6')
    bmb_base = basin_bmb(melt_base)

    print('-- Loading +1degC sensitivity test...', flush=True)
    melt_shift = load_melt('AIS3_melt_deltaT_test_plus1_ssa')
    bmb_shift = basin_bmb(melt_shift)

    print(f'\n{"basin":>5} {"obs":>10} {"bmb_base":>10} {"bmb_+1C":>10} {"d(bmb)/dT":>12}')
    for b in range(num_basins):
        d_bmb = bmb_shift[b] - bmb_base[b]
        print(f'{b:5d} {bmb_obs[b]:10.2f} {bmb_base[b]:10.2f} {bmb_shift[b]:10.2f} {d_bmb / DELTA_SHIFT:12.3f}')

    total_base, total_shift = bmb_base.sum(), bmb_shift.sum()
    print(f'\nTotal: base={total_base:.1f} Gt/yr, +1degC={total_shift:.1f} Gt/yr, '
          f'sensitivity={((total_shift - total_base) / DELTA_SHIFT):.2f} Gt/yr per degC')
    print('DONE')
