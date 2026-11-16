"""Pilot sanity check before building the full per-basin deltaT_basin re-derivation loop
(closing melt_ismip7_calibration.py's documented gap: the protocol wants deltaT_basin
re-fit per candidate gamma_0, currently held fixed at the published prior).

Runs ONE extra forward solve at the published gamma_0=11075.45 with deltaT_basin shifted
+1.0 degC uniformly across all 16 basins, and compares basin-aggregated BMB against the
already-completed g6 run (execution/AIS3_melt_ismip7_g6, same gamma_0, published deltaT_basin,
Zhou tf) to confirm the sign and magnitude of d(BMB)/d(deltaT_basin) empirically against real
ISSM output, before committing to a multi-iteration secant root-finding loop across all 9
gamma_0 candidates (which the math -- melt ~ D*|D| with D=TF-deltaT_basin is a strictly
monotonically increasing odd function of D -- guarantees is monotonic in deltaT_basin, just
need to confirm direction/scale against the real ISSM implementation, not just the textbook
formula, since sign conventions are exactly the kind of thing this project has been burned by
before).
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
DELTA_SHIFT = 1.0  # degC, uniform test perturbation


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


def setup_base_model():
    """Copied verbatim from melt_ismip7_calibration.py's setup_base_model -- see that file."""
    print('-- Loading HO friction-inverted model...', flush=True)
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')

    print('-- Configuring ISMIP6 basal melt parameterisation (OFFICIAL IMBIE2 basins)...', flush=True)
    md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)

    ds_basin = xr.open_dataset(BASIN_FILE)
    basin_grid = ds_basin['basinNumber'].values
    bx, by = ds_basin['x'].values, ds_basin['y'].values
    num_basins = int(np.nanmax(basin_grid)) + 1
    print(f'   {num_basins} basins found in official IMBIE2 grid (expect 16)')

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
    md._basin_id_2d_elements = basin_id_2d

    ds_gamma = xr.open_dataset(GAMMA0_FILE)
    deltaT_grid = ds_gamma['deltaT_basin'].values
    gx, gy = ds_gamma['x'].values, ds_gamma['y'].values
    basin_vals_dt = np.unique(deltaT_grid[np.isfinite(deltaT_grid)])
    gxx, gyy = np.meshgrid(gx, gy)
    dT_lookup = NearestNDInterpolator(np.column_stack([gxx.ravel(), gyy.ravel()]), deltaT_grid.ravel())
    dT_at_basin_grid = dT_lookup(np.column_stack([bxx.ravel(), byy.ravel()])).reshape(basin_grid.shape)
    delta_t_per_basin = np.array([
        np.nanmean(dT_at_basin_grid[basin_grid == b]) if np.any(basin_grid == b) else 0.0
        for b in range(num_basins)
    ])
    delta_t_per_basin = np.nan_to_num(delta_t_per_basin, nan=0.0)
    md.basalforcings.delta_t = delta_t_per_basin
    md.basalforcings.islocal = 1

    print('-- Interpolating Zhou ocean thermal-forcing climatology onto the mesh...', flush=True)
    mesh_x = np.asarray(md.mesh.x).ravel()
    mesh_y = np.asarray(md.mesh.y).ravel()
    tf_list, tf_depths = interp_tf_onto_mesh(ZHOU_TF_FILE, mesh_x, mesh_y)
    md.basalforcings.tf = tf_list
    md.basalforcings.tf_depths = tf_depths

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

    md.inversion.iscontrol = 0
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.ismasstransport = 1
    md.masstransport.spcthickness = np.full(md.mesh.numberofvertices, np.nan)
    md.smb.mass_balance = np.zeros(md.mesh.numberofvertices)
    md.timestepping.start_time = 0
    md.timestepping.final_time = 0.01
    md.timestepping.time_step = 0.01
    md.transient.requested_outputs = ['default', 'BasalforcingsFloatingiceMeltingRate']
    md.settings.waitonlock = 0

    return md, num_basins


if PHASE == 'submit':
    md, num_basins = setup_base_model()
    md.basalforcings.gamma_0 = PUBLISHED_GAMMA0
    md.basalforcings.delta_t = md.basalforcings.delta_t + DELTA_SHIFT
    md.miscellaneous.name = 'AIS3_melt_deltaT_test_plus1'
    print(f'-- Submitting sensitivity-test run: gamma_0={PUBLISHED_GAMMA0}, '
          f'deltaT_basin shifted +{DELTA_SHIFT} degC uniformly...', flush=True)
    pyissm.model.execute.solve(md, 'Transient', load_only=False, runtime_name=False)
    print('DONE submitting. Set PHASE=\'analyze\' and rerun once the job finishes.')

elif PHASE == 'analyze':
    md, num_basins = setup_base_model()
    on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
    x2d = np.asarray(md.mesh.x2d).ravel()
    y2d = np.asarray(md.mesh.y2d).ravel()
    elx2d = np.asarray(md.mesh.elements2d).astype(int) - 1
    tri_x, tri_y = x2d[elx2d], y2d[elx2d]
    elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                              - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
    vertex_area_2d = np.zeros(md.mesh.numberofvertices2d)
    for c in range(3):
        np.add.at(vertex_area_2d, elx2d[:, c], elem_area / 3.0)

    basin_id_2d_elements = md._basin_id_2d_elements
    counts = np.zeros(md.mesh.numberofvertices2d)
    basin_id_2d_vtx = np.zeros(md.mesh.numberofvertices2d)
    for c in range(3):
        np.add.at(basin_id_2d_vtx, elx2d[:, c], basin_id_2d_elements - 1)
        np.add.at(counts, elx2d[:, c], 1)
    basin_id_2d_vtx = np.round(basin_id_2d_vtx / np.maximum(counts, 1)).astype(int)

    ol = np.asarray(md.mask.ocean_levelset).ravel()
    floating_2d = ol[on_base] < 0
    rho_ice = float(md.materials.rho_ice)

    def basin_bmb(melt_myr_2d):
        mass_flux = melt_myr_2d * rho_ice * vertex_area_2d / 1e12
        bmb = np.zeros(num_basins)
        for b in range(num_basins):
            mask = (basin_id_2d_vtx == b) & floating_2d
            bmb[b] = np.nansum(mass_flux[mask])
        return bmb

    def load_run(name):
        mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
        mdi.miscellaneous.name = name
        mdi.cluster = md.cluster
        mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
        melt = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
        return melt[on_base]

    print('-- Loading baseline (g6, gamma_0=11075.45, published deltaT_basin)...', flush=True)
    melt_base = load_run('AIS3_melt_ismip7_g6')
    bmb_base = basin_bmb(melt_base)

    print('-- Loading sensitivity-test run (deltaT_basin +1 degC)...', flush=True)
    melt_shift = load_run('AIS3_melt_deltaT_test_plus1')
    bmb_shift = basin_bmb(melt_shift)

    obs_df = pd.read_csv(MELT_OBS_CSV)
    bmb_obs = obs_df['BMR (Gt/yr)'].values

    print(f'\n{"basin":>5} {"obs":>10} {"bmb_base":>10} {"bmb_+1C":>10} {"d(bmb)/dT":>12}')
    for b in range(num_basins):
        d_bmb = bmb_shift[b] - bmb_base[b]
        print(f'{b:5d} {bmb_obs[b]:10.2f} {bmb_base[b]:10.2f} {bmb_shift[b]:10.2f} {d_bmb / DELTA_SHIFT:12.3f}')

    total_base, total_shift = bmb_base.sum(), bmb_shift.sum()
    print(f'\nTotal: base={total_base:.1f} Gt/yr, +1degC={total_shift:.1f} Gt/yr, '
          f'sensitivity={((total_shift - total_base) / DELTA_SHIFT):.2f} Gt/yr per degC')
    print(f'Sign check: melt {"DECREASES" if total_shift < total_base else "INCREASES"} '
          f'when deltaT_basin increases (expect DECREASES if deltaT subtracts from TF)')
    print('DONE')
