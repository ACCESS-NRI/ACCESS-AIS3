"""ISMIP7-protocol-informed basal melt calibration (config/ais_0.1.py's melt_gamma_tuning
is a much cruder raw-RMSE-vs-ITS_LIVE sweep; this implements the real ISMIP7 draft protocol's
J1 (present-day basin-aggregated melt fit) + J2 (buttressing-weighted fit) + J3 (warming-
sensitivity fit against ocean-model cold/warm states) objective, with Monte Carlo weight
sampling -- see docs/post_inversion_pipeline_worklog.md for the full comparison against the
ISMIP7_AIS_ocean_focus_group_Protocol_Sept_2025.pdf the user supplied.

KNOWN GAP, explicit not silent: the protocol wants deltaT_basin re-derived per candidate
gamma_0 (each K value should get its own basin corrections fit to match present-day melt
before evaluating the objective). Not implemented here -- doing that properly needs either
many more cluster runs (root-finding deltaT per basin per gamma_0) or a from-scratch
reimplementation of ISSM's internal ismip6 melt formula that can't be verified against the
actual C++ source in this environment. deltaT_basin is held fixed at the published prior
for every ensemble member. Flagged in the output, not hidden.

Two phases, toggled by the PHASE variable below (local to this file, NOT ais_0.1.py's shared
steps/save globals -- deliberately isolated after the race-condition incident with those):
  PHASE = 'submit'  -- run the gamma_0 ensemble (Zhou tf) + 2 ocean-model sensitivity runs
                       (Mathiot cold/warm tf), each saved individually to models/
  PHASE = 'analyze' -- load all saved runs, compute J1/J2/J3, Monte Carlo sample the
                       weights, report 5th/50th/95th percentile gamma_0
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

PHASE = 'analyze'  # 'submit' or 'analyze' -- set by hand before each qsub, see launcher comment

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                 'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
GAMMA0_FILE = '/g/data/au88/ismip6/2300/forcings/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc'
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
BFRN_FILE = f'{ISMIP7_OCEAN}/bfrns/BFRN_ismip8km_v2.nc'
MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'
MATHIOT_COLD_TF = f'{ISMIP7_OCEAN}/ocean_modelling_data/Mathiot_NEMO_cold_v3_TF.nc'
MATHIOT_WARM_TF = f'{ISMIP7_OCEAN}/ocean_modelling_data/Mathiot_NEMO_warm_v3_TF.nc'
MATHIOT_COLD_MELT = f'{ISMIP7_OCEAN}/ocean_modelling_data/Mathiot_NEMO_cold_m.nc'
MATHIOT_WARM_MELT = f'{ISMIP7_OCEAN}/ocean_modelling_data/Mathiot_NEMO_warm_m.nc'
TIME_SENTINEL = 1e9

# Gamma_0 ensemble: wide, log-ish spaced, bracketing both the published prior (11075.45) and
# the much lower values the crude RMSE sweep (melt_gamma_tuning) kept preferring -- basin-
# aggregated Gt/yr fitting is a different metric and may prefer a different region entirely.
GAMMA0_VALUES = [100.0, 300.0, 1000.0, 3000.0, 5537.7, 8306.6, 11075.45, 15000.0, 25000.0]


def interp_tf_onto_mesh(tf_path, mesh_x, mesh_y, var='tf'):
    ds = xr.open_dataset(tf_path)
    gx, gy, gz = ds['x'].values, ds['y'].values, ds['z'].values
    tf_full = ds[var].values  # (z, y, x)
    tf_list = []
    for k in range(gz.size):
        interp_k = RegularGridInterpolator((gy, gx), tf_full[k], method='linear',
                                            bounds_error=False, fill_value=np.nan)
        vals = interp_k(np.column_stack([mesh_y, mesh_x]))
        vals = np.nan_to_num(vals, nan=0.0)
        tf_list.append(np.append(vals, TIME_SENTINEL).reshape(-1, 1))
    return tf_list, gz.copy()


def setup_base_model():
    print('-- Loading HO friction-inverted model...', flush=True)
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')

    print('-- Configuring ISMIP6 basal melt parameterisation (OFFICIAL IMBIE2 basins)...', flush=True)
    md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)

    ds_basin = xr.open_dataset(BASIN_FILE)
    basin_grid = ds_basin['basinNumber'].values  # (y, x), 0-15
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
    basin_id_2d = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)  # 1-indexed for ISSM

    md.basalforcings.basin_id = pyissm.model.mesh._project_3d(md, vector=basin_id_2d, type='element', layer=0)
    md.basalforcings.num_basins = num_basins
    md._basin_id_2d_elements = basin_id_2d  # stashed for the analyze phase -- avoids
    # reverse-engineering the 2D basin assignment back out of the projected 3D array, which
    # would depend on an unverified assumption about ISSM's 3D element ordering.

    # deltaT_basin: held FIXED at the published prior for every ensemble member -- KNOWN GAP,
    # see module docstring. Still needed as a starting correction, just not re-derived per K.
    ds_gamma = xr.open_dataset(GAMMA0_FILE)
    deltaT_grid = ds_gamma['deltaT_basin'].values
    gx, gy = ds_gamma['x'].values, ds_gamma['y'].values
    basin_vals_dt = np.unique(deltaT_grid[np.isfinite(deltaT_grid)])
    # Match published deltaT basins to official IMBIE2 basin_id by nearest basin-mean-location
    # correspondence isn't reliable across two different basin numbering schemes -- instead,
    # directly re-derive per-basin deltaT by nearest-neighbour lookup against the SAME grid,
    # using the official basin_id raster to average deltaT_grid within each official basin.
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

    # Same bugs already found/fixed in ais_0.1.py's melt_gamma_tuning step -- this is a fresh
    # load from AIS3_ho_friction_inv.nc, so they all apply again here.
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

    print(f'-- Submitting gamma_0 ensemble ({len(GAMMA0_VALUES)} runs, Zhou tf)...', flush=True)
    for i, g0 in enumerate(GAMMA0_VALUES):
        mdi = copy.deepcopy(md)
        mdi.basalforcings.gamma_0 = float(g0)
        mdi.miscellaneous.name = f'AIS3_melt_ismip7_g{i}'
        print(f'   run g{i}: gamma_0={g0:.1f}', flush=True)
        pyissm.model.execute.solve(mdi, 'Transient', load_only=False, runtime_name=False)

    print('-- Submitting 2 ocean-model sensitivity runs (Mathiot cold/warm tf, gamma_0=published prior)...', flush=True)
    mesh_x = np.asarray(md.mesh.x).ravel()
    mesh_y = np.asarray(md.mesh.y).ravel()
    for label, tf_path in [('cold', MATHIOT_COLD_TF), ('warm', MATHIOT_WARM_TF)]:
        mdi = copy.deepcopy(md)
        tf_list, tf_depths = interp_tf_onto_mesh(tf_path, mesh_x, mesh_y)
        mdi.basalforcings.tf = tf_list
        mdi.basalforcings.tf_depths = tf_depths
        mdi.basalforcings.gamma_0 = 11075.45  # published prior, sensitivity test only needs one
        mdi.miscellaneous.name = f'AIS3_melt_ismip7_mathiot_{label}'
        print(f'   run mathiot_{label}', flush=True)
        pyissm.model.execute.solve(mdi, 'Transient', load_only=False, runtime_name=False)

    print('DONE submitting. Set PHASE=\'analyze\' and rerun once all jobs finish.')


elif PHASE == 'analyze':
    md, num_basins = setup_base_model()
    mesh_x = np.asarray(md.mesh.x).ravel()
    mesh_y = np.asarray(md.mesh.y).ravel()
    ol = np.asarray(md.mask.ocean_levelset).ravel()
    on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()

    # Per-vertex effective area for Gt/yr integration: distribute each element's base-layer
    # area equally to its 3 base vertices (standard FEM lumped-mass approach).
    print('-- Computing per-vertex effective area for basin-integrated BMB...', flush=True)
    x2d = np.asarray(md.mesh.x2d).ravel()
    y2d = np.asarray(md.mesh.y2d).ravel()
    elx2d = np.asarray(md.mesh.elements2d).astype(int) - 1
    tri_x = x2d[elx2d]
    tri_y = y2d[elx2d]
    elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                              - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
    vertex_area_2d = np.zeros(md.mesh.numberofvertices2d)
    for c in range(3):
        np.add.at(vertex_area_2d, elx2d[:, c], elem_area / 3.0)
    # Use the 2D-element basin assignment stashed directly during setup (md._basin_id_2d_elements)
    # rather than reverse-engineering it out of the projected 3D basalforcings.basin_id array --
    # avoids depending on an unverified assumption about ISSM's 3D element ordering.
    basin_id_2d_elements = md._basin_id_2d_elements  # (numberofelements2d,), 1-indexed basin id
    basin_id_2d_vtx = np.zeros(md.mesh.numberofvertices2d)
    counts = np.zeros(md.mesh.numberofvertices2d)
    for c in range(3):
        np.add.at(basin_id_2d_vtx, elx2d[:, c], basin_id_2d_elements - 1)  # back to 0-indexed
        np.add.at(counts, elx2d[:, c], 1)
    basin_id_2d_vtx = np.round(basin_id_2d_vtx / np.maximum(counts, 1)).astype(int)

    rho_ice = float(md.materials.rho_ice)  # kg/m^3

    def basin_aggregated_bmb(melt_myr_2d, floating_2d):
        """melt_myr_2d: (nv2d,) melt rate in m ice/yr at base layer. Returns Gt/yr per basin."""
        bmb = np.zeros(num_basins)
        mass_flux = melt_myr_2d * rho_ice * vertex_area_2d / 1e12  # kg/m2... -> Gt/yr (1e12 kg = 1 Gt)
        for b in range(num_basins):
            mask = (basin_id_2d_vtx == b) & floating_2d
            bmb[b] = np.nansum(mass_flux[mask])
        return bmb

    floating_2d = ol[on_base] < 0

    print('-- Loading gamma_0 ensemble results...', flush=True)
    ensemble = []
    for i, g0 in enumerate(GAMMA0_VALUES):
        name = f'AIS3_melt_ismip7_g{i}'
        try:
            mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
            mdi.miscellaneous.name = name
            mdi.cluster = md.cluster
            mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
            melt = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
            melt_base = melt[on_base]
            bmb = basin_aggregated_bmb(melt_base, floating_2d)
            ensemble.append({'gamma_0': g0, 'bmb': bmb})
            print(f'   g{i} (gamma_0={g0:.1f}): total BMB = {bmb.sum():.1f} Gt/yr')
        except Exception as e:
            print(f'   g{i} (gamma_0={g0:.1f}): FAILED to load ({e})')

    print('-- Loading J1 target (Paolo/Adusumilli basin-aggregated melt)...', flush=True)
    obs_df = pd.read_csv(MELT_OBS_CSV)
    bmb_obs_mean = obs_df['BMR (Gt/yr)'].values
    bmb_obs_std = obs_df['BMR uncert (Gt/yr)'].values

    print('-- Loading J2 data (BFRN buttressing bins)...', flush=True)
    ds_bfrn = xr.open_dataset(BFRN_FILE)
    bfrn_grid = ds_bfrn['BFRN'].values
    bfrn_bins_grid = ds_bfrn['BFRN_bins'].values
    bfrn_medians = ds_bfrn['BFRN_medians'].values
    bfrn_median = float(ds_bfrn['BFRN_median'].values)
    bx_bfrn, by_bfrn = ds_bfrn['x'].values, ds_bfrn['y'].values
    bxx_b, byy_b = np.meshgrid(bx_bfrn, by_bfrn)
    bin_lookup = NearestNDInterpolator(np.column_stack([bxx_b.ravel(), byy_b.ravel()]), bfrn_bins_grid.ravel())
    vertex_bin_2d = bin_lookup(np.column_stack([x2d, y2d]))
    n_bfrn_bins = int(np.nanmax(bfrn_bins_grid[np.isfinite(bfrn_bins_grid)])) + 1

    # J2 target: same Paolo-based gridded melt product isn't directly available per-vertex here
    # (only basin-aggregated CSV was provided) -- approximate BMB_obs per BFRN bin by
    # distributing each basin's observed total proportionally to that basin's floating area
    # within the bin (best available approximation given what's provided; protocol text
    # doesn't fully specify the per-bin target derivation either -- flagged, not hidden).
    def basin_bin_area_matrix():
        M = np.zeros((num_basins, n_bfrn_bins))
        for b in range(num_basins):
            mb = (basin_id_2d_vtx == b) & floating_2d
            for k in range(n_bfrn_bins):
                mk = mb & (vertex_bin_2d == k)
                M[b, k] = vertex_area_2d[mk].sum()
        return M
    area_bb = basin_bin_area_matrix()
    basin_area_tot = area_bb.sum(axis=1, keepdims=True)
    basin_area_tot[basin_area_tot == 0] = 1.0
    bin_obs_from_basin = (bmb_obs_mean[:, None] * area_bb / basin_area_tot).sum(axis=0)  # Gt/yr per bin

    def bin_aggregated_bmb(melt_myr_2d):
        bmb_bin = np.zeros(n_bfrn_bins)
        mass_flux = melt_myr_2d * rho_ice * vertex_area_2d / 1e12
        for k in range(n_bfrn_bins):
            mask = (vertex_bin_2d == k) & floating_2d
            bmb_bin[k] = np.nansum(mass_flux[mask])
        return bmb_bin

    for member in ensemble:
        name = f"AIS3_melt_ismip7_g{GAMMA0_VALUES.index(member['gamma_0'])}"
        mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
        mdi.miscellaneous.name = name
        mdi.cluster = md.cluster
        mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
        melt = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
        member['bmb_bin'] = bin_aggregated_bmb(melt[on_base])

    print('-- Loading J3 data (Mathiot cold/warm ocean-model sensitivity)...', flush=True)
    j3_data = []
    for label, melt_path in [('cold', MATHIOT_COLD_MELT), ('warm', MATHIOT_WARM_MELT)]:
        try:
            mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
            mdi.miscellaneous.name = f'AIS3_melt_ismip7_mathiot_{label}'
            mdi.cluster = md.cluster
            mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
            melt = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
            bmb_param = basin_aggregated_bmb(melt[on_base], floating_2d)

            ds_ref_melt = xr.open_dataset(MATHIOT_COLD_MELT if label == 'cold' else MATHIOT_WARM_MELT)
            # melt_rate's actual units attr is 'kg/m2/a' (mass flux), NOT m ice/yr -- verified
            # directly against the file (ds_ref_melt['melt_rate'].attrs). basin_aggregated_bmb()
            # expects m ice/yr and multiplies by rho_ice internally, so feeding it the raw
            # kg/m2/a field double-applies the density conversion (~917x inflation) -- this was
            # the root cause of the ref total=1e6+ Gt/yr anomaly. Convert to m ice/yr here first.
            ref_grid = ds_ref_melt['melt_rate'].values / rho_ice  # kg/m2/a -> m ice/yr
            rx, ry = ds_ref_melt['x'].values, ds_ref_melt['y'].values
            rxx, ryy = np.meshgrid(rx, ry)
            ref_lookup = RegularGridInterpolator((ry, rx), np.nan_to_num(ref_grid, nan=0.0),
                                                  method='linear', bounds_error=False, fill_value=0.0)
            ref_melt_at_vtx = ref_lookup(np.column_stack([y2d, x2d]))
            bmb_ref = basin_aggregated_bmb(ref_melt_at_vtx, floating_2d)
            j3_data.append({'label': label, 'bmb_param': bmb_param, 'bmb_ref': bmb_ref})
            print(f'   mathiot_{label}: param total={bmb_param.sum():.1f} Gt/yr, ref total={bmb_ref.sum():.1f} Gt/yr')
        except Exception as e:
            print(f'   mathiot_{label}: FAILED ({e})')

    MATHIOT_REFERENCE_GAMMA0 = 11075.45  # matches the single gamma_0 the cold/warm runs used

    print('-- Computing J1, J2, J3 per ensemble member...', flush=True)
    # J3: the two Mathiot cold/warm forward evaluations were only run once, at the published
    # prior gamma_0 -- re-running them per ensemble member would mean 18 more cluster jobs.
    # Instead, exploit the melt formula's own structure: gamma_0 is a pure multiplicative
    # scalar in the ISMIP6 quadratic-local melt equation (confirmed from the pyissm class
    # definition -- deltaT/TF terms don't depend on gamma_0 at all), so each member's own
    # cold/warm response scales EXACTLY linearly from the single computed run: this is not an
    # approximation of the physics, it's the parameterisation's own exact functional form.
    for member in ensemble:
        member['J1'] = np.nanmean(np.abs(member['bmb'] - bmb_obs_mean))
        weights_bfrn = bfrn_medians[np.clip(np.arange(n_bfrn_bins), 0, len(bfrn_medians) - 1)] / bfrn_median
        member['J2'] = np.nanmean(weights_bfrn * np.abs(member['bmb_bin'] - bin_obs_from_basin))
        if len(j3_data) == 2:
            scale = member['gamma_0'] / MATHIOT_REFERENCE_GAMMA0
            j3_terms = []
            for jd in j3_data:
                bmb_param_scaled = jd['bmb_param'] * scale
                j3_terms.append(np.nanmean(np.abs(bmb_param_scaled - jd['bmb_ref'])))
            member['J3'] = np.mean(j3_terms)
        else:
            member['J3'] = np.nan
        print(f"   gamma_0={member['gamma_0']:.1f}: J1={member['J1']:.3f}  J2={member['J2']:.3f}  J3={member['J3']:.3f}")

    print('-- Monte Carlo sampling objective weights (10000 samples)...', flush=True)
    J1_arr = np.array([m['J1'] for m in ensemble])
    J2_arr = np.array([m['J2'] for m in ensemble])
    J3_arr = np.array([m['J3'] for m in ensemble])
    g0_arr = np.array([m['gamma_0'] for m in ensemble])
    has_j3 = np.all(np.isfinite(J3_arr))

    medJ1, medJ2, medJ3 = np.median(J1_arr), np.median(J2_arr), (np.median(J3_arr) if has_j3 else 1.0)
    N_MC = 10000
    rng = np.random.default_rng(0)
    winners = np.zeros(N_MC)
    for s in range(N_MC):
        if has_j3:
            a = rng.uniform(0, 1, 3)
            a = a / a.sum()
            I = a[0] * J1_arr / medJ1 + a[1] * J2_arr / medJ2 + a[2] * J3_arr / medJ3
        else:
            a = rng.uniform(0, 1, 2)
            a = a / a.sum()
            I = a[0] * J1_arr / medJ1 + a[1] * J2_arr / medJ2
        winners[s] = g0_arr[np.argmin(I)]

    p5, p50, p95 = np.percentile(winners, [5, 50, 95])
    print(f'\n=== ISMIP7-informed calibration result ===')
    print(f'   5th percentile gamma_0:  {p5:.1f} m/yr')
    print(f'   50th percentile gamma_0: {p50:.1f} m/yr')
    print(f'   95th percentile gamma_0: {p95:.1f} m/yr')
    print(f'   (J3 included: {has_j3})')
    print(f'   NOTE: deltaT_basin held fixed at published prior for all ensemble members --')
    print(f'   protocol wants this re-derived per candidate gamma_0, not done here, see module docstring.')
    print('DONE')
