"""Close melt_ismip7_calibration.py's documented gap: re-derive deltaT_basin per candidate
gamma_0 rather than holding it fixed at the published prior. Reopened (2026-09-01) after
melt_deltaT_sensitivity_test.py showed the response is enormous and highly basin-dependent
(a uniform +1 degC shift more than quadrupled total continental melt, 1669.5 -> 7010.3 Gt/yr,
with individual basins swinging 2-14x) -- deltaT_basin turns out to be the dominant lever on
melt magnitude, not gamma_0's comparatively gentle linear scaling, so holding it fixed was a
bigger gap than originally assessed.

Per-basin secant root-finding against the Paolo/Adusumilli basin target, done via REAL ISSM
forward solves (not a from-scratch reimplementation of ISSM's internal melt physics, which
can't be verified against the C++ source here). Exploits two already-established facts to
avoid wasting the compute already spent:
  1. melt scales EXACTLY linearly in gamma_0 for any FIXED deltaT_basin (validated repeatedly
     this session) -- so the already-completed g0-g8 ensemble (real solves at deltaT=published,
     one per gamma_0 candidate) gives an exact bootstrap point for every candidate for free.
  2. The +1 degC sensitivity test (at gamma_0=11075.45) gives an initial per-basin slope
     estimate, rescaled to each candidate's own gamma_0 via the same linearity, used only to
     compute the FIRST Newton step -- every subsequent step uses a real secant between two
     actual solves at that candidate's own gamma_0, no more rescaling.

Basins are independent (deltaT_basin[b] only affects melt within basin b), so all 16 basins x
9 gamma_0 candidates are root-found simultaneously: each "round" is one forward solve per
candidate (not per basin), each of which yields a residual for all 16 basins in that candidate
at once, driving all 16 secant updates for that candidate together.

State persisted to models/deltaT_refit_state.json between rounds (ROUND/STEP hand-toggled
before each qsub, same pattern as PHASE elsewhere in this project).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import copy
import json
import numpy as np
import pandas as pd
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

ROUND = 5     # 0 = bootstrap + submit round 1; N = analyze round N results + submit round N+1
STEP = 'analyze'  # 'submit' or 'analyze' -- hand-toggle before each qsub, same pattern as PHASE elsewhere

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
STATE_FILE = f'{MODEL_DIR}/deltaT_refit_state.json'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                 'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
GAMMA0_FILE = '/g/data/au88/ismip6/2300/forcings/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc'
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'
TIME_SENTINEL = 1e9
GAMMA0_VALUES = [100.0, 300.0, 1000.0, 3000.0, 5537.7, 8306.6, 11075.45, 15000.0, 25000.0]
REFERENCE_GAMMA0 = 11075.45  # the g0-g8 ensemble's and the +1degC test's own gamma_0
MAX_STEP = 3.0        # degC, clip per-iteration secant step to avoid overshoot on the strongly
                       # convex response
CONV_REL_TOL = 0.05   # converged if |resid| < 5% of obs, OR
CONV_ABS_TOL = 1.0    # |resid| < 1 Gt/yr (for small-obs basins where relative tol is too strict)
MAX_ROUNDS = 5


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
    """Copied verbatim from melt_ismip7_calibration.py's setup_base_model."""
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

    return md, num_basins, delta_t_per_basin


def basin_geometry(md, num_basins):
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

    return on_base, basin_bmb


def load_melt(name, cluster):
    mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
    mdi.miscellaneous.name = name
    mdi.cluster = cluster
    mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
    return np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()


if ROUND == 0 and STEP == 'submit':
    md, num_basins, published_delta_t = setup_base_model()
    on_base, basin_bmb = basin_geometry(md, num_basins)
    obs_df = pd.read_csv(MELT_OBS_CSV)
    bmb_obs = obs_df['BMR (Gt/yr)'].values

    print('-- Loading g0-g8 ensemble (real solves at published deltaT_basin)...', flush=True)
    bmb0_per_candidate = {}
    for i, g0 in enumerate(GAMMA0_VALUES):
        melt = load_melt(f'AIS3_melt_ismip7_g{i}', md.cluster)
        bmb0_per_candidate[i] = basin_bmb(melt[on_base]).tolist()
        print(f'   g{i} (gamma_0={g0:.1f}): total bmb = {sum(bmb0_per_candidate[i]):.1f} Gt/yr')

    print('-- Loading +1degC sensitivity test (gamma_0=11075.45)...', flush=True)
    melt_base_ref = load_melt('AIS3_melt_ismip7_g6', md.cluster)
    melt_shift_ref = load_melt('AIS3_melt_deltaT_test_plus1', md.cluster)
    bmb_base_ref = basin_bmb(melt_base_ref[on_base])
    bmb_shift_ref = basin_bmb(melt_shift_ref[on_base])
    slope_ref = (bmb_shift_ref - bmb_base_ref) / 1.0  # Gt/yr per degC, at gamma_0=11075.45

    print('\n-- Bootstrapping per-basin Newton proposal for round 1, all 9 candidates...', flush=True)
    state = {'published_delta_t': published_delta_t.tolist(), 'bmb_obs': bmb_obs.tolist(),
             'candidates': {}}
    md.cluster.queue = 'hugemem'  # reuse the same cluster config object for every submit below
    for i, g0 in enumerate(GAMMA0_VALUES):
        scale = g0 / REFERENCE_GAMMA0
        slope_i = slope_ref * scale  # linear-in-gamma_0 rescale of the estimated local slope
        bmb0 = np.array(bmb0_per_candidate[i])
        step = (bmb_obs - bmb0) / np.where(np.abs(slope_i) < 1e-6, np.nan, slope_i)
        step = np.nan_to_num(step, nan=0.0)
        step = np.clip(step, -MAX_STEP, MAX_STEP)
        delta_t_1 = published_delta_t + step
        state['candidates'][str(i)] = {
            'gamma_0': g0,
            'history': [
                {'delta_t': published_delta_t.tolist(), 'bmb': bmb0.tolist()},
            ],
            'proposal': delta_t_1.tolist(),
            'converged': False,
        }
        print(f'   g{i} (gamma_0={g0:.1f}): proposed step (deg C) min={step.min():.3f} '
              f'max={step.max():.3f} mean={step.mean():.3f}')

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=1)
    print(f'\n-- State saved to {STATE_FILE}. Submitting round 1 ({len(GAMMA0_VALUES)} solves)...', flush=True)

    for i, g0 in enumerate(GAMMA0_VALUES):
        mdi = copy.deepcopy(md)
        mdi.basalforcings.gamma_0 = float(g0)
        mdi.basalforcings.delta_t = np.array(state['candidates'][str(i)]['proposal'])
        mdi.miscellaneous.name = f'AIS3_deltaT_refit_g{i}_r1'
        print(f'   submitting g{i} round 1', flush=True)
        pyissm.model.execute.solve(mdi, 'Transient', load_only=False, runtime_name=False)
    print('DONE round 0. Set ROUND=1, STEP=\'analyze\' and rerun once round 1 jobs finish.')

elif STEP == 'analyze':
    md, num_basins, published_delta_t = setup_base_model()
    on_base, basin_bmb = basin_geometry(md, num_basins)

    with open(STATE_FILE) as f:
        state = json.load(f)
    bmb_obs = np.array(state['bmb_obs'])

    print(f'-- Loading round {ROUND} results for all candidates...', flush=True)
    any_unconverged = False
    for i_str, cand in state['candidates'].items():
        i = int(i_str)
        g0 = cand['gamma_0']
        if cand['converged']:
            print(f'   g{i} (gamma_0={g0:.1f}): already converged, skipping')
            continue
        name = f'AIS3_deltaT_refit_g{i}_r{ROUND}'
        melt = load_melt(name, md.cluster)
        bmb_new = basin_bmb(melt[on_base])
        delta_t_new = np.array(cand['proposal'])
        cand['history'].append({'delta_t': delta_t_new.tolist(), 'bmb': bmb_new.tolist()})

        resid = bmb_new - bmb_obs
        rel = np.abs(resid) / np.maximum(np.abs(bmb_obs), 1e-6)
        converged_mask = (rel < CONV_REL_TOL) | (np.abs(resid) < CONV_ABS_TOL)
        print(f'   g{i} (gamma_0={g0:.1f}): total bmb={bmb_new.sum():.1f} Gt/yr '
              f'(target {bmb_obs.sum():.1f}), max|resid|={np.abs(resid).max():.2f} Gt/yr, '
              f'{converged_mask.sum()}/{num_basins} basins converged')

        if converged_mask.all():
            cand['converged'] = True
            print(f'      -> g{i} FULLY CONVERGED after round {ROUND}')
            continue
        any_unconverged = True

        # Real secant using the two most recent history points (per basin).
        h = cand['history']
        x0, f0 = np.array(h[-2]['delta_t']), np.array(h[-2]['bmb'])
        x1, f1 = np.array(h[-1]['delta_t']), np.array(h[-1]['bmb'])
        denom = f1 - f0
        step = np.where(np.abs(denom) < 1e-6, 0.0, -(f1 - bmb_obs) * (x1 - x0) / np.where(denom == 0, np.nan, denom))
        step = np.nan_to_num(step, nan=0.0)
        step = np.clip(step, -MAX_STEP, MAX_STEP)
        # Only update basins that haven't converged yet -- leave converged basins' delta_t fixed.
        delta_t_next = np.where(converged_mask, x1, x1 + step)
        cand['proposal'] = delta_t_next.tolist()

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=1)

    if not any_unconverged:
        print(f'\n=== ALL CANDIDATES CONVERGED after round {ROUND} ===')
        print('DONE. Run melt_deltaT_refit_recalibrate.py next to recompute J1/J2/J3 with the refit deltaT_basin.')
    elif ROUND >= MAX_ROUNDS:
        print(f'\n=== HIT MAX_ROUNDS={MAX_ROUNDS}, stopping (some basins may not be fully converged) ===')
        print('DONE (incomplete). Review per-basin residuals above before proceeding.')
    else:
        next_round = ROUND + 1
        print(f'\n-- Submitting round {next_round} for unconverged candidates...', flush=True)
        for i_str, cand in state['candidates'].items():
            if cand['converged']:
                continue
            i = int(i_str)
            g0 = cand['gamma_0']
            mdi = copy.deepcopy(md)
            mdi.basalforcings.gamma_0 = float(g0)
            mdi.basalforcings.delta_t = np.array(cand['proposal'])
            mdi.miscellaneous.name = f'AIS3_deltaT_refit_g{i}_r{next_round}'
            print(f'   submitting g{i} round {next_round}', flush=True)
            pyissm.model.execute.solve(mdi, 'Transient', load_only=False, runtime_name=False)
        print(f'DONE round {ROUND}. Set ROUND={next_round}, STEP=\'analyze\' and rerun once jobs finish.')
