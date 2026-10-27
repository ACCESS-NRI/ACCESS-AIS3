"""ACCESS-AIS3 pure-SSA production pipeline (Budd/Weertman friction law, p=q=1 -- the best
validated RMSE anywhere in this project, 60.4 m/yr, vs. the Schoof-based HO track's current
94-99 m/yr after several chunked-restart iterations). See docs/inversion_worklog.md for the
full history of how p=q=1 was found, and the "in your opinion" exchange in this session's
transcript for why a dedicated SSA production track is worth having alongside the HO one:
SSA is dramatically cheaper and already converged/stable, at the cost of no vertical-shear
physics (so it can't do thermal-coupled work the way the HO track can).

Deliberately self-contained -- does NOT import or exec ais_0.1.py, and does NOT touch its
shared `steps`/`save` globals. This project hit two real race-condition incidents this
session from toggling those globals while a job was mid-flight; a fully separate file with
its own local `steps`/`save` sidesteps that risk entirely, at the cost of some code
duplication (the handful of helper functions and Budd-specific step logic below are copied
from ais_0.1.py, not imported, so the two files can diverge safely and be run concurrently).

Friction source: models/AIS3_ssa_friction_inv_reg_lcurve/run_001_10_100_0.0001/ -- the real,
production-canonical location the RMSE=60.4 p=q=1 result was salvaged into (see
config/finalize_p1q1_reg_lcurve.py) from the ad-hoc script that originally produced it
(p1q1_reg_sweep2.py, no longer present as a file, only as raw execution output). This
finalization step recovers the ACTUAL computed result rather than re-deriving it from
scratch, which would otherwise require regenerating AIS3_param.nc with Budd config (it
currently has Schoof baked in, for the HO track) and re-running two expensive inversion
stages. Run finalize_p1q1_reg_lcurve.py once before this script's first stage.

Stage status (updated 2026-09-02):
  1. ssa_inverted_solve_budd -- FULLY IMPLEMENTED AND RUN (RMSE=65.67 m/yr full-continental,
     the best full-continental result in the project). AIS3_SSA_inverted.nc.
  2. ssa_relaxation_budd -- FULLY IMPLEMENTED AND RUN (mean |dH|=32.90m, max |dH|=1505.39m
     over the 20yr relaxation). AIS3_SSA_relaxed.nc.
  3. melt_gamma_tuning_ssa -- builds AIS3_melt_final_ssa.nc directly from the validated
     calibration (gamma_0=300 + per-basin refit deltaT_basin, hard-coded; 2026-09-29). The
     values come from the standalone refit chain (ssa_melt_deltaT_basin_refit.py /
     ssa_melt_deltaT_refit_recalibrate.py). The existing AIS3_melt_final_ssa.nc was built by
     finalize_ssa_melt_calibration.py. The step's old coarse gamma_0 sweep (pick 5537.7,
     AIS3_melt_gamma_tuning_ssa.nc) was superseded by that refit and removed.
  4. historical_dhdt_tuning_ssa -- RUN AND VALIDATED. Full 1995-2019 (24yr) transient against
     AIS3_melt_final_ssa.nc, checked against MIPKIT's dhdt_cpom: grounded RMSE=1.402 m/yr,
     independently reconfirmed via check_historical_ssa.py. AIS3_historical_1995_2019_SSA.nc.
  5. projection_ssp_ssa -- IMPLEMENTED (2026-09-29): anomaly forcing on the calibrated
     baseline (TF0 = Zhou, SMB0 = RACMO 1995-2014), CTRL/SSP/attribution/melt variants chosen
     per job via qsub -v, self-chaining 50-yr chunks to 2300. See launch_projection_ssa.pbs and
     docs/projection_experiments.md.

Both new stages use the plain `cluster` (48 cores/190GB/normal) config, not hugemem -- the
SSA mesh is ~15x smaller than the HO track's (no vertical layers), and both stages use the
same synchronous submit-and-wait pattern (`waitonlock` minutes + `load_only=False` in one
call) already validated throughout this pipeline, avoiding the "never actually submits" bug
found and fixed in ais_0.1.py's own ho_relaxation/historical_dhdt_tuning on their first real
run.
"""
import pyissm
import ccdtools as ccdtools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import glob
from pathlib import Path
from scipy.spatial import Delaunay

os.chdir('/g/data/au88/jh7060/ACCESS-AIS3/')
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'

model_dir = '/g/data/au88/jh7060/ACCESS-AIS3/models'
execution_dir = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'

save = True
diagnostics = True
plot = True

cluster = pyissm.model.classes.cluster.gadi()
cluster.codepath = os.environ['ISSM_DIR'] + '/bin'
cluster.executionpath = execution_dir
cluster.storage = 'gdata/au88+gdata/vk83+gdata/av17'
cluster.moduleuse = ['/g/data/vk83/modules/']
cluster.moduleload = ['access-issm_ad/2026.05.0']
cluster.np = 48
cluster.memory = 190
cluster.time = 60 * 48
cluster.login = 'jh7060'
cluster.project = 'au88'

FRICTION_RUN_DIR = f'{model_dir}/AIS3_ssa_friction_inv_reg_lcurve/run_001_10_100_0.0001'
FRICTION_RUN_NAME = 'run_001_10_100_0.0001'

all_steps = ['ssa_inverted_solve_budd', 'ssa_relaxation_budd', 'melt_gamma_tuning_ssa', 'historical_dhdt_tuning_ssa', 'projection_ssp_ssa']
steps = ['historical_dhdt_tuning_ssa']  # edit before each qsub -- see launch_ais_0.1_SSA.pbs
# Per-job override via `qsub -v AIS3_STEPS=a,b` (same as ais_0.1.py) -- lets a launcher pick its
# step without editing the shared `steps` above.
if os.environ.get('AIS3_STEPS'):
    steps = os.environ['AIS3_STEPS'].split(',')


def _interp_weights(src_x, src_y, tgt_x, tgt_y):
    """Precompute a reusable linear-interpolation transform (Delaunay triangulation +
    barycentric weights) ONCE for a fixed (source-grid, target-mesh) pair, so many
    different value arrays defined on that same source grid can each be interpolated in
    milliseconds afterward via _interp_apply.

    Needed for projection_ssp_ssa's time-varying ocean thermal forcing: 30 depths x ~81
    years means ~2430 interpolations of the SAME 761x761 source grid onto the SAME mesh --
    pyissm.data.interp.points_to_mesh (scipy.interpolate.griddata under the hood) rebuilds
    the full Delaunay triangulation AND redoes point-location for every single call, which
    is fine for this pipeline's existing handful-of-calls loops (RACMO's 24 annual SMB
    calls) but empirically verified to cost ~13-20s PER CALL even with triangulation reuse
    alone (evaluation/point-location dominates, not triangulation) -- at 2430 calls that is
    hours just for TF preprocessing, dwarfing the actual transient solve. Precomputing the
    per-target-point simplex + barycentric weights ONCE instead drops each subsequent
    interpolation to ~10ms (verified empirically), since applying already-known weights to
    a new value array is just a weighted sum, not a fresh geometric search.
    """
    src_xy = np.column_stack([np.asarray(src_x).ravel(), np.asarray(src_y).ravel()])
    tgt_xy = np.column_stack([np.asarray(tgt_x).ravel(), np.asarray(tgt_y).ravel()])
    tri = Delaunay(src_xy)
    simplex = tri.find_simplex(tgt_xy)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = tgt_xy - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    return vertices, wts, simplex


def _interp_apply(values, vtx, wts, simplex, fill_value=np.nan):
    """Apply a transform from _interp_weights to a new value array defined on the same
    source grid it was built from -- same interpolation this pipeline uses everywhere else
    (linear, NaN outside the source grid's convex hull), just without repeating the
    triangulation/point-location work each time."""
    values = np.asarray(values).ravel()
    out = np.einsum('nj,nj->n', values[vtx], wts)
    out[simplex == -1] = fill_value
    return out


def load_shelf_rheology_B():
    """Identical to ais_0.1.py's own helper -- copied, not imported, deliberately (see
    module docstring). The validated floating-shelf rheology result, independent of
    friction law entirely."""
    _cl = pyissm.model.classes.cluster.gadi()
    _cl.codepath = os.environ['ISSM_DIR'] + '/bin'
    _cl.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution_newB_rheology'
    _cl.login = 'jh7060'; _cl.project = 'au88'; _cl.storage = 'gdata/au88'
    mshelf = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')
    mshelf.mask.ice_levelset = pyissm.model.param.kill_icebergs(mshelf)
    sel = (mshelf.mask.ocean_levelset < 0) & (mshelf.mask.ice_levelset < 0)
    mshelf = mshelf.extract(sel)
    mshelf.cluster = _cl
    mshelf.settings.waitonlock = 0
    mshelf.inversion.iscontrol = 0
    mshelf.miscellaneous.name = 'run_001_1_10_1e-17'
    mr = pyissm.model.execute.solve(mshelf, 'Stressbalance', load_only=True, runtime_name=False, check_consistency=False)
    return np.asarray(mr.mesh.extractedvertices).ravel(), np.asarray(mr.results.StressbalanceSolution.MaterialsRheologyBbar).ravel()


if 'ssa_inverted_solve_budd' in steps:

    print("-------------------------------------------------------------")
    print(" ASSEMBLING PURE-SSA BUDD (p=q=1) INVERTED MODEL")
    print("-------------------------------------------------------------")

    print("-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    # AIS3_param.nc currently has friction_law='schoof' baked in (built for the HO track) --
    # explicitly force Budd/p=q=1 here rather than trusting whatever class it already has.
    print("-- Forcing Budd/default friction class, p=q=1...")
    md.friction = pyissm.model.classes.friction.default(md.friction)
    md.friction.p = np.full(md.mesh.numberofelements, 1.0)
    md.friction.q = np.full(md.mesh.numberofelements, 1.0)
    md.friction.coefficient = np.full(md.mesh.numberofvertices, 1.8)

    print(f"-- Updating rheology B from floating-ice rheology L-curve...")
    _ev, _Bshelf = load_shelf_rheology_B()
    md.materials.rheology_B[_ev - 1] = _Bshelf

    print(f"-- Grafting friction field from {FRICTION_RUN_NAME} (RMSE=60.4 m/yr, see finalize_p1q1_reg_lcurve.py)...")
    mdf = pyissm.model.io.load_model(f'{FRICTION_RUN_DIR}/{FRICTION_RUN_NAME}.nc')
    _fld = np.asarray(md.friction.coefficient).astype(float)
    _grafted = np.asarray(mdf.results.StressbalanceSolution.FrictionCoefficient).ravel()
    _fld[mdf.mesh.extractedvertices - 1] = _grafted
    md.friction.coefficient = _fld

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print("-- Flooring thin ice at 100m (numerical stability), preserving observed surface where possible...")
    # Identical geometry treatment to ais_0.1.py's ssa_inverted_solve -- must match what
    # ssa_friction_inv_reg_lcurve actually solved against (see finalize_p1q1_reg_lcurve.py,
    # which applies the same flooring before extracting the friction-inversion domain).
    _ri = md.materials.rho_ice; _rw = md.materials.rho_water
    _H = np.asarray(md.geometry.thickness).ravel().copy()
    _ol = np.asarray(md.mask.ocean_levelset).ravel()
    _surf0 = np.asarray(md.geometry.surface).ravel().copy()
    _bed = np.asarray(md.geometry.bed).ravel()
    _H = np.maximum(_H, 100.0)
    _flt = _ol < 0
    _base = np.empty_like(_H); _surf = np.empty_like(_H)
    _base[_flt] = -_H[_flt] * _ri / _rw; _surf[_flt] = _H[_flt] * (1.0 - _ri / _rw)
    _base[_flt] = np.maximum(_base[_flt], _bed[_flt])
    _surf[_flt] = _base[_flt] + _H[_flt]
    _base[~_flt] = _bed[~_flt]
    _surf[~_flt] = _bed[~_flt] + _H[~_flt]
    md.geometry.thickness = _H; md.geometry.base = _base; md.geometry.surface = _surf

    print("-- Re-flooring effective pressure against the updated thickness...")
    _lim = 0.07
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    _Nfloor = _lim * md.materials.rho_ice * md.constants.g * _H
    N = np.maximum(N, _Nfloor)
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = _lim

    # NOTE: ais_0.1.py's ssa_inverted_solve has an extensive Schoof-only block here (3 waves
    # of Dirichlet pins for thin-ice/lower-bound-C/non-ice vertices, working around Schoof's
    # Coulomb-cap instability at the thickness floor -- see docs/inversion_worklog.md 5.4).
    # None of that applies to Budd (resistance grows, however weakly, with velocity, so
    # Newton always finds a finite equilibrium -- no yield-cap runaway mechanism exists for
    # this friction law). Deliberately omitted, not forgotten.

    print("-- Disabling inversion (forward solve only)...")
    md.inversion.iscontrol = 0
    md.verbose.solution = 1

    print("-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_SSA_inverted'
    md.cluster = cluster
    md.settings.waitonlock = 120  # minutes
    md.settings.solver_residue_threshold = 1e-3

    if save:
        print("-- Submitting and waiting on stress balance solution...")
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=False, runtime_name=False)

        if diagnostics:
            vel = np.asarray(md.results.StressbalanceSolution.Vel).ravel()
            vel_obs = np.asarray(md.inversion.vel_obs).ravel()
            il = np.asarray(md.mask.ice_levelset).ravel()
            ol = np.asarray(md.mask.ocean_levelset).ravel()
            gr = (il < 0) & (ol >= 0) & (vel_obs > 0)
            print(f"\nFORWARD SOLVE DIAGNOSTICS:")
            print(f"   Max modelled velocity: {np.nanmax(vel):.2f} m/yr")
            print(f"   Grounded RMSE vs obs:  {np.sqrt(np.nanmean((vel[gr]-vel_obs[gr])**2)):.2f} m/yr (target: ~60.4)")

        if plot:
            fig, ax, _trip = pyissm.plot.plot_model_field(
                md, md.results.StressbalanceSolution.Vel, show_cbar=True, cmap='PuOr',
                cbar_kwargs={'label': 'Modelled velocity (m/a)'})
            ax.set_title('AIS3_SSA_inverted -- pure-SSA Budd p=q=1 stress balance velocity')
            plt.savefig(f'{model_dir}/AIS3_SSA_inverted_velocity.png')

        print(f"\nSaving to {model_dir}/AIS3_SSA_inverted.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_SSA_inverted.nc')
    else:
        print("-- Submitting stress balance solve...")
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=False, runtime_name=False)


if 'ssa_relaxation_budd' in steps:

    print("-------------------------------------------------------------")
    print(" TRANSIENT RELAXATION (PURE-SSA BUDD)")
    print("-------------------------------------------------------------")

    print("-- Loading inverted model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_SSA_inverted.nc')

    md.inversion.iscontrol = 0
    md.verbose.solution = 1

    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.isthermal = 0
    md.transient.isgroundingline = 1
    md.groundingline.migration = 'SubelementMigration'
    md.transient.requested_outputs = ['default', 'Vel', 'Thickness', 'Surface', 'Base', 'MaskOceanLevelset']

    # Same NaN-default/zero-placeholder fixes already found and documented in ais_0.1.py's
    # ssa_relaxation -- this is a fresh transient activation on a freshly-assembled model, so
    # they apply again here.
    md.masstransport.spcthickness = np.full(md.mesh.numberofvertices, np.nan)
    md.smb.mass_balance = np.zeros(md.mesh.numberofvertices)
    md.basalforcings.groundedice_melting_rate = np.zeros(md.mesh.numberofvertices)
    md.basalforcings.floatingice_melting_rate = np.zeros(md.mesh.numberofvertices)

    # TODO: tune final_time/time_step for your actual use case, same as ais_0.1.py's own
    # ssa_relaxation -- 20yr/0.05yr is a placeholder shock-damping window, not a validated
    # choice for this specific track.
    md.timestepping.start_time = 0
    md.timestepping.final_time = 20
    md.timestepping.time_step = 0.05

    print("-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_SSA_relaxed'
    md.cluster = cluster
    md.settings.waitonlock = 1440
    md.settings.solver_residue_threshold = 1e-3

    if save:
        print("-- Submitting and waiting on transient relaxation...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only=False, runtime_name=False)

        if diagnostics:
            dH = np.asarray(md.results.TransientSolution[-1].Thickness).ravel() - np.asarray(md.geometry.thickness).ravel()
            print(f"\nRELAXATION DIAGNOSTICS:")
            print(f"   Max |dH| over relaxation: {np.nanmax(np.abs(dH)):.2f} m")
            print(f"   Mean |dH| over relaxation: {np.nanmean(np.abs(dH)):.2f} m")

        print(f"\nSaving to {model_dir}/AIS3_SSA_relaxed.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_SSA_relaxed.nc')
    else:
        print("-- Submitting transient relaxation...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only=False, runtime_name=False)


## ------------------------------------
## Stage 3: Ocean basal melt -- apply the validated calibration, SSA mesh
## ------------------------------------
# Builds AIS3_melt_final_ssa.nc (historical_dhdt_tuning_ssa's input) from the validated
# ISMIP6 quadratic-local melt calibration: gamma_0 = 300 m/yr with per-basin refit
# deltaT_basin, official IMBIE2 basins, Zhou thermal-forcing climatology (worklog 4k, 4n).
# No solve: the melt parameterisation is evaluated inside the downstream transients.
#
# The calibration SEARCH is not rerun here. It is a multi-round chain of standalone scripts
# (melt_gamma_sweep_ssa.py -> melt_deltaT_sensitivity_test_ssa.py -> ssa_melt_deltaT_basin_refit.py
# -> ssa_melt_deltaT_refit_recalibrate.py; state in models/deltaT_refit_state_ssa.json), and this
# step hard-codes its result. The J1-only gamma_0 sweep with fixed (published) deltaT_basin
# this step used to run was removed: its pick (5537.7) is superseded once deltaT_basin is
# refit per basin (J1 then ~0 for every candidate; J2 decides, worklog 4c).
#
# Loads AIS3_SSA_relaxed.nc as the calibration did; its geometry is the inverted (unrelaxed)
# geometry, by design (worklog 4z correction). Thickness was already restored upstream in
# AIS3_SSA_inverted.nc (restore_floored_thickness_ssa.py, worklog 4k).
#
# Refuses to overwrite an existing output: archive it first (mv -n) or set AIS3_MELT_TAG.
if 'melt_gamma_tuning_ssa' in steps:
    print("-------------------------------------------------------------")
    print(" OCEAN BASAL-MELT CALIBRATION, SSA (VALIDATED: gamma_0=300, REFIT deltaT)")
    print("-------------------------------------------------------------")

    from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

    ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
    ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                     'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
    BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
    TIME_SENTINEL = 1e9

    # Validated values: candidate g1 of ssa_melt_deltaT_basin_refit.py (converged round 5, run
    # AIS3_ssa_deltaT_refit_g1_r5), chosen by ssa_melt_deltaT_refit_recalibrate.py (job
    # 179305728, J2 minimum). delta_t[k] belongs to IMBIE2 basin k+1. Close to, but not the
    # same as, the HO track's refit values (different mesh).
    MELT_GAMMA0 = 300.0
    MELT_DELTA_T = np.array([
        1.8467609040725819, 2.7988123113211087, 2.2562599573273427, 4.599712312498511,
        8.762072283674417, 5.8733775833670565, 3.728768234826943, 0.8688693456218887,
        7.1878933110812575, 12.258715419441897, 5.2128687613028335, 4.89285368124807,
        2.3120433144628567, 3.4396433109318245, 1.2381000126240504, 2.2497361133862843])

    out_path = f"{model_dir}/AIS3_melt_final_ssa{os.environ.get('AIS3_MELT_TAG', '')}.nc"
    if save and os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists -- archive it first (mv -n) or set AIS3_MELT_TAG")

    print("-- Loading relaxed SSA model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_SSA_relaxed.nc')
    md.results.TransientSolution = []  # 400-step relaxation history, unused downstream (worklog 4g)

    print("-- Configuring ISMIP6 basal melt parameterisation (IMBIE2 basins)...")
    md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)
    ds_basin = xr.open_dataset(BASIN_FILE)
    basin_grid = ds_basin['basinNumber'].values
    bxx, byy = np.meshgrid(ds_basin['x'].values, ds_basin['y'].values)
    num_basins = int(np.nanmax(basin_grid)) + 1
    assert num_basins == MELT_DELTA_T.size, f"{num_basins} basins vs {MELT_DELTA_T.size} delta_t values"
    # basin_id is per element: nearest basin at each element centroid (2D mesh, no projection).
    elx = np.asarray(md.mesh.elements).astype(int) - 1
    mesh_x = np.asarray(md.mesh.x).ravel()
    mesh_y = np.asarray(md.mesh.y).ravel()
    ecx = mesh_x[elx[:, :3]].mean(axis=1)
    ecy = mesh_y[elx[:, :3]].mean(axis=1)
    basin_lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
    md.basalforcings.basin_id = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)
    md.basalforcings.num_basins = num_basins
    md.basalforcings.islocal = 1
    md.basalforcings.gamma_0 = MELT_GAMMA0
    md.basalforcings.delta_t = MELT_DELTA_T.copy()

    print("-- Interpolating Zhou ocean thermal-forcing climatology onto the mesh...")
    ds_tf = xr.open_dataset(ZHOU_TF_FILE)
    tfx, tfy, tfz = ds_tf['x'].values, ds_tf['y'].values, ds_tf['z'].values
    tf_full = ds_tf['tf'].values  # (z, y, x)
    tf_list = []
    for k in range(tfz.size):
        vals = RegularGridInterpolator((tfy, tfx), tf_full[k], method='linear',
                                        bounds_error=False, fill_value=np.nan)(np.column_stack([mesh_y, mesh_x]))
        tf_list.append(np.append(np.nan_to_num(vals, nan=0.0), TIME_SENTINEL).reshape(-1, 1))
    md.basalforcings.tf = tf_list
    md.basalforcings.tf_depths = tfz.copy()
    print(f"   {tfz.size} depth layers ({tfz.min():.0f} to {tfz.max():.0f} m)")

    # Forward-run settings the downstream transients inherit (they reset the rest themselves).
    md.inversion.iscontrol = 0
    md.masstransport.spcthickness = np.full(md.mesh.numberofvertices, np.nan)
    md.smb.mass_balance = np.zeros(md.mesh.numberofvertices)

    if save:
        print(f"Saving to {out_path}")
        pyissm.model.io.save_model(md, out_path)


## ------------------------------------
## Stage 4: Historical run tuned against observed dH/dt, SSA mesh
## ------------------------------------
# Mirrors ais_0.1.py's historical_dhdt_tuning (RACMO SMB 1995-2019, compared against MIPKIT's
# dhdt_cpom). Same data-coverage caveat applies: dhdt_cpom stops at 2019, not 2025 -- run
# 1995-2019, not 1995-2025, same as the HO track.
if 'historical_dhdt_tuning_ssa' in steps:
    print("-------------------------------------------------------------")
    print(" HISTORICAL RUN 1995-2019 (SSA) TUNED AGAINST OBSERVED dH/dt")
    print("-------------------------------------------------------------")

    print("-- Loading melt-calibrated SSA model...")
    # BUGFIX (2026-09-21): AIS3_melt_gamma_tuning_ssa.nc only carries that step's own COARSE
    # 9-point sweep pick (gamma_0=5537.7, published/unrefit deltaT_basin) -- it was never
    # updated after ssa_melt_deltaT_basin_refit.py's per-basin secant refit converged on the
    # actual accepted answer (gamma_0=300, refit deltaT_basin, confirmed best-by-J2 in
    # ssa_melt_deltaT_refit_recalibrate.py). Load the properly reconstructed final
    # calibration instead (see finalize_ssa_melt_calibration.py for how it was built).
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_melt_final_ssa.nc')
    # BUGFIX (same as ais_0.1.py's ho_relaxation/historical_dhdt_tuning, 2026-09-02):
    # save_model/load_model returns each basalforcings.tf entry as a plain Python list, not
    # ndarray -- crashes marshalling's numpy elementwise scaling step. Re-cast here too,
    # proactively, since this step reloads a model with tf already populated.
    md.basalforcings.tf = [np.asarray(t, dtype=float) for t in md.basalforcings.tf]

    md.inversion.iscontrol = 0
    md.verbose.solution = 1

    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.isthermal = 0
    md.transient.isgroundingline = 1
    md.groundingline.migration = 'SubelementMigration'

    md.timestepping.start_time = 1995
    md.timestepping.final_time = 2019  # bounded by dhdt_cpom coverage, see note above
    md.timestepping.time_step = 0.1

    print("-- Building time-varying SMB from RACMO 1995-2019 annual means...")
    smb_years = np.arange(1995, 2020)
    racmo_smb_data = ccdtools.catalog.DataCatalog().load_dataset('racmo2.4p1_monthly_11km_1979-2023')
    smbgl = racmo_smb_data['smbgl']
    [racmo_x, racmo_y] = pyissm.tools.general.ll_to_xy(racmo_smb_data['lat'].values, racmo_smb_data['lon'].values, -1)

    nv = md.mesh.numberofvertices
    mb_arr = np.empty((nv + 1, smb_years.size))
    for i, yr in enumerate(smb_years):
        smb_yr = smbgl.sel(time=smbgl['time.year'] == yr)
        # BUGFIX (2026-09-22, same bug already found and fixed on the HO track's identical
        # code): smbgl carries a singleton 'height' dimension (dims: time, height, rlat,
        # rlon) that survives summing over 'time', leaving a 3D (1, rlat, rlon) array where
        # racmo_x/racmo_y are 2D (rlat, rlon) -- points_to_mesh requires matching shapes.
        smb_yr_myr = (smb_yr.sum('time') / md.materials.rho_ice).squeeze('height').to_numpy()
        mb_arr[:nv, i] = pyissm.data.interp.points_to_mesh(racmo_x, racmo_y, smb_yr_myr, md.mesh.x, md.mesh.y)
        print(f"   {yr}: mesh-mean SMB = {np.nanmean(mb_arr[:nv, i]):.3f} m ice eq/yr")
    mb_arr[nv, :] = smb_years.astype(float)
    md.smb = pyissm.model.classes.smb.default(md.smb)
    md.smb.mass_balance = mb_arr

    print("-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_historical_1995_2019_SSA'
    md.cluster = cluster  # plain 48c/190GB config -- SSA mesh, no hugemem needed
    md.settings.waitonlock = 1440  # minutes -- synchronous submit-and-wait, same pattern
    md.settings.solver_residue_threshold = 1e-3

    print("-- Submitting and waiting on historical transient run...")
    md = pyissm.model.execute.solve(md, 'Transient', load_only=False, runtime_name=False)

    # Save BEFORE the diagnostic (2026-09-21, same lesson as ais_0.1.py's chunked
    # historical_dhdt_tuning): don't let a diagnostic-only crash lose an already-completed
    # transient run's results.
    print(f"\nSaving to {model_dir}/AIS3_historical_1995_2019_SSA.nc")
    pyissm.model.io.save_model(md, f'{model_dir}/AIS3_historical_1995_2019_SSA.nc')

    print("-- Comparing simulated dH/dt against MIPKIT's dhdt_cpom (1993-2019)...")
    mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
    # BUGFIX (2026-09-21, found via check_historical_chunk1.py crashing on the HO track's
    # identical pattern): xr_to_mesh requires a 2D variable on a rectilinear grid, but
    # mipkit's own dhdt_cpom is 3D (carries cpom_dhdt_time) -- passing the full
    # dataset+var_name through crashes with "must be 2D on a rectilinear grid". Slice to the
    # last available year and wrap back into a one-variable Dataset (keeping x1km/y1km
    # coords) instead of passing the un-sliced 3D variable directly.
    dhdt_obs_grid = mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=-1)
    dhdt_obs_mesh = pyissm.data.interp.xr_to_mesh(
        dhdt_obs_grid, 'dhdt_cpom', md.mesh.x, md.mesh.y, x_var='x1km', y_var='y1km')

    thick_ts = md.results.TransientSolution
    H0 = np.asarray(thick_ts[0].Thickness).ravel()
    H1 = np.asarray(thick_ts[-1].Thickness).ravel()
    years_elapsed = md.timestepping.final_time - md.timestepping.start_time
    dhdt_sim = (H1 - H0) / years_elapsed

    gr = (np.asarray(md.mask.ice_levelset).ravel() < 0) & (np.asarray(md.mask.ocean_levelset).ravel() > 0)
    mismatch_rmse = np.sqrt(np.nanmean((dhdt_sim[gr] - dhdt_obs_mesh[gr]) ** 2))
    print(f"   Grounded dH/dt mismatch RMSE vs dhdt_cpom = {mismatch_rmse:.3f} m/yr")


## ------------------------------------
## Stage 5: Future projections (CTRL and SSP-forced), SSA mesh
## ------------------------------------
# Design: docs/projection_experiments.md. One experiment per job chain, selected with qsub -v
# (launch_projection_ssa.pbs); nothing in this block reads the shared `steps` edit.
#
#   AIS3_PROJ_SCENARIO  ctrl | ssp126 | ssp245 | ssp370 | ssp534-over | ssp585   (default ctrl)
#   AIS3_PROJ_GCM       CESM2-WACCM | MRI-ESM2-0 | ACCESS-CM2 | ...               (not for ctrl)
#   AIS3_PROJ_FORCING   both | ocean | smb   -- which anomalies are applied        (default both)
#   AIS3_PROJ_MELT      ref (gamma_0=300 + refit deltaT, AIS3_melt_final_ssa.nc) | high (= g7,
#                       gamma_0=15000, the J1+J2+J3 95th percentile, worklog 6.1) | g<i> (refit
#                       candidate i of deltaT_refit_state_ssa.json)                (default ref)
#   AIS3_PROJ_END       final year                         (default 2300, or 2100 for ssp370)
#   AIS3_PROJ_SMOKE=1   test chunk of AIS3_PROJ_SMOKE_YEARS (default 0.2) yr into *_smoke names, no
#                       checkpoint, no chaining
#   AIS3_PROJ_LARMIP_REGION  1-5: LARMIP experiment instead of a scenario -- CTRL forcing plus an
#                       extra uniform melt (md.basalforcings.melt_anomaly) on every vertex in that
#                       region of the LARMIP mask. ISSM adds it only under fully floating elements,
#                       so it follows the grounding line. Runs 200 yr (to 2219).
#   AIS3_PROJ_LARMIP_MODE    step (default: full amplitude from 2019) | ramp (0 -> amplitude
#                       linearly over 100 yr, then held 100 yr)
#   AIS3_PROJ_LARMIP_AMP     extra melt in m ice/yr (default 8)
#   AIS3_LARMIP_MASK         region mask file (default assets/larmip3/LARMIP3_sectors-1km.nc: 1 Weddell,
#                       2 East Antarctica, 3 Ross, 4 Amundsen, 5 Peninsula; 0 = no sector): one 2D
#                       integer variable (AIS3_LARMIP_MASK_VAR, else the first 2D variable) with
#                       values 1-5, on x/y polar-stereographic or lat/lon coordinates
#   AIS3_PROJ_COLLAPSE=1     ISMIP7 ice-shelf collapse (fracture path C) for an SSP experiment: the
#                       GCM/scenario's annual collapse mask (fracture/<ver>/ice_shelf_collapse_mask_*).
#                       Each chunk is split into sub-solves at every January 1st on which the mask
#                       changes on the mesh; before each, masked vertices that hold floating ice are
#                       made ice-free (ice_levelset +1, thickness min_thickness), as are floating pieces
#                       this cuts off from grounded ice (rigid-body modes otherwise). Only floating ice
#                       collapses (ISMIP7 README); ice that ungrounds inside the mask goes at the next
#                       mask change or chunk start. The front stays fixed otherwise (ismovingfront=0):
#                       the new edge is an ordinary calving front. Removed shelf mass leaves no flux
#                       term; it shows as a drop in floating ice volume. History gets collapsed_at
#                       (removal time per vertex). Experiment name gets a _collapse suffix.
#                       (An extra-melt version, leaving collapsed shelves at 1 m, blew up in its smoke
#                       test: detached shelf remnants reached 4e9 m/yr, worklog 6.12.)
#   AIS3_PROJ_COLLAPSE_VER   mask version (default: newest, v2.1 CESM2-WACCM, v2/v1 MRI-ESM2-0; a
#                       non-default version is added to the name, e.g. _collapse-v2)
#   AIS3_PROJ_COLLAPSE_SMOKE_SHIFT  smoke tests only: read the mask this many years ahead
#   AIS3_PROJ_FRICTION=coupledN  sensitivity test (worklog 6.14): effective pressure tied to
#                       flotation (friction.coupling 2: N = max(rho_i g H - rho_w g max(0,-base),
#                       limit * rho_i g H)) instead of the fixed prescribed N (coupling 3), so drag
#                       falls as ice thins towards flotation. friction.coefficient is rescaled ONCE,
#                       from the initial state (C_new = C_old sqrt(N_old/N_new), Budd p=q=1), so the
#                       starting basal drag is unchanged; checkpoints carry it on. Name gets _coupledN.
#   AIS3_PROJ_MELTINTERP / AIS3_PROJ_STATICMELT=1  melt-at-the-grounding-line test (worklog 6.17): ISSM's
#                       ismip6 melt is set only on fully floating elements, so melt_interpolation alone
#                       cannot put melt into grounding-line elements. STATICMELT replaces it with the
#                       default class: a fixed melt field = CTRL's first output (t=2019.1), extended by
#                       nearest neighbour across the grounding line, + the LARMIP step anomaly.
#                       MELTINTERP sets groundingline.melt_interpolation (e.g. SubelementMelt1). One
#                       chunk from the initial state only (AIS3_PROJ_END<=2050); ctrl / LARMIP step.
#   AIS3_PROJ_FRICTION=coupledN_p3  as coupledN plus nonlinear Budd sliding (p=3, q=1: drag ~
#                       C^2 N^(1/3) u^(1/3) instead of C^2 N u). C is rescaled once so the drag is
#                       unchanged at the initial speed: C_new = C_old sqrt(N_old u0^(2/3) / N_new^(1/3)),
#                       u0 in m/s with a 1 m/yr floor. Name gets _coupledN_p3.
#
# Forcing is an anomaly on the reference state the model was calibrated with (worklog 4c):
#   TF  = TF0  (Zhou climatology, as in the historical run) + [TF_gcm(t)  - TF_gcm(1995-2014)]
#   SMB = SMB0 (RACMO 1995-2014 mean)                         + [SMB_gcm(t) - SMB_gcm(1995-2014)]
# ctrl applies no anomaly. GCM SMB is dEBM2-8000m (the only product for all 7 ISMIP7 GCMs); no
# elevation feedback (dacabfdz) yet. Anomalies are annual, placed at mid-year; NaN (outside a
# source grid) -> zero anomaly. Years past the end of a GCM file hold its last year.
#
# Chunks: 2019 -> 2050 -> 2075 -> 2100 -> ... (25 yr) -> END, each one PBS submission (~5 h of the 48-cpu
# solve per 25 yr at the historical run's 12.6 min/model-yr). Each chunk writes a NEW
# checkpoint AIS3_proj_ssa_{exp}_{year}.nc (state adopted from the last step, forcing reset to
# the static baseline, results stripped) and a per-chunk history npz, then qsubs the next chunk
# unless models/AIS3_proj_ssa_{exp}.STOP exists. Nothing is overwritten.
if 'projection_ssp_ssa' in steps:

    print("-------------------------------------------------------------")
    print(" FUTURE PROJECTION, SSA (ANOMALY-FORCED, CHUNKED)")
    print("-------------------------------------------------------------")
    import subprocess
    import re

    FORCING_ROOT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS'
    SMB_PARAM = 'dEBM2-8000m'
    CLIM_Y0, CLIM_Y1 = 1995, 2014
    START_YEAR = 2019.0
    TIME_STEP = 0.1
    TIME_SENTINEL = 1e9
    # 25-yr chunks after 2050 (2026-10-02): a chunk on a slow solve (ssp585 2050-2100 ran at ~20
    # steps/h, twice) then still fits its time request instead of losing the whole chunk.
    CHUNK_EDGES = [2019.0, 2050.0] + [float(y) for y in range(2075, 2301, 25)]

    scenario = os.environ.get('AIS3_PROJ_SCENARIO', 'ctrl')
    gcm = os.environ.get('AIS3_PROJ_GCM', '')
    forcing_mode = os.environ.get('AIS3_PROJ_FORCING', 'both')
    melt_choice = os.environ.get('AIS3_PROJ_MELT', 'ref')
    MELT_MEMBERS = {'high': 'g7'}  # no low member: gamma_0=300 is already the least responsive converged candidate
    end_year = float(os.environ.get('AIS3_PROJ_END', '2100' if scenario == 'ssp370' else '2300'))
    smoke = os.environ.get('AIS3_PROJ_SMOKE', '') == '1'
    assert forcing_mode in ('both', 'ocean', 'smb'), forcing_mode
    if scenario == 'ctrl':
        exp = f'ctrl_{melt_choice}' if melt_choice != 'ref' else 'ctrl'
        use_ocean = use_smb = False
    else:
        assert gcm, 'AIS3_PROJ_GCM is required for an SSP experiment'
        gtag = gcm.split('-')[0].lower()
        ftag = {'both': 'ref', 'ocean': 'oceanonly', 'smb': 'smbonly'}[forcing_mode]
        exp = f'{scenario}_{gtag}_{melt_choice}_{ftag}'  # <scenario>_<gcm>_<melt>_<physics/forcing>
        use_ocean = forcing_mode in ('both', 'ocean')
        use_smb = forcing_mode in ('both', 'smb')
    larmip_region = os.environ.get('AIS3_PROJ_LARMIP_REGION', '')
    if larmip_region:
        assert scenario == 'ctrl', 'LARMIP experiments use CTRL forcing; leave AIS3_PROJ_SCENARIO unset'
        larmip_mode = os.environ.get('AIS3_PROJ_LARMIP_MODE', 'step')
        assert larmip_mode in ('step', 'ramp'), larmip_mode
        larmip_amp = float(os.environ.get('AIS3_PROJ_LARMIP_AMP', '8'))
        exp = f'larmip_r{int(larmip_region)}_{larmip_mode}' + ('' if larmip_amp == 8.0 else f'_{larmip_amp:g}')
        end_year = float(os.environ.get('AIS3_PROJ_END', str(START_YEAR + 200.0)))
    collapse = os.environ.get('AIS3_PROJ_COLLAPSE', '') == '1'
    if collapse:
        assert scenario != 'ctrl', 'ISMIP7 collapse masks exist for SSP scenarios only'
        cands = {}
        for d in glob.glob(f'{FORCING_ROOT}/{gcm}/{scenario}/fracture/v*/'):
            fs = glob.glob(f'{d}ice_shelf_collapse_mask_*.nc')
            if fs:
                assert len(fs) == 1, fs
                cands[Path(d).name] = fs[0]
        if not cands:
            raise FileNotFoundError(f'no ISMIP7 collapse mask for {gcm}/{scenario}')
        newest = max(cands, key=lambda v: tuple(int(p) for p in v[1:].split('.')))
        collapse_ver = os.environ.get('AIS3_PROJ_COLLAPSE_VER') or newest
        assert collapse_ver in cands, f'collapse mask {collapse_ver} not in {sorted(cands)}'
        collapse_file = cands[collapse_ver]
        exp += '_collapse' + ('' if collapse_ver == newest else f'-{collapse_ver}')
        print(f"-- Collapse mask {collapse_file} ({collapse_ver})")
    melt_interp = os.environ.get('AIS3_PROJ_MELTINTERP', '')
    static_melt = os.environ.get('AIS3_PROJ_STATICMELT', '') == '1'
    if static_melt:
        exp += '_staticmelt'
    if melt_interp:
        exp += '_' + {'SubelementMelt1': 'sem1', 'SubelementMelt2': 'sem2', 'FullMeltOnPartiallyFloating': 'fmp',
                      'NoMeltOnPartiallyFloating': 'nmp', 'IntrusionMelt': 'intrusion'}[melt_interp]
    friction_mode = os.environ.get('AIS3_PROJ_FRICTION', '')
    assert friction_mode in ('', 'coupledN', 'coupledN_p3'), friction_mode
    if friction_mode:
        exp += f'_{friction_mode}'
    edges = [e for e in CHUNK_EDGES if e < end_year] + [end_year]
    print(f"-- Experiment {exp}: scenario={scenario} gcm={gcm or '-'} forcing={forcing_mode} "
          f"melt={melt_choice} end={end_year:.0f} smoke={smoke}")

    init_path = f'{model_dir}/AIS3_proj_ssa_init.nc'
    baseline_path = f'{model_dir}/AIS3_proj_ssa_baseline.npz'
    stop_path = f'{model_dir}/AIS3_proj_ssa_{exp}.STOP'

    def _last_step(ts, field, nv):
        if isinstance(ts, list):
            return np.asarray(getattr(ts[-1], field), dtype=float).ravel().copy()
        raw = np.asarray(getattr(ts, field), dtype=float).ravel()
        return raw.reshape(raw.size // nv, nv)[-1].copy()

    def adopt_final_state_2d(md):
        """SSA copy of ais_0.1.py's adopt_final_state: Thickness + MaskOceanLevelset from the last
        step, base/surface recomputed hydrostatically (never below bed), velocities into
        md.initialization."""
        ts = md.results.TransientSolution
        nv = md.mesh.numberofvertices
        thick = _last_step(ts, 'Thickness', nv)
        oln = _last_step(ts, 'MaskOceanLevelset', nv)
        bed = np.asarray(md.geometry.bed, dtype=float).ravel()
        ri, rw = float(md.materials.rho_ice), float(md.materials.rho_water)
        base_new = np.maximum(np.where(oln >= 0, bed, -thick * ri / rw), bed)
        dH = thick - np.asarray(md.geometry.thickness, dtype=float).ravel()
        md.geometry.thickness = thick
        md.geometry.base = base_new
        md.geometry.surface = base_new + thick
        md.mask.ocean_levelset = oln
        for field in ('Vx', 'Vy', 'Vel'):
            setattr(md.initialization, field.lower(), _last_step(ts, field, nv))
        print(f"   adopted last step: mean|dH| vs previous geometry {np.mean(np.abs(dH)):.3f} m, "
              f"max {np.max(np.abs(dH)):.2f} m, grounded vertices {int((oln > 0).sum())}")

    def set_static_forcing(md, tf0, tf_depths, smb0):
        if type(md.basalforcings).__name__ == 'ismip6':  # not for the AIS3_PROJ_STATICMELT test class
            md.basalforcings.tf = [np.append(tf0[k], TIME_SENTINEL).reshape(-1, 1) for k in range(tf0.shape[0])]
            md.basalforcings.tf_depths = np.asarray(tf_depths, dtype=float).copy()
        md.smb = pyissm.model.classes.smb.default(md.smb)
        md.smb.mass_balance = np.asarray(smb0, dtype=float).copy()

    # -- Projection initial state (built once from the historical run's 2019 end state) --------
    if not os.path.exists(init_path):
        print("-- Building projection initial state from AIS3_historical_1995_2019_SSA.nc...")
        md = pyissm.model.io.load_model(f'{model_dir}/AIS3_historical_1995_2019_SSA.nc')
        md.basalforcings.tf = [np.asarray(t, dtype=float) for t in md.basalforcings.tf]
        adopt_final_state_2d(md)
        md.results.TransientSolution = []
        nv = md.mesh.numberofvertices
        mb = np.asarray(md.smb.mass_balance, dtype=float)
        mb_years = mb[nv, :]
        sel = (mb_years >= CLIM_Y0) & (mb_years <= CLIM_Y1)
        assert sel.sum() == CLIM_Y1 - CLIM_Y0 + 1, f'historical SMB years {mb_years}'
        smb0 = np.nan_to_num(mb[:nv, sel].mean(axis=1), nan=0.0)
        tf0 = np.stack([np.asarray(t, dtype=float)[:nv, 0] for t in md.basalforcings.tf])
        tf_depths0 = np.asarray(md.basalforcings.tf_depths, dtype=float).ravel()
        print(f"   SMB0 = RACMO {CLIM_Y0}-{CLIM_Y1} mean: mesh-mean {smb0.mean():.4f} m ice eq/yr "
              f"(2019 column {np.nanmean(mb[:nv, mb_years == 2019]):.4f}); TF0 {tf0.shape}, "
              f"gamma_0={float(md.basalforcings.gamma_0)}")
        set_static_forcing(md, tf0, tf_depths0, smb0)
        md.timestepping.start_time = START_YEAR
        md.timestepping.final_time = START_YEAR
        np.savez(baseline_path + '.tmp.npz', tf0=tf0, tf_depths=tf_depths0, smb0=smb0)
        os.replace(baseline_path + '.tmp.npz', baseline_path)
        pyissm.model.io.save_model(md, init_path + '.tmp')
        os.replace(init_path + '.tmp', init_path)
        print(f"   Saved {init_path} and {baseline_path}")
        del md

    base = np.load(baseline_path)
    tf0, tf_depths0, smb0 = base['tf0'], base['tf_depths'], base['smb0']

    # -- Resume from the latest checkpoint of this experiment, else the initial state ----------
    ckpts = []
    for f in glob.glob(f'{model_dir}/AIS3_proj_ssa_{exp}_*.nc'):
        tail = Path(f).stem[len(f'AIS3_proj_ssa_{exp}_'):]
        if tail.isdigit():
            ckpts.append((float(tail), f))
    ckpts.sort()
    if ckpts and not smoke:
        current_time, resume_file = ckpts[-1]
    else:
        current_time, resume_file = START_YEAR, init_path
    if current_time >= end_year:
        raise SystemExit(f"{exp} already complete (checkpoint at {current_time:.0f})")
    t1 = next(e for e in edges if e > current_time)
    if os.environ.get('AIS3_PROJ_CHUNK_YEARS'):  # shorter chunks for runs that slow down (worklog 6.16)
        t1 = min(t1, current_time + float(os.environ['AIS3_PROJ_CHUNK_YEARS']))
    if smoke:
        t1 = current_time + float(os.environ.get('AIS3_PROJ_SMOKE_YEARS', '0.2'))
    print(f"-- Chunk {current_time:.1f} -> {t1:.1f} from {resume_file}")
    md = pyissm.model.io.load_model(resume_file)
    md.results.TransientSolution = []
    nv = md.mesh.numberofvertices
    assert abs(float(md.timestepping.final_time) - current_time) < 1e-6, \
        f'checkpoint final_time {md.timestepping.final_time} != {current_time}'

    if friction_mode in ('coupledN', 'coupledN_p3'):
        if resume_file == init_path:
            assert int(md.friction.coupling) == 3, md.friction.coupling
            g_, ri_, rw_ = float(md.constants.g), float(md.materials.rho_ice), float(md.materials.rho_water)
            lim_ = float(md.friction.effective_pressure_limit)
            H_ = np.asarray(md.geometry.thickness, dtype=float).ravel()
            base_ = np.asarray(md.geometry.base, dtype=float).ravel()
            floor_ = lim_ * ri_ * g_ * H_
            N_old = np.maximum(np.maximum(np.asarray(md.friction.effective_pressure, dtype=float).ravel(), 0.0), floor_)
            N_new = np.maximum(ri_ * g_ * H_ - rw_ * g_ * np.maximum(-base_, 0.0), floor_)
            if friction_mode == 'coupledN':
                ratio = np.sqrt(np.where(N_new > 0, N_old / np.maximum(N_new, 1e-6), 1.0))
            else:  # drag C^2 N u (p=q=1)  ->  C^2 N^(1/3) u^(1/3) (p=3, q=1), equal at u0
                assert np.all(np.asarray(md.friction.p) == 1) and np.all(np.asarray(md.friction.q) == 1)
                yts_ = float(md.constants.yts)
                u0_ = np.maximum(np.asarray(md.initialization.vel, dtype=float).ravel(), 1.0) / yts_
                ratio = np.sqrt(N_old * u0_ ** (2.0 / 3.0) / np.maximum(N_new, 1e-6) ** (1.0 / 3.0))
                md.friction.p = 3.0 * np.ones_like(np.asarray(md.friction.p, dtype=float))
                md.friction.q = np.ones_like(np.asarray(md.friction.q, dtype=float))
            C_old = np.asarray(md.friction.coefficient, dtype=float).ravel()
            md.friction.coefficient = C_old * ratio
            md.friction.coupling = 2
            gr_ = (np.asarray(md.mask.ocean_levelset).ravel() > 0) & (np.asarray(md.mask.ice_levelset).ravel() < 0)
            print(f"   friction {friction_mode}: coupling 3 -> 2, C rescaled once; grounded N_new/N_old median "
                  f"{np.median(N_new[gr_] / N_old[gr_]):.3f} (p5 {np.percentile(N_new[gr_] / N_old[gr_], 5):.3f}, "
                  f"p95 {np.percentile(N_new[gr_] / N_old[gr_], 95):.3f}); C factor median {np.median(ratio[gr_]):.3f}")
        else:
            assert int(md.friction.coupling) == 2, f'{resume_file}: coupledN checkpoint has coupling {md.friction.coupling}'
            assert np.all(np.asarray(md.friction.p) == (3.0 if friction_mode == 'coupledN_p3' else 1.0)), 'friction.p'
            print(f"   friction {friction_mode}: carried over from the checkpoint (coupling 2, rescaled C)")

    # -- Melt parameters -------------------------------------------------------------------
    if melt_choice != 'ref':
        import json
        with open(f'{model_dir}/deltaT_refit_state_ssa.json') as f:
            cand = json.load(f)['candidates'][MELT_MEMBERS.get(melt_choice, melt_choice).lstrip('g')]
        md.basalforcings.gamma_0 = float(cand['gamma_0'])
        md.basalforcings.delta_t = np.asarray(cand['history'][-1]['delta_t'], dtype=float)
    print(f"   melt: gamma_0={float(md.basalforcings.gamma_0)}, deltaT range "
          f"[{np.min(md.basalforcings.delta_t):.3f}, {np.max(md.basalforcings.delta_t):.3f}]")

    if larmip_region:
        from scipy.interpolate import NearestNDInterpolator
        mask_path = os.environ.get('AIS3_LARMIP_MASK', '/g/data/au88/jh7060/ACCESS-AIS3/assets/larmip3/LARMIP3_sectors-1km.nc')
        ds_m = xr.open_dataset(mask_path)
        mvar = os.environ.get('AIS3_LARMIP_MASK_VAR') or next(v for v in ds_m.data_vars if ds_m[v].ndim == 2)
        mvals = ds_m[mvar].values.astype(float)
        if 'x' in ds_m[mvar].dims and 'y' in ds_m[mvar].dims:
            mx, my = np.meshgrid(ds_m['x'].values, ds_m['y'].values)
        else:
            lat, lon = ds_m['lat'].values, ds_m['lon'].values
            if lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            mx, my = pyissm.tools.general.ll_to_xy(lat, lon, -1)
        ok = np.isfinite(mvals)
        region_v = np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mvals[ok])(
            np.asarray(md.mesh.x).ravel(), np.asarray(md.mesh.y).ravel())).astype(int)
        in_r = region_v == int(larmip_region)
        fl = np.asarray(md.mask.ocean_levelset).ravel() < 0
        print(f"   LARMIP region {larmip_region} ({mask_path}:{mvar}): {int(in_r.sum())} vertices, "
              f"{int((in_r & fl).sum())} currently floating; vertices per region "
              + ', '.join(f'{r}:{int((region_v == r).sum())}' for r in range(1, 6)))
        assert (in_r & fl).sum() > 0, 'no floating ice in this LARMIP region'
        if larmip_mode == 'step':
            md.basalforcings.melt_anomaly = np.where(in_r, larmip_amp, 0.0)
        else:  # 0 at the start, amplitude after 100 yr, held to 200 yr (ISSM interpolates linearly)
            ma = np.zeros((nv + 1, 3))
            ma[:nv, 1] = ma[:nv, 2] = np.where(in_r, larmip_amp, 0.0)
            ma[nv] = [START_YEAR, START_YEAR + 100.0, START_YEAR + 200.0]
            md.basalforcings.melt_anomaly = ma
        print(f"   melt_anomaly: {larmip_mode}, {larmip_amp:g} m/yr in region {larmip_region}")

    # -- Forcing for this chunk --------------------------------------------------------------
    years = np.arange(int(np.floor(current_time)), int(np.ceil(t1)) + 1)
    col_times = years + 0.5

    def _files_for(root, pattern_year_fn, y0, y1):
        return [f for f in sorted(glob.glob(root)) if pattern_year_fn(f)[1] >= y0 and pattern_year_fn(f)[0] <= y1]

    def _tf_years(f):
        y = Path(f).stem.split('_')[-1].split('-')
        return int(y[0]), int(y[-1])

    def _load_tf(gcm_, scen, y0, y1):
        files = _files_for(f'{FORCING_ROOT}/{gcm_}/{scen}/ocean/tf/v3/tf_AIS_*.nc', _tf_years, y0, y1)
        if not files:
            raise FileNotFoundError(f'no TF files for {gcm_}/{scen} {y0}-{y1}')
        # cftime throughout: proleptic-Gregorian files (MRI-ESM2-0) otherwise decode to pandas timestamps
        # before ~2262 and cftime after, which open_mfdataset cannot combine.
        da = xr.open_mfdataset(files, combine='by_coords',
                               decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))['tf']
        return da.sel(time=(da['time.year'] >= y0) & (da['time.year'] <= y1))

    def _smb_file(gcm_, scen, yr):
        m = glob.glob(f'{FORCING_ROOT}/{gcm_}/{scen}/{SMB_PARAM}/acabf/v*/acabf_AIS_{gcm_}_{scen}_{SMB_PARAM}_v*_{yr}.nc')
        return m[0] if m else None

    def _smb_year_myr(gcm_, scen, yr, rho_ice, yts):
        f = _smb_file(gcm_, scen, yr)
        if f is None:
            return None
        a = xr.open_dataset(f)['acabf']  # kg m-2 s-1, monthly
        return (a.mean('time') * yts / rho_ice).to_numpy()

    if use_ocean or use_smb:
        ref_grid = xr.open_dataset(sorted(glob.glob(f'{FORCING_ROOT}/{gcm}/historical/ocean/tf/v3/*.nc'))[0])
        gxx, gyy = np.meshgrid(ref_grid['x'].values, ref_grid['y'].values)
        vtx, wts, simplex = _interp_weights(gxx, gyy, md.mesh.x, md.mesh.y)
        to_mesh = lambda a: _interp_apply(a, vtx, wts, simplex)

    if use_ocean:
        clim_path = f'{model_dir}/AIS3_proj_ssa_tfclim_{gcm}_{CLIM_Y0}-{CLIM_Y1}.npy'
        if not os.path.exists(clim_path):
            print(f"   building {gcm} historical TF climatology {CLIM_Y0}-{CLIM_Y1} on the mesh...")
            clim = _load_tf(gcm, 'historical', CLIM_Y0, CLIM_Y1)
            assert clim.sizes['time'] == CLIM_Y1 - CLIM_Y0 + 1, clim.sizes
            clim = clim.mean('time', skipna=True).to_numpy()
            np.save(clim_path + f'.{os.uname().nodename}.{os.getpid()}.tmp.npy', np.stack([to_mesh(clim[k]) for k in range(clim.shape[0])]))
            os.replace(clim_path + f'.{os.uname().nodename}.{os.getpid()}.tmp.npy', clim_path)
        tf_clim = np.load(clim_path)
        tf_da = _load_tf(gcm, scenario, int(years[0]), int(years[-1]))
        tf_years = tf_da['time.year'].values
        assert np.allclose(-tf_da['z'].values, -tf_depths0) or np.allclose(tf_da['z'].values, tf_depths0), 'TF depth levels differ from TF0'
        tf_cols = np.empty((tf0.shape[0], nv + 1, years.size))
        for j, yr in enumerate(years):
            src = yr if yr in tf_years else tf_years[tf_years <= yr].max()
            slab = tf_da.isel(time=int(np.where(tf_years == src)[0][0])).to_numpy()
            for k in range(tf0.shape[0]):
                anom = np.nan_to_num(to_mesh(slab[k]) - tf_clim[k], nan=0.0)
                tf_cols[k, :nv, j] = tf0[k] + anom
            tf_cols[:, nv, j] = col_times[j]
            if j in (0, years.size - 1):
                fl = np.asarray(md.mask.ocean_levelset).ravel() < 0
                d = tf_cols[:, :nv, j] - tf0
                print(f"      TF {yr} (from {src}): anomaly over floating vertices, all depths: "
                      f"mean {d[:, fl].mean():+.3f} C, p5 {np.percentile(d[:, fl], 5):+.3f}, p95 {np.percentile(d[:, fl], 95):+.3f}")
        md.basalforcings.tf = [tf_cols[k] for k in range(tf0.shape[0])]
        md.basalforcings.tf_depths = tf_depths0.copy()
        del tf_cols
    else:
        md.basalforcings.tf = [np.append(tf0[k], TIME_SENTINEL).reshape(-1, 1) for k in range(tf0.shape[0])]
        md.basalforcings.tf_depths = tf_depths0.copy()

    md.smb = pyissm.model.classes.smb.default(md.smb)
    if use_smb:
        rho_ice, yts = float(md.materials.rho_ice), float(md.constants.yts)
        sclim_path = f'{model_dir}/AIS3_proj_ssa_smbclim_{gcm}_{SMB_PARAM}_{CLIM_Y0}-{CLIM_Y1}.npy'
        if not os.path.exists(sclim_path):
            print(f"   building {gcm} {SMB_PARAM} historical SMB climatology {CLIM_Y0}-{CLIM_Y1}...")
            acc = [_smb_year_myr(gcm, 'historical', y, rho_ice, yts) for y in range(CLIM_Y0, CLIM_Y1 + 1)]
            assert all(a is not None for a in acc), 'missing historical acabf years'
            np.save(sclim_path + f'.{os.uname().nodename}.{os.getpid()}.tmp.npy', to_mesh(np.mean(acc, axis=0)))
            os.replace(sclim_path + f'.{os.uname().nodename}.{os.getpid()}.tmp.npy', sclim_path)
        smb_clim = np.load(sclim_path)
        mb = np.empty((nv + 1, years.size))
        last_a, last_src = None, None
        for j, yr in enumerate(years):
            a = _smb_year_myr(gcm, scenario, yr, rho_ice, yts)
            if a is None:
                assert last_src is not None, f'no acabf for {gcm}/{scenario} {yr}'
                a = last_a
            else:
                last_a, last_src = a, yr
            mb[:nv, j] = smb0 + np.nan_to_num(to_mesh(a) - smb_clim, nan=0.0)
            mb[nv, j] = col_times[j]
            if j in (0, years.size - 1):
                print(f"      SMB {yr}: mesh-mean anomaly {np.mean(mb[:nv, j] - smb0):+.4f} m ice eq/yr "
                      f"(SMB0 mesh-mean {smb0.mean():.4f})")
        md.smb.mass_balance = mb
    else:
        md.smb.mass_balance = smb0.copy()

    if melt_interp:
        md.groundingline.melt_interpolation = melt_interp
    if static_melt:
        from scipy.spatial import cKDTree
        assert resume_file == init_path and t1 <= 2050.0 + 1e-6, 'STATICMELT test: one chunk from the initial state'
        assert not (use_ocean or use_smb) and (not larmip_region or larmip_mode == 'step'), 'STATICMELT: ctrl/LARMIP step'
        m0 = np.asarray(np.load(f'{model_dir}/AIS3_proj_ssa_ctrl_2019-2050_history.npz')['melt'][0], dtype=float)
        xm, ym = np.asarray(md.mesh.x).ravel(), np.asarray(md.mesh.y).ravel()
        src = (m0 != 0) & (np.asarray(md.mask.ocean_levelset).ravel() < 0)
        m0 = m0[src][cKDTree(np.column_stack([xm[src], ym[src]])).query(np.column_stack([xm, ym]))[1]]
        anom = np.nan_to_num(np.asarray(md.basalforcings.melt_anomaly, dtype=float).ravel()) if larmip_region else 0.0
        bf = pyissm.model.classes.basalforcings.default()
        bf.groundedice_melting_rate = np.asarray(md.basalforcings.groundedice_melting_rate, dtype=float).copy()
        bf.geothermalflux = np.asarray(md.basalforcings.geothermalflux, dtype=float).copy()
        bf.floatingice_melting_rate = m0 + anom
        md.basalforcings = bf
        fl_ = np.asarray(md.mask.ocean_levelset).ravel() < 0
        print(f"   STATICMELT: default basal forcing, CTRL t=2019.1 melt extended past the GL; floating-vertex "
              f"mean {np.mean(m0[fl_]):.2f} m/yr (+ anomaly mean {np.mean(np.broadcast_to(anom, m0.shape)[fl_]):.2f}); "
              f"melt_interpolation {md.groundingline.melt_interpolation}")

    # -- ISMIP7 ice-shelf collapse (path C): mask per year, applied between sub-solves -----------
    if collapse:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        ds_c = xr.open_dataset(collapse_file, decode_times=False)
        assert np.allclose(ds_c['x'].values, ref_grid['x'].values) and \
            np.allclose(ds_c['y'].values, ref_grid['y'].values), 'collapse mask grid != TF grid'
        cyears = ds_c['time'].values.astype(int)
        # Smoke tests only: read the mask this many years ahead (the 2019 mask is ice-free on our mesh).
        cshift = int(os.environ.get('AIS3_PROJ_COLLAPSE_SMOKE_SHIFT', '0'))
        assert cshift == 0 or smoke, 'AIS3_PROJ_COLLAPSE_SMOKE_SHIFT is for smoke tests only'
        fl = (np.asarray(md.mask.ocean_levelset).ravel() < 0) & (np.asarray(md.mask.ice_levelset).ravel() < 0)
        segs = []  # (first year, mesh mask), one per year in which the mask changes on the mesh
        for yr in years:
            src = int(yr + cshift) if yr + cshift in cyears else int(cyears[cyears <= yr + cshift].max())
            grid = ds_c['mask'].sel(time=src).values
            on = np.nan_to_num(to_mesh(grid.astype(float)), nan=0.0) >= 0.5
            if segs and np.array_equal(on, segs[-1][1]):
                continue
            if segs and (segs[-1][1] & ~on).any():
                print(f"      WARNING collapse mask {src}: {int((segs[-1][1] & ~on).sum())} vertices un-collapse")
            segs.append((int(yr), on))
            print(f"      collapse mask {yr} (from {src}): {int(on.sum())} vertices, {int((on & fl).sum())} "
                  f"floating ice at chunk start; {int(grid.sum()) * 64} km^2 on the 8 km grid")
        elements0 = np.asarray(md.mesh.elements).astype(int) - 1
        fric0 = np.asarray(md.friction.coefficient, dtype=float).ravel()
        collapsed_at = np.full(nv, np.nan)

        def _detached(ice, oln):
            """Ice vertices outside every piece that holds an anchoring element (fully grounded, basal
            friction coefficient > 0 at all three vertices), where pieces are fully-ice elements joined
            across shared edges. Vertex-only joins, partly grounded elements and frictionless grounded
            spots (pinning points that were floating in the inversion) do not anchor: such pieces
            reached 1e15-1e17 m/yr in smoke 180862834."""
            ie = np.where(np.all(ice[elements0], axis=1))[0]
            e = elements0[ie]
            ed = np.sort(np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]]), axis=1)
            key = ed[:, 0].astype(np.int64) * nv + ed[:, 1]
            owner = np.tile(np.arange(ie.size), 3)
            order = np.argsort(key, kind='stable')
            same = key[order][1:] == key[order][:-1]
            a_, b_ = owner[order][:-1][same], owner[order][1:][same]
            _, lab = connected_components(coo_matrix((np.ones(a_.size), (a_, b_)), shape=(ie.size, ie.size)),
                                          directed=False)
            anchored = np.zeros(lab.max() + 1, bool)
            np.logical_or.at(anchored, lab, np.all((oln[e] > 0) & (fric0[e] > 0), axis=1))
            held = np.zeros(nv, bool)
            held[e[anchored[lab]].ravel()] = True
            return ice & ~held

        def apply_collapse(md, on, t):
            """Make masked floating ice ice-free (ice_levelset +1, thickness min_thickness), together with
            the unanchored pieces (see _detached) this leaves, which would otherwise have rigid-body
            modes. With ismovingfront=0 the new edge is an ordinary calving front."""
            il = np.asarray(md.mask.ice_levelset, dtype=float).ravel().copy()
            oln = np.asarray(md.mask.ocean_levelset, dtype=float).ravel()
            ice = il < 0
            cut = on & ice & (oln < 0)
            # every unanchored piece goes, also when nothing is cut this year: pieces left away from
            # the cut (grounding-line migration between years) gave 1e15 m/yr rigid-body modes and the
            # 2089 failures; removing all of them (113 vertices) gave a normal solve (worklog 6.15)
            berg = _detached(ice & ~cut, oln)
            gone = cut | berg
            if not gone.any():
                print(f"      collapse at {t:.1f}: no masked floating ice, no unanchored ice")
                return
            il[gone] = 1.0
            md.mask.ice_levelset = il
            H = np.asarray(md.geometry.thickness, dtype=float).ravel().copy()
            bed = np.asarray(md.geometry.bed, dtype=float).ravel()
            H[gone] = float(md.masstransport.min_thickness)
            base = np.asarray(md.geometry.base, dtype=float).ravel().copy()
            # removed vertices take the flotation state of 1 m ice (removed grounded pieces mostly float)
            hfl = -H[gone] * float(md.materials.rho_ice) / float(md.materials.rho_water)
            floats = bed[gone] < hfl
            base[gone] = np.where(floats, hfl, bed[gone])
            oln_new = oln.copy()
            oln_new[gone] = np.where(floats, -1.0, np.where(oln[gone] > 0, oln[gone], 1.0))
            md.mask.ocean_levelset = oln_new
            md.geometry.thickness, md.geometry.base, md.geometry.surface = H, base, base + H
            collapsed_at[gone & np.isnan(collapsed_at)] = t
            print(f"      collapse at {t:.1f}: {int(cut.sum())} masked floating ice vertices + {int(berg.sum())} "
                  f"unanchored -> ice-free; ice vertices left {int((il < 0).sum())}")

    # -- Solver setup, same physics as historical_dhdt_tuning_ssa -------------------------------
    md.inversion.iscontrol = 0
    md.verbose.solution = 1
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.isthermal = 0
    md.transient.isgroundingline = 1
    md.groundingline.migration = 'SubelementMigration'
    md.transient.requested_outputs = [
        'Thickness', 'Surface', 'Base', 'MaskOceanLevelset', 'Vx', 'Vy', 'Vel',
        'BasalforcingsFloatingiceMeltingRate', 'SmbMassBalance',
        'IceVolume', 'IceVolumeAboveFloatation', 'GroundedArea', 'FloatingArea',
        'TotalSmb', 'TotalFloatingBmb', 'TotalGroundedBmb', 'GroundinglineMassFlux', 'IcefrontMassFlux',
    ]
    md.timestepping.time_step = TIME_STEP
    md.settings.output_frequency = 1 if smoke else int(round(1.0 / TIME_STEP))  # annual
    md.settings.solver_residue_threshold = 1e-3
    if collapse:  # the 2089 collapse sub-solve was rejected at residue 1.9e-3 (twice), worklog 6.15
        md.settings.solver_residue_threshold = float(os.environ.get('AIS3_PROJ_RESIDUE', '1e-2'))

    run_name = f'AIS3_proj_ssa_{exp}_{current_time:.0f}-{t1:.0f}' + ('_smoke' if smoke else '')
    md.cluster = cluster
    # Wait budget (includes the inner job's queue wait) = this driver's own walltime minus 90 min
    # for loading results, the history file and the checkpoint save.
    wall_min = 46 * 60
    try:
        qs = subprocess.run(['qstat', '-f', os.environ['PBS_JOBID']], capture_output=True, text=True).stdout
        hh, mm, _ = re.search(r'Resource_List.walltime = (\d+):(\d+):(\d+)', qs).groups()
        wall_min = int(hh) * 60 + int(mm)
    except (KeyError, AttributeError):
        pass

    # One solve per chunk; with collapse, one per stretch between January-1st mask changes, the
    # collapse applied to the adopted state before each. Each sub-solve gets only the forcing
    # columns around its own years (the full chunk's TF is ~10 GB).
    sub_edges = [current_time] + ([float(y) for y, _ in segs if current_time < y < t1] if collapse else []) + [t1]
    tf_full, mb_full = getattr(md.basalforcings, 'tf', None), md.smb.mass_balance  # no tf: STATICMELT test

    def _window(a, s0, s1):
        a = np.asarray(a)
        if a.ndim != 2 or a.shape[1] < 2:
            return a
        keep = (a[-1] >= s0 - 1.0) & (a[-1] <= s1 + 1.0)
        return a[:, keep] if keep.any() else a

    import time as _time
    t_wall0 = _time.time()
    # Inner walltime floor per collapse sub-solve: 4 h until 2026-10-23, when 2148 used 3h42 (years
    # after ~2145 hit the iteration cap on 6-9 of 10 steps, worklog 6.18)
    collapse_inner_min = int(os.environ.get('AIS3_PROJ_COLLAPSE_INNER_MIN', '480'))
    steps_list = []
    failed = False

    def _solve_watched(md, budget_min):
        """solve() without pyissm's blind lock-file wait (collapse runs only, for now): submit, then
        poll for the lock file AND for the end of the inner job (the PBS usage footer in its outlog).
        A job that has ended without writing its lock file has failed; pyissm would wait out the whole budget (a NaN at
        2059 left the collapse driver idle for 23 h, worklog 6.12). Returns (md, ok)."""
        # waitonlock must stay > 0 in the marshalled model -- ISSM writes the lock file only then (the
        # first version set it to 0 and every solve looked failed); batch=True makes solve() return
        # straight after qsub instead of running its own wait.
        md.settings.waitonlock = max(1, int(budget_min))
        # a reused run directory may hold a stale lock (would load old results) or a finished outlog
        # (would look like a failure): set them aside first (renamed, not deleted)
        for ext in ('lock', 'outlog', 'errlog', 'outbin'):
            f = f'{md.cluster.executionpath}/{md.miscellaneous.name}/{md.miscellaneous.name}.{ext}'
            if os.path.exists(f):
                os.rename(f, f'{f}.prev{int(_time.time())}')
        md = pyissm.model.execute.solve(md, 'Transient', batch=True, load_only=False, runtime_name=False)
        rdir = f'{md.cluster.executionpath}/{md.private.runtimename}'
        lock = f'{rdir}/{md.miscellaneous.name}.lock'
        outlog = f'{rdir}/{md.miscellaneous.name}.outlog'  # PBS appends its usage footer when the job ends
        t_start, ended = _time.time(), 0
        while True:
            _time.sleep(60)
            if os.path.exists(lock):
                break
            try:
                ended = ended + 1 if 'Exit Status:' in open(outlog, errors='replace').read()[-4000:] else 0
            except OSError:
                ended = 0
            if ended >= 2:  # job over for two polls and still no lock file
                print(f"   inner job for {md.miscellaneous.name} ended without a lock file -- failed; see {outlog}")
                return md, False
            if (_time.time() - t_start) / 60 > budget_min:
                print(f"   wait budget {budget_min} min used up for {md.miscellaneous.name}")
                return md, False
        return pyissm.model.execute.load_results_from_cluster(md), True
    for s0, s1 in zip(sub_edges[:-1], sub_edges[1:]):
        if steps_list:  # carry the previous sub-solve's end state over
            # Not enough driver time left for another sub-solve (inner time + 1 h queue allowance):
            # end the chunk at s0 instead, so it still writes its checkpoint and queues the next one.
            need = max(collapse_inner_min, 60 * int(np.ceil(10.0 * (s1 - s0) * 12.6 / 60))) + 60
            if wall_min - 90 - (_time.time() - t_wall0) / 60 < need:
                print(f"-- driver time short for {s0:.1f}-{s1:.1f}: ending this chunk at {s0:.1f}")
                t1 = s0
                break
            adopt_final_state_2d(md)
            md.results.TransientSolution = []
        sub_name = run_name if len(sub_edges) == 2 else f'{run_name}_sub{s0:.1f}'
        if collapse:
            apply_collapse(md, [m for y, m in segs if y <= s0 + 1e-9][-1], s0)
        if len(sub_edges) > 2:
            if tf_full is not None:
                md.basalforcings.tf = [_window(a, s0, s1) for a in tf_full]
            md.smb.mass_balance = _window(mb_full, s0, s1)
        md.timestepping.start_time = s0
        md.timestepping.final_time = s1
        md.miscellaneous.name = sub_name
        # Inner solve: 3x the historical run's 12.6 min per model year (20 h for 2019-2050, 16 h for a
        # 25-yr chunk), enough for a solve running at ~1/3 of normal speed.
        # AIS3_PROJ_TIME_FACTOR overrides the 3x, for reruns after slow-node walltime kills (worklog 6.14)
        md.cluster.time = max(60, 60 * int(np.ceil(float(os.environ.get('AIS3_PROJ_TIME_FACTOR', '3'))
                                                   * (s1 - s0) * 12.6 / 60)))
        if collapse:  # steps after a large collapse hit the 100-iteration cap (~12 min each, smoke 180869837)
            md.cluster.time = max(collapse_inner_min, 60 * int(np.ceil(10.0 * (s1 - s0) * 12.6 / 60)))
        md.settings.waitonlock = max(30, int(wall_min - 90 - (_time.time() - t_wall0) / 60))
        print(f"   inner solve time {md.cluster.time} min; wait budget {md.settings.waitonlock} min")
        print(f"-- Submitting and waiting on {sub_name}...")
        if collapse:
            md, ok = _solve_watched(md, md.settings.waitonlock)
            if not ok:  # NaNs here were not reproducible (2059 failed once, passed on rerun): retry once
                budget = max(30, int(wall_min - 90 - (_time.time() - t_wall0) / 60))
                if budget > md.cluster.time + 60:
                    print(f"-- retrying {sub_name} once")
                    md, ok = _solve_watched(md, budget)
            if not ok:
                # Keep what this chunk did: end it at s0 with a checkpoint (the state the failed
                # solve started from, collapse at s0 already applied) and do not chain.
                failed, t1 = True, s0
                print(f"-- {sub_name} failed: ending this chunk at {s0:.1f}, chain stopped")
                if not steps_list:
                    raise SystemExit(f"{sub_name}: first sub-solve of the chunk failed -- nothing to save")
                break
        else:
            md = pyissm.model.execute.solve(md, 'Transient', load_only=False, runtime_name=False)

        ts = md.results.TransientSolution
        # md.results.TransientSolution is a pyissm `solution` holding a `.steps` list; attribute access
        # on the solution itself returns every step stacked, not one step.
        sub_steps = list(ts.steps) if hasattr(ts, 'steps') else (ts if isinstance(ts, list) else [ts])
        sub_times = np.array([float(getattr(s, 'time', np.nan)) for s in sub_steps])
        # The solve must have reached s1. A solve killed at its walltime leaves a partial outbin that
        # pyissm loads without complaint; adopting its last step and saving it as t1 silently skipped
        # 19, 13 and 4 model years in three runs (worklog 6.10). Stop here instead: no history, no
        # checkpoint, no next chunk.
        if sub_times.size == 0 or abs(sub_times[-1] - s1) > 1e-6:
            raise SystemExit(f"{sub_name}: solve ended at t={sub_times[-1] if sub_times.size else 'none'}, not {s1} "
                             f"(inner job killed or failed?) -- no checkpoint written, chain stopped")
        steps_list += sub_steps
    run_name = f'AIS3_proj_ssa_{exp}_{current_time:.0f}-{t1:.0f}' + ('_smoke' if smoke else '')
    md.miscellaneous.name = run_name
    times = np.array([float(getattr(s, 'time', np.nan)) for s in steps_list])

    def _stack(field):
        return np.stack([np.asarray(getattr(s, field), dtype=np.float32).ravel() for s in steps_list])

    scalar_names = ['IceVolume', 'IceVolumeAboveFloatation', 'GroundedArea', 'FloatingArea', 'TotalSmb',
                    'TotalFloatingBmb', 'TotalGroundedBmb', 'GroundinglineMassFlux', 'IcefrontMassFlux']
    scalars = {n: np.array([float(np.asarray(getattr(s, n, np.nan)).ravel()[0]) for s in steps_list])
               for n in scalar_names}
    hist_path = f'{model_dir}/{run_name}_history.npz'
    np.savez(hist_path, times=times, thickness=_stack('Thickness'),
             ocean_levelset=_stack('MaskOceanLevelset'), vel=_stack('Vel'),
             melt=_stack('BasalforcingsFloatingiceMeltingRate'), smb=_stack('SmbMassBalance'),
             **({'collapsed_at': collapsed_at} if collapse else {}),
             **{f'scalar_{n}': v for n, v in scalars.items()})
    rho_sw, A_OCEAN = 1028.0, 3.625e14
    vaf = scalars['IceVolumeAboveFloatation']
    print(f"   saved {hist_path} ({times.size} outputs, t={times[0]:.2f}..{times[-1]:.2f})")
    print(f"   VAF change over chunk {vaf[-1] - vaf[0]:+.4e} m^3 "
          f"(= {-(vaf[-1] - vaf[0]) * float(md.materials.rho_ice) / rho_sw / A_OCEAN * 1000:+.2f} mm SLE); "
          f"grounded area {scalars['GroundedArea'][0]:.4e} -> {scalars['GroundedArea'][-1]:.4e} m^2")
    for n in ('TotalSmb', 'TotalFloatingBmb', 'GroundinglineMassFlux', 'IcefrontMassFlux'):
        print(f"   {n}: first {scalars[n][0]:.4e}, last {scalars[n][-1]:.4e}")

    if smoke:
        print("-- Smoke test complete (no checkpoint, no chaining).")
    else:
        if not failed:  # after a failed sub-solve md already holds the adopted state at t1 = s0
            adopt_final_state_2d(md)
        else:  # ...but its timestepping still says the failed sub-solve's s0 -> s1
            md.timestepping.start_time = md.timestepping.final_time = t1
        md.results.TransientSolution = []
        set_static_forcing(md, tf0, tf_depths0, smb0)
        ckpt = f'{model_dir}/AIS3_proj_ssa_{exp}_{t1:.0f}.nc'
        pyissm.model.io.save_model(md, ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)
        print(f"   saved checkpoint {ckpt}")
        if failed:
            raise SystemExit(f"{exp}: sub-solve from {t1:.1f} failed; checkpoint {ckpt} holds its starting state "
                             f"-- chain stopped for diagnosis")
        if t1 < end_year and not os.path.exists(stop_path):
            env = ','.join(f'{k}={os.environ[k]}' for k in
                           ('AIS3_STEPS', 'AIS3_PROJ_SCENARIO', 'AIS3_PROJ_GCM', 'AIS3_PROJ_FORCING',
                            'AIS3_PROJ_MELT', 'AIS3_PROJ_END', 'AIS3_PROJ_LARMIP_REGION',
                            'AIS3_PROJ_LARMIP_MODE', 'AIS3_PROJ_LARMIP_AMP', 'AIS3_LARMIP_MASK',
                            'AIS3_LARMIP_MASK_VAR', 'AIS3_PROJ_COLLAPSE', 'AIS3_PROJ_COLLAPSE_VER',
                            'AIS3_PROJ_FRICTION', 'AIS3_PROJ_TIME_FACTOR', 'AIS3_PROJ_RESIDUE', 'AIS3_PROJ_COLLAPSE_INNER_MIN',
                            'AIS3_PROJ_CHUNK_YEARS')
                       if os.environ.get(k))
            log = f'/g/data/au88/jh7060/ACCESS-AIS3/config/logs/proj_ssa_{exp}_{t1:.0f}-next'
            out = subprocess.run(['qsub', '-v', env, '-N', f'proj_{exp}'[:15], '-o', log + '.out', '-e', log + '.err',
                                  '/g/data/au88/jh7060/ACCESS-AIS3/config/launch_projection_ssa.pbs'],
                                 capture_output=True, text=True)
            print(f"   next chunk submitted: {out.stdout.strip()} {out.stderr.strip()}")
        elif t1 < end_year:
            print(f"   {stop_path} exists -- not submitting the next chunk")
        else:
            print(f"-- {exp} complete through {end_year:.0f}")
