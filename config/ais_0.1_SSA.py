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
  3. melt_gamma_tuning_ssa -- RUN. Coarse 9-point gamma_0 sweep pick (AIS3_melt_gamma_tuning_
     ssa.nc) was superseded by the per-basin deltaT_basin secant refit chain
     (ssa_melt_deltaT_basin_refit.py / ssa_melt_deltaT_refit_recalibrate.py), which converged
     on gamma_0=300 with a refit deltaT_basin -- the actual accepted calibration, reconstructed
     as a genuinely complete model by finalize_ssa_melt_calibration.py -> AIS3_melt_final_ssa.nc
     (2026-09-21, after historical_dhdt_tuning_ssa's first run revealed AIS3_melt_gamma_tuning_
     ssa.nc itself was never updated with the refit result).
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
## Stage 3: Ocean melt (gamma) calibration, SSA mesh
## ------------------------------------
# J1-only (basin-aggregated Gt/yr vs. Paolo/Adusumilli, official IMBIE2 basins) rather than
# the HO track's full J1+J2+J3+Monte Carlo (config/melt_ismip7_calibration.py) -- J1 alone
# already proved robust on HO (SS4b: basin-aggregated fitting was NOT gamed the way naive
# per-vertex RMSE was), and this track is explicitly meant to be the cheap/simple one. No
# vertical-layer projection needed here (unlike the HO track's on_base/x2d/elements2d
# machinery) -- the SSA mesh IS the 2D mesh, md.mesh.x/y/elements are already what's needed.
# KNOWN GAP, same as HO's initial pass: deltaT_basin held fixed at the published prior, not
# re-derived per candidate gamma_0 -- HO's own experience (SS4c) showed this can matter a
# lot; deferred here too, revisit if this track's melt calibration needs to be trusted in
# detail rather than as a reasonable working value.
if 'melt_gamma_tuning_ssa' in steps:
    print("-------------------------------------------------------------")
    print(" OCEAN BASAL-MELT GAMMA CALIBRATION (SSA)")
    print("-------------------------------------------------------------")

    from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
    import pandas as pd
    import copy as _copy

    ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
    ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                     'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
    GAMMA0_FILE = '/g/data/au88/ismip6/2300/forcings/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc'
    BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
    MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'
    TIME_SENTINEL = 1e9
    GAMMA0_VALUES = [100.0, 300.0, 1000.0, 3000.0, 5537.7, 8306.6, 11075.45, 15000.0, 25000.0]

    print("-- Loading relaxed SSA model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_SSA_relaxed.nc')
    # BUGFIX (2026-09-02, found via a real OOM at 175.89GB/190GB after only 1 of 9 candidates
    # submitted): AIS3_SSA_relaxed.nc carries its full 400-timestep transient history (Vel/
    # Thickness/Surface/Base/MaskOceanLevelset x 400 steps x ~1.6M vertices) -- none of it is
    # used below (only geometry/mask/mesh/materials at the relaxed endpoint matter for melt
    # calibration), but it gets carried along and re-copy.deepcopy'd every submit-loop
    # iteration, ballooning memory fast. Strip it once, right after loading.
    md.results.TransientSolution = []

    print("-- Configuring ISMIP6 basal melt parameterisation (OFFICIAL IMBIE2 basins)...")
    md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)
    ds_basin = xr.open_dataset(BASIN_FILE)
    basin_grid = ds_basin['basinNumber'].values
    bx, by = ds_basin['x'].values, ds_basin['y'].values
    num_basins = int(np.nanmax(basin_grid)) + 1
    print(f"   {num_basins} basins found in official IMBIE2 grid (expect 16)")

    elx = np.asarray(md.mesh.elements).astype(int) - 1
    mesh_x = np.asarray(md.mesh.x).ravel()
    mesh_y = np.asarray(md.mesh.y).ravel()
    ecx = mesh_x[elx[:, :3]].mean(axis=1)
    ecy = mesh_y[elx[:, :3]].mean(axis=1)
    bxx, byy = np.meshgrid(bx, by)
    basin_lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
    basin_id_elements = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)
    md.basalforcings.basin_id = basin_id_elements  # per-element, no 3D projection needed on a 2D mesh
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

    print("-- Interpolating Zhou ocean thermal-forcing climatology onto the mesh...")
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

    print("-- Assigning cluster and updating settings...")
    md.inversion.iscontrol = 0
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.ismasstransport = 1
    md.masstransport.spcthickness = np.full(md.mesh.numberofvertices, np.nan)
    md.smb.mass_balance = np.zeros(md.mesh.numberofvertices)
    md.timestepping.start_time = 0
    md.timestepping.final_time = 0.01
    md.timestepping.time_step = 0.01
    md.transient.requested_outputs = ['default', 'BasalforcingsFloatingiceMeltingRate']
    md.cluster = cluster  # plain 48c/190GB config -- SSA mesh, no hugemem needed
    md.settings.waitonlock = 0

    print("-- Loading J1 target (Paolo/Adusumilli basin-aggregated melt)...")
    obs_df = pd.read_csv(MELT_OBS_CSV)
    bmb_obs = obs_df['BMR (Gt/yr)'].values

    elx_tri = np.asarray(md.mesh.elements).astype(int) - 1
    tri_x, tri_y = mesh_x[elx_tri], mesh_y[elx_tri]
    elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                              - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
    vertex_area = np.zeros(md.mesh.numberofvertices)
    for c in range(3):
        np.add.at(vertex_area, elx_tri[:, c], elem_area / 3.0)
    counts = np.zeros(md.mesh.numberofvertices)
    basin_id_vtx = np.zeros(md.mesh.numberofvertices)
    for c in range(3):
        np.add.at(basin_id_vtx, elx_tri[:, c], basin_id_elements - 1)
        np.add.at(counts, elx_tri[:, c], 1)
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

    if save:
        print("-- Loading gamma_0 sweep results and comparing to Paolo/Adusumilli...")
        best = None
        for i, g0 in enumerate(GAMMA0_VALUES):
            name = f'AIS3_melt_gamma_tuning_ssa_g{i}'
            try:
                mdi = pyissm.model.io.load_model(f'{model_dir}/AIS3_SSA_relaxed.nc')
                mdi.results.TransientSolution = []  # same memory fix as the submit branch above
                mdi.miscellaneous.name = name
                mdi.cluster = cluster
                mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
            except Exception as e:
                print(f"   g{i} (gamma_0={g0:.1f}): FAILED to load ({e})")
                continue
            melt = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
            bmb = basin_bmb(melt)
            J1 = np.nanmean(np.abs(bmb - bmb_obs))
            print(f"   g{i} (gamma_0={g0:.1f}): total bmb={bmb.sum():.1f} Gt/yr "
                  f"(target {bmb_obs.sum():.1f}), J1={J1:.3f}")
            if best is None or J1 < best[1]:
                best = (g0, J1, mdi)

        if best is not None:
            print(f"\nBest gamma_0 = {best[0]:.1f} m/yr (J1 = {best[1]:.3f})")
            md_best = best[2]
            # BUGFIX (2026-09-21, found via historical_dhdt_tuning_ssa crashing with
            # "'default' object has no attribute 'tf'"): md_best above is a fresh reload of
            # AIS3_SSA_relaxed.nc with only its TransientSolution results grafted on (via
            # load_only=True) -- basalforcings was NEVER reconfigured back to the ismip6
            # type/config actually used for the submitted run (that config only ever existed
            # on the _copy.deepcopy submitted in the 'else' branch above, which doesn't
            # round-trip through save_model/load_model here). Every OTHER downstream
            # consumer (melt_deltaT_sensitivity_test_ssa.py, ssa_melt_deltaT_basin_refit.py)
            # never hit this because they reconstruct basalforcings from scratch themselves
            # rather than trusting this file's saved config -- historical_dhdt_tuning_ssa is
            # the first to assume otherwise. Re-apply the same ismip6 configuration used for
            # the submitted sweep, fixing gamma_0 to the winning candidate, so this saved
            # file is actually self-consistent with the run it claims to represent.
            md_best.basalforcings = pyissm.model.classes.basalforcings.ismip6(md_best.basalforcings)
            md_best.basalforcings.basin_id = basin_id_elements
            md_best.basalforcings.num_basins = num_basins
            md_best.basalforcings.delta_t = np.nan_to_num(delta_t_per_basin, nan=0.0)
            md_best.basalforcings.islocal = 1
            md_best.basalforcings.tf = tf_list
            md_best.basalforcings.tf_depths = tfz.copy()
            md_best.basalforcings.gamma_0 = float(best[0])
            print(f"Saving to {model_dir}/AIS3_melt_gamma_tuning_ssa.nc")
            pyissm.model.io.save_model(md_best, f'{model_dir}/AIS3_melt_gamma_tuning_ssa.nc')
        else:
            print("   No sweep runs loaded successfully -- nothing to save.")
    else:
        print(f"-- Submitting gamma_0 sweep ({len(GAMMA0_VALUES)} runs)...")
        for i, g0 in enumerate(GAMMA0_VALUES):
            mdi = _copy.deepcopy(md)
            mdi.basalforcings.gamma_0 = float(g0)
            mdi.miscellaneous.name = f'AIS3_melt_gamma_tuning_ssa_g{i}'
            print(f"   run g{i}: gamma_0={g0:.1f}")
            pyissm.model.execute.solve(mdi, 'Transient', load_only=False, runtime_name=False)


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
#   AIS3_PROJ_MELT      ref (gamma_0=300 + refit deltaT, AIS3_melt_final_ssa.nc) | g<i> (refit
#                       candidate i of deltaT_refit_state_ssa.json)                (default ref)
#   AIS3_PROJ_END       final year                         (default 2300, or 2100 for ssp370)
#   AIS3_PROJ_SMOKE=1   0.2-yr test chunk into *_smoke names, no checkpoint, no chaining
#
# Forcing is an anomaly on the reference state the model was calibrated with (worklog 4c):
#   TF  = TF0  (Zhou climatology, as in the historical run) + [TF_gcm(t)  - TF_gcm(1995-2014)]
#   SMB = SMB0 (RACMO 1995-2014 mean)                         + [SMB_gcm(t) - SMB_gcm(1995-2014)]
# ctrl applies no anomaly. GCM SMB is dEBM2-8000m (the only product for all 7 ISMIP7 GCMs); no
# elevation feedback (dacabfdz) yet. Anomalies are annual, placed at mid-year; NaN (outside a
# source grid) -> zero anomaly. Years past the end of a GCM file hold its last year.
#
# Chunks: 2019 -> 2050 -> 2100 -> ... -> END, each one PBS submission (~10 h of the 48-cpu
# solve per 50 yr at the historical run's 12.6 min/model-yr). Each chunk writes a NEW
# checkpoint AIS3_proj_ssa_{exp}_{year}.nc (state adopted from the last step, forcing reset to
# the static baseline, results stripped) and a per-chunk history npz, then qsubs the next chunk
# unless models/AIS3_proj_ssa_{exp}.STOP exists. Nothing is overwritten.
if 'projection_ssp_ssa' in steps:

    print("-------------------------------------------------------------")
    print(" FUTURE PROJECTION, SSA (ANOMALY-FORCED, CHUNKED)")
    print("-------------------------------------------------------------")
    import subprocess

    FORCING_ROOT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS'
    SMB_PARAM = 'dEBM2-8000m'
    CLIM_Y0, CLIM_Y1 = 1995, 2014
    START_YEAR = 2019.0
    TIME_STEP = 0.1
    TIME_SENTINEL = 1e9
    CHUNK_EDGES = [2019.0, 2050.0, 2100.0, 2150.0, 2200.0, 2250.0, 2300.0]

    scenario = os.environ.get('AIS3_PROJ_SCENARIO', 'ctrl')
    gcm = os.environ.get('AIS3_PROJ_GCM', '')
    forcing_mode = os.environ.get('AIS3_PROJ_FORCING', 'both')
    melt_choice = os.environ.get('AIS3_PROJ_MELT', 'ref')
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
    if smoke:
        t1 = current_time + 0.2
    print(f"-- Chunk {current_time:.1f} -> {t1:.1f} from {resume_file}")
    md = pyissm.model.io.load_model(resume_file)
    md.results.TransientSolution = []
    nv = md.mesh.numberofvertices
    assert abs(float(md.timestepping.final_time) - current_time) < 1e-6, \
        f'checkpoint final_time {md.timestepping.final_time} != {current_time}'

    # -- Melt parameters -------------------------------------------------------------------
    if melt_choice != 'ref':
        import json
        with open(f'{model_dir}/deltaT_refit_state_ssa.json') as f:
            cand = json.load(f)['candidates'][melt_choice.lstrip('g')]
        md.basalforcings.gamma_0 = float(cand['gamma_0'])
        md.basalforcings.delta_t = np.asarray(cand['history'][-1]['delta_t'], dtype=float)
    print(f"   melt: gamma_0={float(md.basalforcings.gamma_0)}, deltaT range "
          f"[{np.min(md.basalforcings.delta_t):.3f}, {np.max(md.basalforcings.delta_t):.3f}]")

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
        da = xr.open_mfdataset(files, combine='by_coords')['tf']
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
            np.save(clim_path + '.tmp.npy', np.stack([to_mesh(clim[k]) for k in range(clim.shape[0])]))
            os.replace(clim_path + '.tmp.npy', clim_path)
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
            np.save(sclim_path + '.tmp.npy', to_mesh(np.mean(acc, axis=0)))
            os.replace(sclim_path + '.tmp.npy', sclim_path)
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
    md.timestepping.start_time = current_time
    md.timestepping.final_time = t1
    md.timestepping.time_step = TIME_STEP
    md.settings.output_frequency = 1 if smoke else int(round(1.0 / TIME_STEP))  # annual
    md.settings.solver_residue_threshold = 1e-3

    run_name = f'AIS3_proj_ssa_{exp}_{current_time:.0f}-{t1:.0f}' + ('_smoke' if smoke else '')
    md.miscellaneous.name = run_name
    md.cluster = cluster
    md.cluster.time = 60 * (1 if smoke else 24)
    md.settings.waitonlock = 60 * (4 if smoke else 40)  # includes the inner job's queue wait

    print(f"-- Submitting and waiting on {run_name}...")
    md = pyissm.model.execute.solve(md, 'Transient', load_only=False, runtime_name=False)

    # -- History (annual) and scalars -------------------------------------------------------
    ts = md.results.TransientSolution
    steps_list = ts if isinstance(ts, list) else [ts]
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
        adopt_final_state_2d(md)
        md.results.TransientSolution = []
        set_static_forcing(md, tf0, tf_depths0, smb0)
        ckpt = f'{model_dir}/AIS3_proj_ssa_{exp}_{t1:.0f}.nc'
        pyissm.model.io.save_model(md, ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)
        print(f"   saved checkpoint {ckpt}")
        if t1 < end_year and not os.path.exists(stop_path):
            env = ','.join(f'{k}={os.environ[k]}' for k in
                           ('AIS3_STEPS', 'AIS3_PROJ_SCENARIO', 'AIS3_PROJ_GCM', 'AIS3_PROJ_FORCING',
                            'AIS3_PROJ_MELT', 'AIS3_PROJ_END') if os.environ.get(k))
            log = f'/g/data/au88/jh7060/ACCESS-AIS3/config/logs/proj_ssa_{exp}_{t1:.0f}-next'
            out = subprocess.run(['qsub', '-v', env, '-N', f'proj_{exp}'[:15], '-o', log + '.out', '-e', log + '.err',
                                  '/g/data/au88/jh7060/ACCESS-AIS3/config/launch_projection_ssa.pbs'],
                                 capture_output=True, text=True)
            print(f"   next chunk submitted: {out.stdout.strip()} {out.stderr.strip()}")
        elif t1 < end_year:
            print(f"   {stop_path} exists -- not submitting the next chunk")
        else:
            print(f"-- {exp} complete through {end_year:.0f}")
