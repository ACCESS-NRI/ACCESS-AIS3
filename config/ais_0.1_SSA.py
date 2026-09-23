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
  3. melt_gamma_tuning_ssa -- IMPLEMENTED, not yet run. J1-only (basin-aggregated Gt/yr vs.
     Paolo/Adusumilli, official IMBIE2 basins) rather than the HO track's full J1+J2+J3+MC --
     J1 alone proved robust on HO (didn't get gamed the way naive per-vertex RMSE did), and
     this track is meant to be the cheap/simple one. deltaT_basin held fixed at the published
     prior (same known gap HO started with) -- not re-derived here. AIS3_melt_gamma_tuning_
     ssa.nc.
  4. historical_dhdt_tuning_ssa -- IMPLEMENTED, not yet run. Mirrors ais_0.1.py's
     historical_dhdt_tuning exactly (RACMO SMB 1995-2019 vs. MIPKIT dhdt_cpom, same
     data-coverage caveat: stops at 2019, not 2025). AIS3_historical_1995_2019_SSA.nc.

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

all_steps = ['ssa_inverted_solve_budd', 'ssa_relaxation_budd', 'melt_gamma_tuning_ssa', 'historical_dhdt_tuning_ssa']
steps = ['historical_dhdt_tuning_ssa']  # edit before each qsub -- see launch_ais_0.1_SSA.pbs


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
