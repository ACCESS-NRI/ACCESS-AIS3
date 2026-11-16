"""GPU (hypre BoomerAMG, one A100 GPU + 16 ranks) vs production (MUMPS, 48 cores) on the AIS3 SSA
track, beyond the single stress balance of solver_hypre_gpu_test.py (109 s vs 137 s there, with two
MUMPS fallbacks on the first Picard iterations). Two questions:

1. Does smoothing md.initialization remove the Picard-1 fallback? The initial velocity is vel_obs,
   whose piecewise-constant gap fills give zero strain rate on ~9.5k vertices, so the first Picard
   matrix is near-singular (diagonal over 16.8 orders of magnitude). A few passes of mesh-neighbour
   averaging (gpu_solver_utils.smooth_initial_velocity) break the flat patches.
     SBM  one stress balance from AIS3_SSA_inverted.nc, smoothed start, MUMPS 48 cores
     SBH  the same on the GPU
   Both are compared with case M of solver_gamg_cpu_test.py (unsmoothed start, MUMPS 48 cores).

2. Does the GPU win hold over many time steps? Each step starts from the previous velocity, so after
   the first step the matrices should look like the converged one (110 BoomerAMG its, no fallback).
   The other analyses (mass transport, level sets) stay on MUMPS but run on 16 ranks, not 48.
     TRM  one year of the historical run's setup (AIS3_melt_final_ssa.nc, 1995-1996, dt 0.1 yr,
          ismip6 melt, subelement grounding line), with RACMO 1995 SMB held fixed; MUMPS 48 cores
     TRH  the same on the GPU
   Production start (md.initialization as in the historical run), not smoothed.

GPU cases: hypre BoomerAMG on md.toolkits.StressbalanceAnalysis, KSP_MAX_IT (default 300) before
MUMPS recovery; recovery and all other analyses MUMPS on CUDA vectors (gpu_solver_utils.use_gpu_toolkits).

First round (2026-10-28): smoothing removed the Picard-1 fallback (SBH 86 s vs SBM 141 s, no
recovery). TRH crashed at time step 3 (SEGV after UCX gdr_copy errors: PETSc was passing device
buffers to the CUDA-aware MPI); the GPU queue script now sets PETSC_OPTIONS=-use_gpu_aware_mpi 0.
Rerun as CASES=TRH TAG=_hostmpi. Runs that did not finish are skipped in the comparison.

Env CASES=SBM+SBH+TRM+TRH (default all; '+' because qsub -v splits on ','), TAG suffixes the case
names (rerun without touching an earlier case's directory), PASSES (default 5) smoothing passes.
PHASE 'submit' submits (waitonlock=0) and refuses if a case's execution dir exists;
PHASE 'analyze' compares each case with its reference and reports timings, iterations, fallbacks.
Writes only to execution_gpu_test/ and models/solver_gpu_benchmark_state.json.
"""
import os
import sys
from pathlib import Path
# Site settings (paths, NCI project, ISSM builds): edit config/site.env, not this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ais3_site as site

os.chdir(site.ROOT)
os.environ['ISSM_DIR'] = site.ISSM_DIR
import json
import re
import numpy as np
import pyissm
from gpu_solver_utils import (cpu_cluster, gpu_cluster, use_gpu_toolkits, smooth_initial_velocity,
                              parse_outlog)

PHASE = os.environ.get('PHASE', 'submit')
CASES = [c for c in re.split(r'[,+]', os.environ.get('CASES', '')) if c]
TAG = os.environ.get('TAG', '')
PASSES = int(os.environ.get('PASSES', '5'))
KSP_MAX_IT = os.environ.get('KSP_MAX_IT', '300')
MODEL_DIR = f'{site.ROOT}/models'
EXEC_DIR = f'{site.ROOT}/execution_gpu_test'
STATE = f'{MODEL_DIR}/solver_gpu_benchmark_state.json'
REF_SB = 'AIS3_solvertest_M_mumps'      # unsmoothed start, 48-core MUMPS (solver_gamg_cpu_test.py)

# key: (name, kind, solver)
ALL_CASES = {'SBM': (f'AIS3_bench_sb_mumps_smooth{TAG}', 'sb', 'mumps'),
             'SBH': (f'AIS3_bench_sb_hypre16_smooth{TAG}', 'sb', 'hypre'),
             'TRM': (f'AIS3_bench_tr1y_mumps{TAG}', 'tr', 'mumps'),
             'TRH': (f'AIS3_bench_tr1y_hypre16{TAG}', 'tr', 'hypre')}


def verbose(md):
    md.verbose.solution = 1
    md.verbose.convergence = 1
    md.verbose.solver = 1           # "Petsc: N KSP iterations" per linear solve
    md.settings.solver_residue_threshold = 1e-3   # as in the SSA track
    return md


def stressbalance_model():
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_inverted.nc')
    md.inversion.iscontrol = 0
    return verbose(md)


def transient_model():
    """historical_dhdt_tuning_ssa's setup (ais_0.1_SSA.py), one year, SMB fixed at RACMO 1995."""
    import ccdtools
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_melt_final_ssa.nc')
    md.basalforcings.tf = [np.asarray(t, dtype=float) for t in md.basalforcings.tf]
    md.inversion.iscontrol = 0
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.isthermal = 0
    md.transient.isgroundingline = 1
    md.transient.requested_outputs = ['default', 'Thickness', 'MaskOceanLevelset']
    md.groundingline.migration = 'SubelementMigration'
    md.timestepping.start_time = 1995
    md.timestepping.final_time = 1996
    md.timestepping.time_step = 0.1
    racmo = ccdtools.catalog.DataCatalog().load_dataset('racmo2.4p1_monthly_11km_1979-2023')
    smbgl = racmo['smbgl']
    rx, ry = pyissm.tools.general.ll_to_xy(racmo['lat'].values, racmo['lon'].values, -1)
    smb = (smbgl.sel(time=smbgl['time.year'] == 1995).sum('time') / md.materials.rho_ice).squeeze('height').to_numpy()
    md.smb = pyissm.model.classes.smb.default(md.smb)
    md.smb.mass_balance = pyissm.data.interp.points_to_mesh(rx, ry, smb, md.mesh.x, md.mesh.y)
    return verbose(md)


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {'cases': {}}


if PHASE == 'submit':
    cases = {k: v for k, v in ALL_CASES.items() if not CASES or k in CASES}
    for key, (name, _, _) in cases.items():
        if os.path.exists(f'{EXEC_DIR}/{name}'):
            raise SystemExit(f'{EXEC_DIR}/{name} exists; archive it with mv -n or set TAG before resubmitting {key}')
    state = load_state()
    for kind in ('sb', 'tr'):
        todo = {k: v for k, v in cases.items() if v[1] == kind}
        if not todo:
            continue
        md = stressbalance_model() if kind == 'sb' else transient_model()
        if kind == 'sb':
            info = smooth_initial_velocity(md, PASSES)
            print(f'   smoothed md.initialization: {info}', flush=True)
            state['smoothing'] = info
        mumps = dict(md.toolkits.DefaultAnalysis)
        for key, (name, _, solver) in todo.items():
            md.toolkits.DefaultAnalysis = dict(mumps)
            md.toolkits.RecoveryAnalysis = dict(mumps)
            if hasattr(md.toolkits, 'StressbalanceAnalysis'):
                del md.toolkits.StressbalanceAnalysis
            if solver == 'hypre':
                md = use_gpu_toolkits(md, KSP_MAX_IT)
                md.cluster = gpu_cluster(EXEC_DIR, 16, 180 if kind == 'tr' else 60)
            else:
                md.cluster = cpu_cluster(EXEC_DIR, 180 if kind == 'tr' else 60)
            md.miscellaneous.name = name
            md.settings.waitonlock = 0
            print(f'-- Submitting {key} ({name})', flush=True)
            pyissm.model.execute.solve(md, 'Stressbalance' if kind == 'sb' else 'Transient',
                                       load_only=False, runtime_name=False)
            state['cases'][key + TAG] = {'name': name, 'kind': kind, 'solver': solver}
        del md
    with open(STATE, 'w') as f:
        json.dump(state, f, indent=1)
    print(f'DONE submitting {list(cases)}; state in {STATE}. Resubmit with PHASE=analyze once they finish.')

elif PHASE == 'analyze':
    state = load_state()
    logs = {}

    def show(key, lg):
        its = lg.get('ksp_its', [])
        print(f'  {key}: core {lg.get("core_s")} s, linear solver {lg.get("solver_s")} s, init {lg.get("init_s")} s, '
              f'walltime {lg.get("walltime")}, SU {lg.get("su")}, exit {lg.get("exit_status")}')
        print(f'     Picard its per stress balance {lg.get("picard_per_solve")}')
        print(f'     linear solves {len(its)}, KSP its {its[:40]}{" ..." if len(its) > 40 else ""}')
        print(f'     recovery solves {lg.get("recovery")}, returned to toolkit {lg.get("back_to_toolkit")}')

    sb = {k: v for k, v in state['cases'].items() if v['kind'] == 'sb'}
    if sb:
        md = stressbalance_model()
        md.cluster = cpu_cluster(EXEC_DIR)
        ice = np.asarray(md.mask.ice_levelset).ravel() < 0
        md.miscellaneous.name = REF_SB
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
        v_ref = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().copy()
        print(f'\n=== One SSA stress balance, smoothed start ({state.get("smoothing")}) ===')
        show('M (unsmoothed, MUMPS 48)', parse_outlog(f'{EXEC_DIR}/{REF_SB}/{REF_SB}.outlog'))
        for key, c in sb.items():
            lg = logs[key] = parse_outlog(f'{EXEC_DIR}/{c["name"]}/{c["name"]}.outlog')
            show(key, lg)
            md.miscellaneous.name = c['name']
            try:
                md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
                v = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel()
                dv = np.abs(v - v_ref)[ice]
                rel = dv / np.maximum(v_ref[ice], 1.0)
                lg['vel_vs_M'] = {'max_abs': float(dv.max()), 'rel_p99': float(np.percentile(rel, 99))}
                print(f'     velocity vs M: max|dV| {dv.max():.4g} m/yr, rel p99 {np.percentile(rel, 99):.3g}')
            except Exception as e:
                print(f'     could not load results ({e})')
        del md

    tr = {k: v for k, v in state['cases'].items() if v['kind'] == 'tr'}
    if tr:
        md = transient_model()
        md.cluster = cpu_cluster(EXEC_DIR)
        ice = np.asarray(md.mask.ice_levelset).ravel() < 0
        final = {}
        nv = md.mesh.numberofvertices

        def last_step(field):
            a = np.asarray(field, float).ravel()
            return a[-nv:].copy()   # results may come stacked over output steps

        print('\n=== One-year SSA transient 1995-1996 (10 steps) ===')
        for key, c in tr.items():
            lg = logs[key] = parse_outlog(f'{EXEC_DIR}/{c["name"]}/{c["name"]}.outlog')
            show(key, lg)
            if lg.get('core_s') is None:
                print('     did not finish: not used in the comparison')
                continue
            md.miscellaneous.name = c['name']
            try:
                md = pyissm.model.execute.solve(md, 'Transient', load_only=True, runtime_name=False)
                ts = md.results.TransientSolution
                last = ts[-1] if isinstance(ts, list) else ts
                final[c['solver']] = (last_step(last.Vel), last_step(last.Thickness))
            except Exception as e:
                print(f'     could not load results ({e})')
        if {'mumps', 'hypre'} <= set(final):
            (vm, hm), (vh, hh) = final['mumps'], final['hypre']
            dv = np.abs(vh - vm)[ice]
            rel = dv / np.maximum(vm[ice], 1.0)
            dh = np.abs(hh - hm)[ice]
            cmp = {'vel_max_abs': float(dv.max()), 'vel_rel_p99': float(np.percentile(rel, 99)),
                   'thk_max_abs': float(dh.max()), 'thk_mean_abs': float(dh.mean())}
            state['transient_gpu_vs_mumps'] = cmp
            print(f'  final step, GPU vs MUMPS: velocity max|dV| {cmp["vel_max_abs"]:.4g} m/yr, rel p99 '
                  f'{cmp["vel_rel_p99"]:.3g}; thickness max|dH| {cmp["thk_max_abs"]:.4g} m, '
                  f'mean {cmp["thk_mean_abs"]:.3g} m')
    state['analysis'] = logs
    with open(STATE, 'w') as f:
        json.dump(state, f, indent=1)
    print('DONE')
