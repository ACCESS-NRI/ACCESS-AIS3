"""Hypre BoomerAMG on one A100 for the continental SSA stress balance, inside ISSM. Step 2 of moving
AIS3 onto the GPU solver work, after solver_gamg_cpu_test.py and the matrix lab (ISSM/test/matrix_lab/):
on the converged-viscosity AIS3 SSA system, GPU BoomerAMG took 3.1 s vs 15.7 s for 16-core MUMPS.

Runs the GPU ISSM fork (site.env AIS3_GPU_ISSM_DIR, built against the CUDA + Hypre PETSc by
build_gpu_hypre_gdata.pbs) through pyissm on dgxa100, several MPI ranks sharing one GPU. Same model
and settings as case M of solver_gamg_cpu_test.py (AIS3_SSA_inverted.nc, one SSA stress balance):
  H<np>  BoomerAMG + right-preconditioned GMRES on md.toolkits.StressbalanceAnalysis only;
         RecoveryAnalysis stays MUMPS.
The first Picard solve starts from md.initialization = vel_obs, whose piecewise-constant gap fills
make that matrix near-singular (diagonal over 16.8 orders); BoomerAMG is expected to fail there, ISSM
to re-solve it with MUMPS, and (fork fix in Solverx.cpp) to go back to BoomerAMG afterwards.

First attempt (H1, H4, 2026-10-20): BoomerAMG broke down on Picard 1 as expected, but the MUMPS
recovery solve then reported residue exactly 1 and ISSM stopped. The recovery options carried no
vec_type, so ISSM built the direct solve's solution vector as a host vector next to the CUSPARSE
matrix and CUDA right-hand side. RecoveryAnalysis therefore gets vec_type=cuda (RECOVERY below).

H1_cudarecov then worked (recovery on Picard 1, back to BoomerAMG, velocity within 4e-7 (p99) of
M), but took 724 s core vs 137 s: assembly etc. ~600 s on one rank. Ranks sharing the GPU made the
solve 10-20x slower until CUDA MPS was started (lab: np16 4.2 s with MPS vs 25 s without), so the
queue script starts an MPS daemon when NP > 1.

Env NP (default 16) = MPI ranks on the one GPU; TAG suffixes the case name (rerun without touching
an earlier case's directory); KSP_MAX_IT (default 2000) caps BoomerAMG-GMRES before recovery.
PHASE (env var):
  'submit'  submit case H<NP> (waitonlock=0). Refuses if its execution dir exists.
  'analyze' compare velocity with case M, report timings, KSP iterations and recovery solves.
Writes only to execution_gpu_test/ and models/solver_hypre_gpu_test_state.json. Cluster, options
and log parsing live in gpu_solver_utils.py (shared with solver_gpu_benchmark.py).
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
import numpy as np
import pyissm
from gpu_solver_utils import hypre_options, RECOVERY_EXTRA, gpu_cluster as _gpu_cluster, parse_outlog

PHASE = os.environ.get('PHASE', 'submit')
NP = int(os.environ.get('NP', '16'))
MODEL_DIR = f'{site.ROOT}/models'
EXEC_DIR = f'{site.ROOT}/execution_gpu_test'
STATE = f'{MODEL_DIR}/solver_hypre_gpu_test_state.json'
REF = 'AIS3_solvertest_M_mumps'          # 48-core MUMPS, from solver_gamg_cpu_test.py

HYPRE = hypre_options(os.environ.get('KSP_MAX_IT', '2000'))


def gpu_cluster():
    return _gpu_cluster(EXEC_DIR, NP)


def base_model():
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_inverted.nc')
    md.inversion.iscontrol = 0
    md.verbose.solution = 1
    md.verbose.convergence = 1
    md.verbose.solver = 1           # "Petsc: N KSP iterations" per linear solve
    md.settings.solver_residue_threshold = 1e-3   # as in the SSA track
    return md


TAG = os.environ.get('TAG', '')
name = f'AIS3_solvertest_H_hypre_np{NP}{TAG}'
if PHASE == 'submit':
    if os.path.exists(f'{EXEC_DIR}/{name}'):
        raise SystemExit(f'{EXEC_DIR}/{name} exists; archive it with mv -n before resubmitting')
    md = base_model()
    md.cluster = gpu_cluster()
    md.toolkits.StressbalanceAnalysis = dict(HYPRE)
    md.toolkits.RecoveryAnalysis = {**md.toolkits.RecoveryAnalysis, **RECOVERY_EXTRA}
    md.miscellaneous.name = name
    md.settings.waitonlock = 0
    print(f'-- Submitting {name}: np {NP} on 1 GPU, {md.toolkits.StressbalanceAnalysis}; '
          f'recovery {md.toolkits.RecoveryAnalysis}', flush=True)
    pyissm.model.execute.solve(md, 'Stressbalance', load_only=False, runtime_name=False)
    state = {'cases': {}}
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
    state['cases'][f'H{NP}{TAG}'] = name
    with open(STATE, 'w') as f:
        json.dump(state, f, indent=1)
    print(f'DONE submitting {name}; resubmit with PHASE=analyze once it finishes.')

elif PHASE == 'analyze':
    with open(STATE) as f:
        state = json.load(f)
    md = base_model()
    md.cluster = gpu_cluster()
    ice = np.asarray(md.mask.ice_levelset).ravel() < 0
    md.miscellaneous.name = REF
    md.cluster.executionpath = EXEC_DIR
    md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
    v_ref = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().copy()
    logs = {'M': parse_outlog(f'{EXEC_DIR}/{REF}/{REF}.outlog')}
    for key, nm in state['cases'].items():
        logs[key] = parse_outlog(f'{EXEC_DIR}/{nm}/{nm}.outlog')
        md.miscellaneous.name = nm
        try:
            md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
            v = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel()
            dv = np.abs(v - v_ref)[ice]
            rel = dv / np.maximum(v_ref[ice], 1.0)
            logs[key]['vel_vs_M'] = {'max_abs': float(dv.max()), 'mean_abs': float(dv.mean()),
                                     'rel_p99': float(np.percentile(rel, 99)), 'rel_max': float(rel.max())}
        except Exception as e:
            print(f'   {key}: could not load results ({e})')
    print('\n=== SSA stress balance from AIS3_SSA_inverted.nc ===')
    for key, lg in logs.items():
        its = lg.get('ksp_its', [])
        print(f'  {key}: core {lg.get("core_s")} s, linear solver {lg.get("solver_s")} s, init {lg.get("init_s")} s, '
              f'walltime {lg.get("walltime")}, SU {lg.get("su")}, exit {lg.get("exit_status")}')
        print(f'     Picard its {lg.get("picard_its")}, linear solves {len(its)}, KSP its {its[:25]}')
        print(f'     recovery solves {lg.get("recovery")}, returned to toolkit {lg.get("back_to_toolkit")}; '
              f'reasons {sorted(set(lg.get("ksp_reasons", [])))}')
        if 'vel_vs_M' in lg:
            d = lg['vel_vs_M']
            print(f'     velocity vs M: max|dV| {d["max_abs"]:.4g} m/yr, mean {d["mean_abs"]:.4g}, '
                  f'rel p99 {d["rel_p99"]:.3g}, rel max {d["rel_max"]:.3g}')
    state['analysis'] = logs
    with open(STATE, 'w') as f:
        json.dump(state, f, indent=1)
    print('DONE')
