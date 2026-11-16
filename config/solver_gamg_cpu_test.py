"""Can an iterative AMG solver replace MUMPS on the continental SSA model? Step 1 of moving AIS3
onto the GPU solver work (ISSM/test/GPU_CHANGES.md): the GPU benchmarks were all on the idealised
SquareShelf, so first check that GMRES + GAMG converges on the real AIS3 SSA matrix (grounding
lines, Budd friction, shelf/interior viscosity contrast), on CPU with the production ISSM build
(PETSc 3.21 has GAMG built in; it has no Hypre or CUDA).

Single SSA stress-balance solves from AIS3_SSA_inverted.nc, everything identical except the
stress-balance toolkit:
  M  MUMPS LU (production default)
  G  GMRES(100) + smoothed-aggregation GAMG, mat_block_size=2, ksp_rtol=1e-12
     (the CPU-GAMG settings of the GPU benchmark, ISSM/test/gpu_hypre_test.py)
Only md.toolkits.StressbalanceAnalysis is changed, so RecoveryAnalysis stays MUMPS: if GAMG misses
md.settings.solver_residue_threshold, ISSM silently re-solves with MUMPS. The analysis flags this
from the outlog, since such a case would look correct but its timing would be MUMPS's.

Env CASES=M+G+S1... (or comma-separated outside qsub -v) limits which cases are submitted (default all); the state json is merged.
PHASE (env var) 'submit': submit the solves (waitonlock=0). Refuses if a case's execution dir exists.
PHASE 'analyze': load each case, compare velocity with M, report timings, KSP iterations, recovery.
PHASE 'coords' (after a dump case; env DUMPCASE=D or D2): map matrix DOF pairs to vertices, write
coords.bin for rigid-body modes.
Writes only to execution_gpu_test/ (incl. matrix_dump_ssa_picard1/) and models/solver_gamg_cpu_test_state.json.
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

PHASE = os.environ.get('PHASE', 'submit')
CASES = [c for c in re.split(r'[,+]', os.environ.get('CASES', '')) if c]   # '+' also works: qsub -v splits on ','
MODEL_DIR = f'{site.ROOT}/models'
EXEC_DIR = f'{site.ROOT}/execution_gpu_test'
STATE = f'{MODEL_DIR}/solver_gamg_cpu_test_state.json'

# Values are strings: pyissm's write_toolkits_file writes a falsy value (0, '') as a bare flag.
GAMG = {
    'toolkit': 'petsc',
    'mat_type': 'mpiaij',
    'ksp_type': 'gmres',
    'pc_type': 'gamg',
    'mat_block_size': '2',          # SSA: 2 DOFs/node, aggregate by node (model has no spcs to split blocks)
    'pc_gamg_agg_nsmooths': '1',
    'pc_gamg_threshold': '0.08',
    'pc_gamg_coarse_eq_limit': '2000',
    'ksp_rtol': '1e-12',
    'ksp_max_it': '5000',
    'ksp_gmres_restart': '100',
    'ksp_converged_reason': '',     # bare flag: log why each linear solve stopped
}
# G broke down on the first Picard solve (DIVERGED_BREAKDOWN at 200 its, residue 1e-2), so ISSM
# switched to RecoveryAnalysis and stayed on MUMPS for the rest. Breakdown with an SPD Picard matrix
# points at the preconditioner. Variants, each changing one suspect (S1-S3) and then all of them (S4):
#   S1 Chebyshev smoother's eigenvalue estimates under the shelf/interior coefficient jumps -> SOR
#   S2 the 0.08 drop threshold discarding weak-but-real couplings -> keep all (0.0)
#   S3 rows scaled over many orders of magnitude -> symmetric diagonal scaling
SOR = {'mg_levels_ksp_type': 'richardson', 'mg_levels_pc_type': 'sor'}
THR0 = {'pc_gamg_threshold': '0.0'}
DSCALE = {'ksp_diagonal_scale': '', 'ksp_diagonal_scale_fix': ''}
VIEW = {'ksp_view': ''}             # print the hierarchy after setup (levels, complexity)
# None of S1-S4 converged either. D dumps the first Picard system (MUMPS, maxiter=1) as PETSc binary
# files, so preconditioners can be tried on it directly, outside ISSM (ISSM/test/matrix_lab/).
# That system turned out to be pathological because of the initial guess: md.initialization is
# vel_obs, whose piecewise-constant gap fills give ~zero strain rate and huge viscosity on ~9.5k
# vertices (diagonal spread 16.8 orders). D2 runs all Picard iterations; PETSc's viewer rewrites the
# files on every solve, so D2 keeps the last (converged-viscosity) system, the one a transient sees.
DUMP_DIRS = {'D': f'{EXEC_DIR}/matrix_dump_ssa_picard1', 'D2': f'{EXEC_DIR}/matrix_dump_ssa_picardlast'}


def dump_opts(d):
    return {'ksp_view_mat': f'binary:{d}/A.bin',
            'ksp_view_rhs': f'binary:{d}/b.bin',
            'ksp_view_solution': f'binary:{d}/x.bin'}


ALL_CASES = {'M': ('AIS3_solvertest_M_mumps', None),
             'D': ('AIS3_solvertest_D_dump', 'dump'),
             'D2': ('AIS3_solvertest_D2_dumplast', 'dump'),
             'G': ('AIS3_solvertest_G_gamg', GAMG),
             'S1': ('AIS3_solvertest_S1_sor', {**GAMG, **SOR, **VIEW}),
             'S2': ('AIS3_solvertest_S2_thr0', {**GAMG, **THR0, **VIEW}),
             'S3': ('AIS3_solvertest_S3_dscale', {**GAMG, **DSCALE, **VIEW}),
             'S4': ('AIS3_solvertest_S4_all', {**GAMG, **SOR, **THR0, **DSCALE, **VIEW})}


def normal_cluster():
    c = pyissm.model.classes.cluster.gadi()
    c.codepath = os.environ['ISSM_DIR'] + '/bin'
    c.executionpath = EXEC_DIR
    c.storage = site.STORAGE
    c.moduleuse = ['/g/data/vk83/modules/']
    c.moduleload = ['access-issm_ad/2026.05.0']
    c.login = site.USER
    c.project = site.PROJECT
    c.np = 48
    c.memory = 190
    c.time = 60 * 4
    return c


def base_model():
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_inverted.nc')
    md.inversion.iscontrol = 0
    md.verbose.solution = 1
    md.verbose.convergence = 1
    md.verbose.solver = 1           # "Petsc: N KSP iterations" per linear solve
    md.settings.solver_residue_threshold = 1e-3   # as in the SSA track
    md.cluster = normal_cluster()
    return md


def parse_outlog(path):
    if not os.path.exists(path):
        return {'outlog': 'missing'}
    txt = open(path, errors='ignore').read()

    def num(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    reasons = re.findall(r'Linear solve (?:converged|did not converge) due to (\w+) iterations (\d+)', txt)
    return {
        'core_s': num(r'Total Core solution elapsed time:\s*([\d.eE+-]+)'),
        'solver_s': num(r'Linear solver elapsed time:\s*([\d.eE+-]+)'),
        'init_s': num(r'FemModel initialization elapsed time:\s*([\d.eE+-]+)'),
        'su': num(r'Service Units:\s*([\d.]+)'),
        'ksp_its': [int(n) for n in re.findall(r'Petsc: (\d+) KSP iterations', txt)],
        'ksp_reasons': sorted({r for r, _ in reasons}),
        'recovery': len(re.findall(r'Trying Recovery Mode', txt)),
        'residue_too_high': re.findall(r'solver residue too high!: norm\(KU-F\)/norm\(F\)=(\S+)', txt)[:5],
        'exit_status': num(r'Exit Status:\s*(-?\d+)'),
    }


if PHASE == 'submit':
    cases = {k: v for k, v in ALL_CASES.items() if not CASES or k in CASES}
    for key, (name, _) in cases.items():
        if os.path.exists(f'{EXEC_DIR}/{name}'):
            raise SystemExit(f'{EXEC_DIR}/{name} exists; archive it with mv -n before resubmitting case {key}')
    md = base_model()
    sb = md.stressbalance
    print(f'   vertices {md.mesh.numberofvertices}, restol {sb.restol}, reltol {sb.reltol}, abstol {sb.abstol}, '
          f'maxiter {sb.maxiter}, friction {type(md.friction).__name__}', flush=True)
    mumps = dict(md.toolkits.DefaultAnalysis)
    maxiter = md.stressbalance.maxiter
    for key, (name, opts) in cases.items():
        if opts == 'dump':
            os.makedirs(DUMP_DIRS[key])     # fails if an earlier dump is there; archive it with mv -n first
            md.toolkits.StressbalanceAnalysis = {**mumps, **dump_opts(DUMP_DIRS[key])}
            md.stressbalance.maxiter = 1 if key == 'D' else maxiter   # D: the one Picard solve G failed on
        else:
            md.toolkits.StressbalanceAnalysis = dict(opts if opts is not None else mumps)
            md.stressbalance.maxiter = maxiter
        md.miscellaneous.name = name
        md.settings.waitonlock = 0
        print(f'-- Submitting case {key} ({name}): {md.toolkits.StressbalanceAnalysis}', flush=True)
        pyissm.model.execute.solve(md, 'Stressbalance', load_only=False, runtime_name=False)
    all_cases = {}
    if os.path.exists(STATE):
        with open(STATE) as f:
            all_cases = json.load(f)['cases']
    all_cases.update({k: v[0] for k, v in cases.items()})
    with open(STATE, 'w') as f:
        json.dump({'cases': all_cases}, f, indent=1)
    print(f'DONE submitting {list(cases)}; state in {STATE}. Resubmit with PHASE=analyze once they finish.')

elif PHASE == 'analyze':
    with open(STATE) as f:
        cases = json.load(f)['cases']
    md = base_model()
    il = np.asarray(md.mask.ice_levelset).ravel()
    ol = np.asarray(md.mask.ocean_levelset).ravel()
    ice = il < 0
    vel_obs = np.asarray(md.inversion.vel_obs, float).ravel()
    vel, logs = {}, {}
    for key, name in cases.items():
        logs[key] = parse_outlog(f'{EXEC_DIR}/{name}/{name}.outlog')
        md.miscellaneous.name = name
        try:
            md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
            vel[key] = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().copy()
        except Exception as e:
            print(f'   case {key}: could not load results ({e})')

    print('\n=== Timings and linear-solver behaviour (inner job, 48 cores) ===')
    for key, lg in logs.items():
        its = lg.get('ksp_its', [])
        print(f'  {key} ({cases[key]})')
        print(f'     core {lg.get("core_s")} s, linear solver {lg.get("solver_s")} s, init {lg.get("init_s")} s, '
              f'SU {lg.get("su")}, exit {lg.get("exit_status")}')
        print(f'     linear solves {len(its)}, KSP its per solve {its[:12]}{" ..." if len(its) > 12 else ""}'
              f'{f", max {max(its)}" if its else ""}; reasons {lg.get("ksp_reasons")}')
        if lg.get('recovery') or lg.get('residue_too_high'):
            print(f'     !! recovery mode {lg["recovery"]}x, residues {lg["residue_too_high"]}: '
                  f'MUMPS re-solved at least one step; timing is not GAMG alone')

    if 'M' in vel:
        print('\n=== Velocity vs case M (ice vertices) ===')
        gr = ice & (ol > 0) & np.isfinite(vel_obs) & (vel_obs > 0)
        for key, v in vel.items():
            if key == 'M':
                continue
            d = np.abs(v - vel['M'])[ice]
            rel = d / np.maximum(vel['M'][ice], 1.0)
            print(f'  {key}: max|dV| {d.max():.4g} m/yr, mean|dV| {d.mean():.4g}, rel p50 {np.median(rel):.3g}, '
                  f'p99 {np.percentile(rel, 99):.3g}, max {rel.max():.3g}')
        for key, v in vel.items():
            print(f'  {key}: max speed {np.nanmax(v[ice]):.1f} m/yr, grounded RMSE vs obs '
                  f'{np.sqrt(np.nanmean((v[gr] - vel_obs[gr]) ** 2)):.2f} m/yr')
    with open(STATE) as f:
        state = json.load(f)
    state['analysis'] = logs
    with open(STATE, 'w') as f:
        json.dump(state, f, indent=1)
    print(f'\nLog summary added to {STATE}')
    print('DONE')

elif PHASE == 'coords':
    # GAMG's rigid-body near-null space needs (x, y) per matrix DOF pair, but ISSM's free-DOF
    # ordering is not exported. Recover it from case D: the dumped solution x.bin holds (vx, vy)
    # per DOF pair in matrix order, and D's results hold the same velocities per vertex.
    from scipy.spatial import cKDTree

    def read_petsc_vec(path):
        with open(path, 'rb') as f:
            cid, n = np.fromfile(f, '>i4', 2)
            assert cid == 1211214, f'{path}: not a PETSc Vec ({cid})'
            return np.fromfile(f, '>f8', n).astype(float)

    def write_petsc_vec(path, v):
        with open(path, 'xb') as f:     # 'x': never overwrite
            np.array([1211214, v.size], '>i4').tofile(f)
            np.asarray(v, '>f8').tofile(f)

    dkey = os.environ.get('DUMPCASE', 'D')
    name, DUMP_DIR = ALL_CASES[dkey][0], DUMP_DIRS[dkey]
    md = base_model()
    md.miscellaneous.name = name
    md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
    vx = np.asarray(md.results.StressbalanceSolution.Vx, float).ravel()
    vy = np.asarray(md.results.StressbalanceSolution.Vy, float).ravel()
    u = read_petsc_vec(f'{DUMP_DIR}/x.bin').reshape(-1, 2)
    print(f'   matrix DOF pairs {u.shape[0]}, vertices {vx.size}', flush=True)
    # results may be in m/yr and the solver vector in m/s: find the factor from the largest speeds
    yts = md.constants.yts
    scale = yts if np.abs(u).max() * yts / max(np.abs(vx).max(), 1e-30) < 10 else 1.0
    tree = cKDTree(np.column_stack([vx, vy]) / scale)
    dist, idx = tree.query(u, k=2)
    vmag = np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-30)
    rel = dist[:, 0] / vmag
    ambiguous = dist[:, 1] <= 10 * dist[:, 0] + 1e-30 * vmag
    unique = np.unique(idx[:, 0]).size
    print(f'   scale {scale:g}; match rel error max {rel.max():.3g}, p99 {np.percentile(rel, 99):.3g}; '
          f'{unique} distinct vertices for {u.shape[0]} pairs; {ambiguous.sum()} ambiguous pairs', flush=True)
    if rel.max() > 1e-8 or unique != u.shape[0]:
        raise SystemExit('DOF -> vertex match is not clean; not writing coordinates')
    v = idx[:, 0]
    xy = np.column_stack([np.asarray(md.mesh.x).ravel()[v], np.asarray(md.mesh.y).ravel()[v]]).ravel()
    np.save(f'{DUMP_DIR}/dof_vertex.npy', v)
    write_petsc_vec(f'{DUMP_DIR}/coords.bin', xy)
    print(f'Wrote {DUMP_DIR}/coords.bin (interleaved x,y per DOF pair) and dof_vertex.npy (0-based vertex per pair)')
    print('DONE')
