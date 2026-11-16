"""Why does HO flow ~10-15% faster than observed (worklog 5a)? Split rheology from friction with
single HO stress-balance solves (no transient) from the relaxed model, AIS3_ho_relaxed.nc.

Cases (same geometry, same everything else):
  A  as is                                   (HO rheology_B, HO friction as used downstream)
  B  SSA's rheology_B copied to all layers   (tests the softer-HO-ice explanation)
  C  chunk-3 inverted FrictionC, unsmoothed  (tests "downstream HO used stale friction"; only if
                                              the friction in use differs from it)
  D  HO rheology_B rescaled per column so its depth-average equals SSA's B, keeping HO's
     vertical profile (B found to be the main driver, worklog 5c; D tests a usable HO fix)
  D2 as D, but on grounded columns match a DEFORMATION-WEIGHTED column B instead of the plain
     mean: B_eff = (sum w B^-n / sum w)^(-1/n), w = (1 - sigma)^(n+1) (shear-flux weighting, so
     the soft basal layers count as much as they do for flow); floating columns keep the plain
     mean (plug flow). D left an excess where HO's basal layers are soft (worklog 5c).
Env CASES=A,B,... limits which cases are submitted (default all); the state json is merged, so
earlier cases stay listed for the analysis. Env OUT_TAG suffixes the analysis plot name.
PHASE (env var) 'submit': friction provenance check, then submit the solves (waitonlock=0).
PHASE 'analyze': load each case's results, compare speeds with observed speed and SSA.
New execution names AIS3_ho_sbtest_*; no existing file is modified.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import gc
import json
import numpy as np
import xarray as xr
import pyissm

PHASE = os.environ.get('PHASE', 'submit')
CASES = [c for c in os.environ.get('CASES', '').split(',') if c]
OUT_TAG = os.environ.get('OUT_TAG', '')
MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
STATE = f'{MODEL_DIR}/ho_stressbalance_tests_state.json'
BASIN_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/'
              'imbie2/basin_numbers_ismip8km_v2.nc')
_base = pyissm.model.classes.cluster.gadi()


def hugemem_cluster():
    c = pyissm.model.classes.cluster.gadi()
    c.codepath = os.environ['ISSM_DIR'] + '/bin'
    c.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
    c.storage = 'gdata/au88+gdata/vk83+gdata/av17'
    c.moduleuse = ['/g/data/vk83/modules/']
    c.moduleload = ['access-issm_ad/2026.05.0']
    c.login = 'jh7060'
    c.project = 'au88'
    c.queue = 'hugemem'
    c.np = 96
    c.memory = 1450 * 2
    c.time = 60 * 12
    return c


def base_model():
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_relaxed.nc')
    md.results.TransientSolution = []
    # A stress-balance-only solve with ismip6 basalforcings crashed deterministically in all three
    # cases (SEGV on ~40 of 96 ranks right after the horizontal velocity converged, in the ISMIP6
    # floating-melt / vertical-velocity step; worklog 5c). ismip6 had only ever run inside
    # Transient solves before. Basal melt only sets the basal vertical-velocity condition, not the
    # horizontal speed compared here, so use the default class with zero melt.
    md.basalforcings = pyissm.model.classes.basalforcings.default(md.basalforcings)
    md.basalforcings.groundedice_melting_rate = np.zeros(md.mesh.numberofvertices)
    md.basalforcings.floatingice_melting_rate = np.zeros(md.mesh.numberofvertices)
    md.inversion.iscontrol = 0
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.stressbalance.restol = 0.01
    md.stressbalance.maxiter = 300
    md.stressbalance.reltol = 0.1
    md.stressbalance.abstol = np.nan
    md.settings.solver_residue_threshold = 1e-3
    md.cluster = hugemem_cluster()
    return md


def cmp(a, b):
    d = np.asarray(a, float).ravel() - np.asarray(b, float).ravel()
    return f'max|d| {np.abs(d).max():.4g}, mean|d| {np.abs(d).mean():.4g}'


if PHASE == 'submit':
    print('-- Friction provenance...', flush=True)
    bk = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv_chunk3_backup_presmoothing.nc')
    F_c2 = np.asarray(bk.friction.C, float).ravel().copy()                       # chunk-3 start = chunk-2 output
    F_c3 = np.asarray(bk.results.StressbalanceSolution.FrictionC, float).ravel().copy()  # chunk-3 output
    del bk
    cur = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
    F_c3s = np.asarray(cur.results.StressbalanceSolution.FrictionC, float).ravel().copy()  # smoothed chunk 3
    del cur
    gc.collect()

    md = base_model()
    F_use = np.asarray(md.friction.C, float).ravel().copy()
    assert F_use.size == F_c2.size == F_c3.size == F_c3s.size, 'friction fields on different meshes'
    print(f'   relaxed-model friction.C vs chunk-2 output (chunk-3 start): {cmp(F_use, F_c2)}')
    print(f'   relaxed-model friction.C vs chunk-3 output (unsmoothed):    {cmp(F_use, F_c3)}')
    print(f'   relaxed-model friction.C vs chunk-3 output (smoothed):      {cmp(F_use, F_c3s)}')
    print(f'   chunk-3 vs chunk-2 output:                                  {cmp(F_c3, F_c2)}')
    on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
    ol = np.asarray(md.mask.ocean_levelset).ravel()
    g = on_base & (ol > 0)
    r = F_c3[g] / np.maximum(F_use[g], 1e-12)
    print(f'   grounded base vertices: median C_chunk3/C_in_use {np.median(r):.3f}, p10 {np.percentile(r, 10):.3f}, '
          f'p90 {np.percentile(r, 90):.3f}')
    print(f'   rheology_n HO: {np.unique(np.asarray(md.materials.rheology_n).ravel())[:5]}')

    print('-- SSA rheology_B...', flush=True)
    mds = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
    B_ssa = np.asarray(mds.materials.rheology_B, float).ravel().copy()
    x2 = np.asarray(mds.mesh.x).ravel().copy()
    print(f'   rheology_n SSA: {np.unique(np.asarray(mds.materials.rheology_n).ravel())[:5]}')
    del mds
    gc.collect()
    nv3 = md.mesh.numberofvertices
    nvb = int(md.mesh.numberofvertices2d)
    assert np.allclose(np.asarray(md.mesh.x).ravel()[:nvb], x2), 'HO base layer != SSA mesh'
    B_ho = np.asarray(md.materials.rheology_B, float).ravel().copy()

    # Case D: scale each column of HO's B so its depth-average (trapezoid in sigma, from the synced
    # mesh.z) equals SSA's B at that column, keeping HO's vertical profile.
    nl = nv3 // nvb
    z = np.asarray(md.mesh.z, float).ravel().reshape(nl, nvb)
    span = z[-1] - z[0]
    sig = (z - z[0]) / np.where(np.abs(span) > 1e-9, span, 1.0)[None, :]
    w = np.diff(sig, axis=0)
    Bh = B_ho.reshape(nl, nvb)
    B_davg = (0.5 * (Bh[1:] + Bh[:-1]) * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)
    scale = B_ssa / B_davg
    B_D = (Bh * scale[None, :]).ravel()
    gcol = np.asarray(md.mask.ocean_levelset).ravel()[:nvb] > 0
    n_glen = float(np.unique(np.asarray(md.materials.rheology_n).ravel())[0])
    wdef = (1.0 - sig) ** (n_glen + 1.0)
    wmid = 0.5 * (wdef[1:] + wdef[:-1]) * w
    A_mid = 0.5 * (Bh[1:] ** -n_glen + Bh[:-1] ** -n_glen)
    B_eff = ((wmid * A_mid).sum(axis=0) / np.maximum(wmid.sum(axis=0), 1e-30)) ** (-1.0 / n_glen)
    scale2 = np.where(gcol, B_ssa / B_eff, scale)
    B_D2 = (Bh * scale2[None, :]).ravel()
    print(f'   case D2 grounded scale (SSA B / HO deformation-weighted B): median {np.median(scale2[gcol]):.3f}, '
          f'p10 {np.percentile(scale2[gcol], 10):.3f}, p90 {np.percentile(scale2[gcol], 90):.3f}; '
          f'HO B_eff / plain mean, grounded median {np.median((B_eff / B_davg)[gcol]):.3f}')
    print(f'   case D column scale (SSA B / HO depth-avg B), grounded columns: median {np.median(scale[gcol]):.3f}, '
          f'p10 {np.percentile(scale[gcol], 10):.3f}, p90 {np.percentile(scale[gcol], 90):.3f}; '
          f'floating median {np.median(scale[~gcol]):.3f}')

    cases = {'A': 'AIS3_ho_sbtest_A_asis', 'B': 'AIS3_ho_sbtest_B_ssaB'}
    same_c3 = np.allclose(F_use, F_c3)
    if not same_c3:
        cases['C'] = 'AIS3_ho_sbtest_C_chunk3C'
    cases['D'] = 'AIS3_ho_sbtest_D_rescaledB'
    cases['D2'] = 'AIS3_ho_sbtest_D2_defweightedB'
    if CASES:
        cases = {k: v for k, v in cases.items() if k in CASES}
    for key, name in cases.items():
        md.materials.rheology_B = (np.tile(B_ssa, nv3 // nvb) if key == 'B'
                                   else B_D.copy() if key == 'D'
                                   else B_D2.copy() if key == 'D2' else B_ho.copy())
        md.friction.C = F_c3.copy() if key == 'C' else F_use.copy()
        md.miscellaneous.name = name
        md.settings.waitonlock = 0
        print(f'-- Submitting case {key} ({name})...', flush=True)
        pyissm.model.execute.solve(md, 'Stressbalance', load_only=False, runtime_name=False)
    all_cases = {}
    if os.path.exists(STATE):
        with open(STATE) as f:
            all_cases = json.load(f)['cases']
    all_cases.update(cases)
    with open(STATE, 'w') as f:
        json.dump({'cases': all_cases, 'friction_in_use_equals_chunk3': bool(same_c3)}, f, indent=1)
    print(f'DONE submitting {list(cases)}; state in {STATE}. Resubmit with PHASE=analyze once they finish.')

elif PHASE == 'analyze':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import NearestNDInterpolator
    with open(STATE) as f:
        cases = json.load(f)['cases']
    md = base_model()
    nv3 = md.mesh.numberofvertices
    nvb = int(md.mesh.numberofvertices2d)
    nl = nv3 // nvb
    il = np.asarray(md.mask.ice_levelset).ravel()[:nvb]
    ol = np.asarray(md.mask.ocean_levelset).ravel()[:nvb]
    speeds = {}
    for key, name in cases.items():
        md.miscellaneous.name = name
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
        v = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().reshape(nl, nvb)
        speeds[key] = dict(surf=v[-1].copy(), base=v[0].copy(), davg=v.mean(axis=0))
        print(f'   loaded case {key}', flush=True)
    del md
    gc.collect()
    mds = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
    x = np.asarray(mds.mesh.x).ravel(); y = np.asarray(mds.mesh.y).ravel()
    el = np.asarray(mds.mesh.elements).astype(int) - 1
    vel_obs = np.asarray(mds.inversion.vel_obs, float).ravel()
    ts = mds.results.TransientSolution
    vel_ssa = (np.asarray(ts[0].Vel).ravel() if isinstance(ts, list)
               else np.asarray(ts.Vel).ravel()[:mds.mesh.numberofvertices])
    del mds
    gc.collect()
    tx, ty = x[el], y[el]
    ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
    area = np.zeros(x.size)
    for c in range(3):
        np.add.at(area, el[:, c], ea / 3.0)
    ds_b = xr.open_dataset(BASIN_FILE)
    bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
    lk = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), ds_b['basinNumber'].values.ravel())
    bel = lk(np.column_stack([tx.mean(1), ty.mean(1)])).astype(float)
    acc = np.zeros(x.size); cnt = np.zeros(x.size)
    for c in range(3):
        np.add.at(acc, el[:, c], bel)
        np.add.at(cnt, el[:, c], 1)
    basin = np.round(acc / np.maximum(cnt, 1)).astype(int)
    gr = (il < 0) & (ol > 0) & np.isfinite(vel_obs) & (vel_obs > 0)

    def am(v, m):
        return (v[m] * area[m]).sum() / area[m].sum()

    def mr(a, b, m):
        ok = m & (b > 1.0)
        return np.median(a[ok] / b[ok])

    keys = list(cases)
    hdr = '   '.join(f'{k}surf {k}/obs' for k in keys)
    print(f'\n=== Grounded ice (relaxed HO geometry). Area-weighted mean speed (m/yr) and median ratio to observed ===')
    print(f'{"":>5} {"obs":>6} {"SSA":>6} | ' + ' | '.join(f'{k}:surf {k}:davg {k}/obs' for k in keys))
    for label, m in [('all', gr)] + [(f'b{b}', gr & (basin == b)) for b in range(16)]:
        if m.sum() < 50:
            continue
        parts = [f'{am(speeds[k]["surf"], m):6.1f} {am(speeds[k]["davg"], m):6.1f} {mr(speeds[k]["surf"], vel_obs, m):5.2f}'
                 for k in keys]
        print(f'{label:>5} {am(vel_obs, m):6.1f} {am(vel_ssa, m):6.1f} | ' + ' | '.join(parts))
    for k in keys[1:]:
        print(f'   median surf-speed ratio {k}/A over grounded: {mr(speeds[k]["surf"], speeds["A"]["surf"], gr):.3f}')
    print(f'   median SSA/obs over grounded: {mr(vel_ssa, vel_obs, gr):.3f}')

    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 6.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        ratio = np.log10(np.maximum(speeds[k]['surf'], 1e-3) / np.maximum(vel_obs, 1e-3))
        sc = ax.scatter(x[gr] / 1e3, y[gr] / 1e3, c=ratio[gr], s=0.3, cmap='PuOr_r', vmin=-0.5, vmax=0.5,
                        rasterized=True)
        ax.set_title(f'case {k} ({cases[k]}): log10(HO surface speed / observed)', fontsize=9)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/ho_stressbalance_tests{OUT_TAG}.png', dpi=130)
    print(f'Saved models/ho_stressbalance_tests{OUT_TAG}.png')
    print('DONE')
