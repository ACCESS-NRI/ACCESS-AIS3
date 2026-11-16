"""J3 (warming-sensitivity) term for the SSA per-basin-refit melt candidates.

ssa_melt_deltaT_refit_recalibrate.py picked gamma_0=300 on J1+J2 alone; J3 was never computed
for the refit deltaT_basin (worklog 4c/4k). J3 is the only term that tests how melt responds to
ocean warming, and with deltaT refit to present-day melt the response scales like
sqrt(gamma_0), so gamma_0=300 is ~6x less sensitive than the published prior. This script
computes J3 for all 9 refit candidates against the ISMIP7 term-3 targets
(melt_{cold,warm}_target_term3_v2.nc: basin-mean melt, kg/m2/a, 7 ocean models x 16 basins).

Melt is evaluated in Python with ISSM's own ismip6 local formula
(ISSM src/c/classes/Elements/Element.cpp Ismip6FloatingiceMeltingRate and
src/c/modules/FloatingiceMeltingRatex/FloatingiceMeltingRatex.cpp):
    melt = gamma_0 * (1028*cp/(rho_ice*Lf))^2 * max(TF(draft) + deltaT_basin, 0)^2
on all-floating base elements (0 elsewhere), TF linearly interpolated in depth and clamped at
the end levels, P1DG element values averaged to vertices. It is first checked against three
existing ISSM refit solves (Zhou TF); the J3 numbers are only printed if that check passes.
Submitting 126 cluster solves (9 candidates x 7 models x 2 states) is not needed.

Run: qsub launch_ssa_melt_j3_refit.pbs
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import json
import numpy as np
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
STATE_FILE = f'{MODEL_DIR}/deltaT_refit_state_ssa.json'
OUT_JSON = f'{MODEL_DIR}/ssa_melt_j3_refit.json'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
OMD = f'{ISMIP7_OCEAN}/ocean_modelling_data'
ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                 'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
FLOATMASK_FILE = f'{ISMIP7_OCEAN}/floatingmasks/floatingmask_ismip8km.nc'
VERIFY_CANDIDATES = [1, 4, 8]

# J1/J2 of the refit candidates, from ssa_melt_deltaT_refit_recalibrate.py (job 179305728).
J1_REFIT = [6.3809, 0.8455, 1.2151, 0.8968, 1.4311, 1.1730, 1.1004, 0.6706, 1.3191]
J2_REFIT = [115.443, 12.624, 45.529, 102.877, 151.365, 209.268, 240.196, 268.723, 345.259]

# Term-3 target rows -> (TF file, own-melt file) per state. The two FESOM "naughten_ais" rows are
# matched to ACCESS/MMM below by comparing each file's own basin-mean melt with the target.
OCEAN_MODELS = {
    'mathiot': ('Mathiot_NEMO_{s}_v3_TF.nc', 'Mathiot_NEMO_{s}_v2_m.nc'),
    'timmermann': ('Timmermann_FESOM_{s}_v3_TF.nc', 'Timmermann_FESOM_{s}_v2_m.nc'),
    'fesom_access': ('Naughten_FESOM_ACCESS_{s}_v2_TF.nc', 'Naughten_FESOM_ACCESS_{s}_v2_m.nc'),
    'fesom_mmm': ('Naughten_FESOM_MMM_{s}_v2_TF.nc', 'Naughten_FESOM_MMM_{s}_v2_m.nc'),
    'jourdain_naughten': ('Jourdain-Naughten_NEMO-MITgcm_{s}_TF.nc', 'Jourdain-Naughten_NEMO-MITgcm_{s}_m.nc'),
    'naughten_naughten': ('Naughten_MITamu-MITwed_{s}_TF.nc', 'Naughten_MITamu-MITwed_{s}_m.nc'),
    'haid': ('Haid_FESOM_{s}_TF.nc', 'Haid_FESOM_{s}_m.nc'),
}
STATES = ['cold', 'warm']

cluster = pyissm.model.classes.cluster.gadi()
cluster.codepath = os.environ['ISSM_DIR'] + '/bin'
cluster.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
cluster.storage = 'gdata/au88+gdata/vk83+gdata/av17'
cluster.moduleuse = ['/g/data/vk83/modules/']
cluster.moduleload = ['access-issm_ad/2026.05.0']
cluster.np = 48
cluster.memory = 190
cluster.time = 60 * 48
cluster.login = 'jh7060'
cluster.project = 'au88'

print('-- Loading AIS3_melt_final_ssa.nc (same geometry and basin_id as the refit runs)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_melt_final_ssa.nc')
md.basalforcings.tf = [np.asarray(t, dtype=float) for t in md.basalforcings.tf]  # round-trip fix (4d)
md.results.TransientSolution = []
nv = md.mesh.numberofvertices
mesh_x = np.asarray(md.mesh.x, dtype=float).ravel()
mesh_y = np.asarray(md.mesh.y, dtype=float).ravel()
elx = np.asarray(md.mesh.elements).astype(int) - 1
base = np.asarray(md.geometry.base, dtype=float).ravel()
ol = np.asarray(md.mask.ocean_levelset, dtype=float).ravel()
il = np.asarray(md.mask.ice_levelset, dtype=float).ravel()
basin_el = np.asarray(md.basalforcings.basin_id, dtype=float).ravel().astype(int) - 1  # 0-based
num_basins = int(md.basalforcings.num_basins)
rho_ice = float(md.materials.rho_ice)
coef = ((1028.0 * float(md.materials.mixed_layer_capacity)) / (rho_ice * float(md.materials.latentheat))) ** 2
print(f'   nv={nv}, elements={elx.shape[0]}, basins={num_basins}, melt coefficient={coef:.6e}')

# Element masks exactly as ISSM tests them: IsIceInElement (any vertex ice_levelset<0) and
# IsAllFloating (all vertices ocean_levelset<0).
el_float = (ol[elx] < 0).all(axis=1) & (il[elx] < 0).any(axis=1)
n_conn = np.zeros(nv)
for c in range(3):
    np.add.at(n_conn, elx[:, c], 1.0)

tri_x, tri_y = mesh_x[elx], mesh_y[elx]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
vertex_area = np.zeros(nv)
for c in range(3):
    np.add.at(vertex_area, elx[:, c], elem_area / 3.0)
# Vertex basin: majority of attached elements, same rounding as the refit scripts.
basin_vtx = np.zeros(nv)
for c in range(3):
    np.add.at(basin_vtx, elx[:, c], basin_el)
basin_vtx = np.round(basin_vtx / np.maximum(n_conn, 1)).astype(int)
floating = ol < 0


def tf_on_mesh(path):
    """(n_depth, nv) TF on the mesh, NaN where the source has no data; and positive depths."""
    ds = xr.open_dataset(path)
    gx, gy, gz = ds['x'].values.astype(float), ds['y'].values.astype(float), ds['z'].values
    tf = ds['tf'].values
    out = np.empty((gz.size, nv))
    pts = np.column_stack([mesh_y, mesh_x])
    for k in range(gz.size):
        f = RegularGridInterpolator((gy, gx), tf[k], method='linear', bounds_error=False, fill_value=np.nan)
        out[k] = f(pts)
    return out, -np.asarray(gz, dtype=float)


def tf_at_draft(tf_mesh, depths):
    """ISSM's depth interpolation: linear between levels, clamped to the first/last level."""
    d = -base
    k = np.clip(np.searchsorted(depths, d, side='right') - 1, 0, depths.size - 2)
    a2 = np.clip((d - depths[k]) / (depths[k + 1] - depths[k]), 0.0, 1.0)
    idx = np.arange(nv)
    return (1.0 - a2) * tf_mesh[k, idx] + a2 * tf_mesh[k + 1, idx]


def melt_vertices(tf_draft, gamma_0, delta_t):
    """Per-element P1DG melt, averaged to vertices over all attached elements (0 on non-floating)."""
    tf_el = tf_draft[elx]
    m_el = gamma_0 * coef * np.maximum(tf_el + delta_t[basin_el][:, None], 0.0) ** 2
    m_el[~el_float] = 0.0
    acc = np.zeros(nv)
    for c in range(3):
        np.add.at(acc, elx[:, c], m_el[:, c])
    return acc / np.maximum(n_conn, 1)


def basin_totals_gt(melt_myr):
    flux = melt_myr * rho_ice * vertex_area / 1e12
    return np.array([np.nansum(flux[(basin_vtx == b) & floating]) for b in range(num_basins)])


def basin_mean_kgm2(melt_myr, valid):
    out = np.full(num_basins, np.nan)
    for b in range(num_basins):
        m = (basin_vtx == b) & floating & valid
        if vertex_area[m].sum() > 0:
            out[b] = np.sum(melt_myr[m] * rho_ice * vertex_area[m]) / vertex_area[m].sum()
    return out


with open(STATE_FILE) as f:
    state = json.load(f)
cands = []
for i_str, cand in state['candidates'].items():
    i = int(i_str)
    final_round = len(cand['history']) - 1
    cands.append({'i': i, 'gamma_0': float(cand['gamma_0']), 'final_round': final_round,
                  'converged': bool(cand['converged']),
                  'delta_t': np.asarray(cand['history'][-1]['delta_t'], dtype=float)})

# ---------------------------------------------------------------------------------------------
print('\n-- Check: Python formula vs existing ISSM refit solves (Zhou TF)...', flush=True)
zhou_mesh, zhou_depths = tf_on_mesh(ZHOU_TF_FILE)
zhou_draft = tf_at_draft(np.nan_to_num(zhou_mesh, nan=0.0), zhou_depths)  # ISSM input had NaN->0
verify_ok = True
for c in (cands[i] for i in VERIFY_CANDIDATES):
    name = f"AIS3_ssa_deltaT_refit_g{c['i']}_r{c['final_round']}"
    md.miscellaneous.name = name
    md.cluster = cluster
    md.results.TransientSolution = []
    md = pyissm.model.execute.solve(md, 'Transient', load_only=True, runtime_name=False, check_consistency=False)
    issm = np.asarray(md.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate, dtype=float).ravel()
    md.results.TransientSolution = []
    py = melt_vertices(zhou_draft, c['gamma_0'], c['delta_t'])
    diff = np.abs(py - issm)
    bt_i, bt_p = basin_totals_gt(issm), basin_totals_gt(py)
    rel_tot = abs(bt_p.sum() - bt_i.sum()) / bt_i.sum()
    rel_basin = np.max(np.abs(bt_p - bt_i) / np.maximum(np.abs(bt_i), 1.0))
    print(f"   {name}: max|diff| {diff.max():.4e} m/yr (99.9th pct {np.percentile(diff, 99.9):.4e}), "
          f"max melt {issm.max():.3f}; total {bt_i.sum():.2f} (ISSM) vs {bt_p.sum():.2f} (Python) Gt/yr, "
          f"worst basin rel diff {rel_basin:.2e}")
    if rel_tot > 1e-3 or rel_basin > 1e-2:
        verify_ok = False
if not verify_ok:
    raise SystemExit('Python melt does not reproduce ISSM within tolerance -- J3 not computed.')
print('   PASS: Python melt reproduces ISSM (total within 0.1%, every basin within 1%).')

# ---------------------------------------------------------------------------------------------
print('\n-- Target rows vs each ocean model\'s own melt (basin means on the 8 km grid)...', flush=True)
ds_b = xr.open_dataset(BASIN_FILE)
basin_grid = ds_b['basinNumber'].values
fm = xr.open_dataset(FLOATMASK_FILE)['mask'].values
targets = {s: xr.open_dataset(f'{OMD}/melt_{s}_target_term3_v2.nc') for s in STATES}
target_rows = list(targets['cold']['model'].values)


def own_basin_means(path):
    ds = xr.open_dataset(path)
    v = 'basal_melt' if 'basal_melt' in ds else 'melt_rate'
    m = ds[v].values
    out = np.full(num_basins, np.nan)
    for b in range(num_basins):
        sel = (basin_grid == b) & (fm > 0) & np.isfinite(m)
        if sel.sum() > 0:
            out[b] = np.nanmean(m[sel])
    return out


own = {k: own_basin_means(f"{OMD}/{files[1].format(s='warm')}") for k, files in OCEAN_MODELS.items()}
row_to_model = {}
for r in target_rows:
    t = targets['warm']['melt_rate'].sel(model=r).values
    best, best_err = None, np.inf
    for k, o in own.items():
        ok = np.isfinite(t) & np.isfinite(o)
        if ok.sum() == 0 or np.isfinite(t).sum() != ok.sum():
            continue
        err = np.median(np.abs(o[ok] - t[ok]) / np.maximum(np.abs(t[ok]), 1.0))
        if err < best_err:
            best, best_err = k, err
    row_to_model[r] = best
    print(f"   target '{r}': best match {best} (median rel. diff {best_err:.3f}, "
          f"{int(np.isfinite(t).sum())} basins)")
DEFAULT_ROWS = {'mathiot': 'mathiot', 'timmermann': 'timmermann', 'naughten_ais_1': 'fesom_access',
                'naughten_ais_2': 'fesom_mmm', 'jourdain_naughten': 'jourdain_naughten',
                'naughten_naughten': 'naughten_naughten', 'haid': 'haid'}
if len(set(row_to_model.values())) != len(target_rows) or None in row_to_model.values():
    print(f'   WARNING: matching not one-to-one ({row_to_model}); using name order {DEFAULT_ROWS}')
    row_to_model = DEFAULT_ROWS

# ---------------------------------------------------------------------------------------------
print('\n-- Parameterised melt under each ocean model\'s cold/warm TF, per candidate...', flush=True)
drafts = {}
for r, k in row_to_model.items():
    for s in STATES:
        tfm, dep = tf_on_mesh(f"{OMD}/{OCEAN_MODELS[k][0].format(s=s)}")
        drafts[(r, s)] = tf_at_draft(tfm, dep)  # NaN where the ocean model has no data

results = []
for c in cands:
    param = {}
    for r in target_rows:
        for s in STATES:
            td = drafts[(r, s)]
            valid = np.isfinite(td)
            param[(r, s)] = basin_mean_kgm2(melt_vertices(np.nan_to_num(td, nan=0.0), c['gamma_0'], c['delta_t']), valid)
    abs_err, sens_err, ratios = [], [], []
    for r in target_rows:
        tc = targets['cold']['melt_rate'].sel(model=r).values
        tw = targets['warm']['melt_rate'].sel(model=r).values
        pc, pw = param[(r, 'cold')], param[(r, 'warm')]
        for b in range(num_basins):
            for t, p in ((tc[b], pc[b]), (tw[b], pw[b])):
                if np.isfinite(t) and np.isfinite(p):
                    abs_err.append(abs(p - t))
            if np.all(np.isfinite([tc[b], tw[b], pc[b], pw[b]])):
                sens_err.append(abs((pw[b] - pc[b]) - (tw[b] - tc[b])))
                if abs(tw[b] - tc[b]) > 1.0:
                    ratios.append((pw[b] - pc[b]) / (tw[b] - tc[b]))
    c['J3'] = float(np.mean(abs_err))
    c['J3_sens'] = float(np.mean(sens_err))
    c['sens_ratio_median'] = float(np.median(ratios))
    c['n_entries'] = len(abs_err)
    c['param_basin_means'] = {f'{r}_{s}': param[(r, s)].tolist() for (r, s) in param}
    results.append(c)

print(f"\n{'i':>2} {'gamma_0':>9} {'conv':>5} {'J1':>8} {'J2':>9} {'J3':>9} {'J3_sens':>9} "
      f"{'dMelt ratio':>11}  (J3 in kg/m2/a; ratio = param warm-cold / target warm-cold, median)")
for c in results:
    print(f"{c['i']:2d} {c['gamma_0']:9.1f} {str(c['converged']):>5} {J1_REFIT[c['i']]:8.3f} "
          f"{J2_REFIT[c['i']]:9.3f} {c['J3']:9.1f} {c['J3_sens']:9.1f} {c['sens_ratio_median']:11.3f}")

# ---------------------------------------------------------------------------------------------
print('\n-- Monte Carlo over objective weights (10000 samples), ISMIP7-style...', flush=True)
use = [c for c in results if c['converged']]
g0 = np.array([c['gamma_0'] for c in use])
J = np.array([[J1_REFIT[c['i']] for c in use], [J2_REFIT[c['i']] for c in use], [c['J3'] for c in use]])
Jn = J / np.median(J, axis=1, keepdims=True)
rng = np.random.default_rng(0)
summary = {}
for label, rows in (('J1+J2+J3', [0, 1, 2]), ('J2+J3', [1, 2]), ('J1+J2', [0, 1])):
    win = np.empty(10000)
    for n in range(10000):
        a = rng.uniform(0, 1, len(rows))
        a /= a.sum()
        win[n] = g0[np.argmin((a[:, None] * Jn[rows]).sum(axis=0))]
    p = np.percentile(win, [5, 50, 95])
    summary[label] = p.tolist()
    vals, counts = np.unique(win, return_counts=True)
    print(f"   {label:9s}: 5th/50th/95th gamma_0 = {p[0]:.1f} / {p[1]:.1f} / {p[2]:.1f}   "
          f"win share: " + ', '.join(f'{v:.0f}:{100 * n / win.size:.1f}%' for v, n in zip(vals, counts)))

with open(OUT_JSON, 'w') as f:
    json.dump({'candidates': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
                              for c in results],
               'row_to_model': row_to_model, 'mc_percentiles': summary}, f, indent=1)
print(f'\nSaved {OUT_JSON}')
print('DONE')
