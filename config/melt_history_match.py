"""History-matching calibration of the ismip6 local melt law on a dense gamma_0 grid (worklog 6.30).

Follows 6.29 (gamma_0=300 does not survive the real ISMIP7 J2 target or J4). Differences from the
ISMIP7 protocol (Reese et al., egusphere-2026-5337), and why:
  - each term's residuals are scaled by the target's own uncertainty plus an explicit structural
    error (delta x |target|), not by the term's median over the candidate ensemble, so the answer
    does not depend on which candidates are tried and no arbitrary U[0,1] term weights are needed;
  - candidates are ruled out (history matching) when a term's RMS standardised residual
    I = sqrt(mean(r^2)) exceeds 3; a pseudo-posterior exp(-sum_terms I^2 / 2) on a log-uniform
    gamma_0 prior gives a central value and range with every term weighted equally;
  - deltaT_b is refit per basin to present-day melt, then optionally capped at +-2 degC (the
    protocol's own limit); with the cap, a binding basin shows up as J1 misfit;
  - J3 uses warm-minus-cold melt changes (the protocol's eq. 6), all 7 ocean models.
Terms (all on the SSA mesh, Python ismip6 local melt as ssa_melt_j3_refit.py, checked against ISSM):
  J1  16 IMBIE2 basin totals vs Paolo/Davison/Adusumilli (Gt/yr)
  J2  10 BFRN-bin totals vs meltobs/melt_target_term2_v3.nc (Gt/yr)
  J3  warm-cold basin-mean change vs melt_{cold,warm}_target_term3_v2.nc (kg/m2/a), sigma from the
      two states' uncertainties in quadrature
  J4  Pine Island / Dotson totals under each observed year's TF vs melt_observations_target_term4.nc
Read-only; writes models/melt_history_match.json (refuses to overwrite).
"""
import os
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import json
import numpy as np
import pandas as pd
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
OUT = f'{M}/melt_history_match.json'
OC = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
OMD = f'{OC}/ocean_modelling_data'
ZHOU_TF = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
           'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
OCEAN_MODELS = {
    'mathiot': 'Mathiot_NEMO_{s}_v3_TF.nc', 'timmermann': 'Timmermann_FESOM_{s}_v3_TF.nc',
    'fesom_access': 'Naughten_FESOM_ACCESS_{s}_v2_TF.nc', 'fesom_mmm': 'Naughten_FESOM_MMM_{s}_v2_TF.nc',
    'jourdain_naughten': 'Jourdain-Naughten_NEMO-MITgcm_{s}_TF.nc',
    'naughten_naughten': 'Naughten_MITamu-MITwed_{s}_TF.nc', 'haid': 'Haid_FESOM_{s}_TF.nc'}
GAMMA = np.logspace(np.log10(300.0), np.log10(100000.0), 40)
DELTAS = (0.1, 0.3, 0.5)     # structural error as a fraction of |target|; 0.3 is the main case
CAP = 2.0
CUT = 3.0
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')

md = pyissm.model.io.load_model(f'{M}/AIS3_melt_final_ssa.nc')
md.results.TransientSolution = []
nv = md.mesh.numberofvertices
mx, my = np.asarray(md.mesh.x, float).ravel(), np.asarray(md.mesh.y, float).ravel()
elx = np.asarray(md.mesh.elements).astype(int) - 1
base = np.asarray(md.geometry.base, float).ravel()
ol = np.asarray(md.mask.ocean_levelset, float).ravel()
il = np.asarray(md.mask.ice_levelset, float).ravel()
basin_el = np.asarray(md.basalforcings.basin_id, float).ravel().astype(int) - 1
nb = int(md.basalforcings.num_basins)
rho_ice = float(md.materials.rho_ice)
coef = ((1028.0 * float(md.materials.mixed_layer_capacity)) / (rho_ice * float(md.materials.latentheat))) ** 2
el_float = (ol[elx] < 0).all(axis=1) & (il[elx] < 0).any(axis=1)
n_conn = np.zeros(nv)
for c in range(3):
    np.add.at(n_conn, elx[:, c], 1.0)
tx, ty = mx[elx], my[elx]
earea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
varea = np.zeros(nv)
for c in range(3):
    np.add.at(varea, elx[:, c], earea / 3.0)
bv = np.zeros(nv)
for c in range(3):
    np.add.at(bv, elx[:, c], basin_el)
basin_v = np.round(bv / np.maximum(n_conn, 1)).astype(int)
floating = ol < 0


def tf_on_mesh(path):
    ds = xr.open_dataset(path)
    gx, gy, gz = ds['x'].values.astype(float), ds['y'].values.astype(float), ds['z'].values
    tf = ds['tf'].values
    out = np.empty((gz.size, nv))
    pts = np.column_stack([my, mx])
    for k in range(gz.size):
        out[k] = RegularGridInterpolator((gy, gx), tf[k], method='linear', bounds_error=False, fill_value=np.nan)(pts)
    return out, -np.asarray(gz, float)


def tf_at_draft(path, nan_to_zero=False):
    tfm, dep = tf_on_mesh(path)
    if nan_to_zero:
        tfm = np.nan_to_num(tfm, nan=0.0)
    d = -base
    k = np.clip(np.searchsorted(dep, d, side='right') - 1, 0, dep.size - 2)
    a2 = np.clip((d - dep[k]) / (dep[k + 1] - dep[k]), 0.0, 1.0)
    i = np.arange(nv)
    return (1.0 - a2) * tfm[k, i] + a2 * tfm[k + 1, i]


def melt_v(tf_draft, g0, dt):
    m = g0 * coef * np.maximum(np.nan_to_num(tf_draft, nan=0.0)[elx] + dt[basin_el][:, None], 0.0) ** 2
    m[~el_float] = 0.0
    acc = np.zeros(nv)
    for c in range(3):
        np.add.at(acc, elx[:, c], m[:, c])
    return acc / np.maximum(n_conn, 1)


FLUX = rho_ice * varea / 1e12


def basin_totals(m):
    return np.bincount(basin_v[floating], weights=(m * FLUX)[floating], minlength=nb)


def basin_mean_kgm2(m, valid):
    sel = floating & valid
    num = np.bincount(basin_v[sel], weights=(m * rho_ice * varea)[sel], minlength=nb)
    den = np.bincount(basin_v[sel], weights=varea[sel], minlength=nb)
    out = np.full(nb, np.nan)
    out[den > 0] = num[den > 0] / den[den > 0]
    return out


# ---- targets -----------------------------------------------------------------------------------
obs = pd.read_csv(f'{OC}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv')
t1, s1 = obs['BMR (Gt/yr)'].values, obs['BMR uncert (Gt/yr)'].values
t2d = xr.open_dataset(f'{OC}/meltobs/melt_target_term2_v3.nc')
t2, s2 = t2d['melt_mean'].values, t2d['melt_mean_err'].values
bf = xr.open_dataset(f'{OC}/bfrns/BFRN_ismip8km_v2.nc')
bxx, byy = np.meshgrid(bf['x'].values, bf['y'].values)
vbin = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), bf['BFRN_bins'].values.ravel())(np.column_stack([mx, my]))
nbin = t2.size
vbin_i = np.where(np.isfinite(vbin), vbin, -1).astype(int)


def bin_totals(m):
    sel = floating & (vbin_i >= 0)
    return np.bincount(vbin_i[sel], weights=(m * FLUX)[sel], minlength=nbin)[:nbin]


tc = xr.open_dataset(f'{OMD}/melt_cold_target_term3_v2.nc')
tw = xr.open_dataset(f'{OMD}/melt_warm_target_term3_v2.nc')
rows = [str(r) for r in tc['model'].values]
row_to_model = json.load(open(f'{M}/ssa_melt_j3_refit.json'))['row_to_model']
d3 = tw['melt_rate'].values - tc['melt_rate'].values                       # (rows, basins)
s3 = np.sqrt(tw['melt_rate_uncert'].values ** 2 + tc['melt_rate_uncert'].values ** 2)
sm_ds = xr.open_dataset(f'{OC}/shelfmask/shelf_mask_ismip8km.nc')
smv = np.nan_to_num(sm_ds['shelf_mask'].values[0])
sxx, syy = np.meshgrid(sm_ds['x'].values, sm_ds['y'].values)
vshelf = NearestNDInterpolator(np.column_stack([sxx.ravel(), syy.ravel()]), smv.ravel())(np.column_stack([mx, my]))
shelf_ids = json.load(open(f'{M}/melt_rescore_real_targets.json'))['shelf_ids']
t4d = xr.open_dataset(f'{OC}/ocean_observations_data/melt_observations_target_term4.nc')
obs4 = [(str(r), int(y), float(t4d['melt_rate'].sel(region=r, year=y)), float(t4d['melt_rate_uncert'].sel(region=r, year=y)))
        for r in t4d['region'].values for y in t4d['year'].values
        if np.isfinite(float(t4d['melt_rate'].sel(region=r, year=y)))]
t4 = np.array([o[2] for o in obs4])
s4 = np.array([o[3] for o in obs4])

print('-- interpolating TF fields (Zhou, 7 ocean models x 2 states, observed Amundsen years)...', flush=True)
zdraft = tf_at_draft(ZHOU_TF, nan_to_zero=True)
ocean = {(r, s): tf_at_draft(f'{OMD}/{OCEAN_MODELS[row_to_model[r]].format(s=s)}') for r in rows for s in ('cold', 'warm')}
years = sorted({o[1] for o in obs4})
odraft = {y: tf_at_draft(f'{OC}/ocean_observations_data/Obs_{y}_TF.nc') for y in years}
shelf_mask = {r: (vshelf == shelf_ids[r]) & floating for r in shelf_ids}

# ---- per-basin deltaT refit (element-based totals for the root finding) ------------------------
fe = np.where(el_float)[0]
TFe = zdraft[elx[fe]]                                          # (nfe, 3)
We = FLUX[elx[fe]] * floating[elx[fe]] / n_conn[elx[fe]]       # vertex share of each element value
be = basin_el[fe]


def refit_dt(g0):
    lo, hi = np.full(nb, -8.0), np.full(nb, 40.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        tot = np.bincount(be, weights=(We * np.maximum(TFe + mid[be][:, None], 0.0) ** 2).sum(axis=1), minlength=nb) * g0 * coef
        lo = np.where(tot < t1, mid, lo)
        hi = np.where(tot >= t1, mid, hi)
    return 0.5 * (lo + hi)


def evaluate(g0, dt):
    m0 = melt_v(zdraft, g0, dt)
    r = {'dt': dt, 'bmb': basin_totals(m0), 'bin': bin_totals(m0)}
    p3 = np.full(d3.shape, np.nan)
    for i, row in enumerate(rows):
        pc = basin_mean_kgm2(melt_v(ocean[(row, 'cold')], g0, dt), np.isfinite(ocean[(row, 'cold')]))
        pw = basin_mean_kgm2(melt_v(ocean[(row, 'warm')], g0, dt), np.isfinite(ocean[(row, 'warm')]))
        p3[i] = pw - pc
    r['d3'] = p3
    v4 = np.empty(len(obs4))
    cache = {}
    for n, (reg, y, _, _) in enumerate(obs4):
        if y not in cache:
            cache[y] = melt_v(odraft[y], g0, dt)
        v4[n] = float(np.sum((cache[y] * FLUX)[shelf_mask[reg]]))
    r['j4'] = v4
    return r


def implaus(mod, tgt, sig, delta, floor):
    ok = np.isfinite(mod) & np.isfinite(tgt) & np.isfinite(sig)
    tot = np.sqrt(sig[ok] ** 2 + (delta * np.abs(tgt[ok])) ** 2 + floor ** 2)
    return float(np.sqrt(np.mean(((mod[ok] - tgt[ok]) / tot) ** 2)))


runs = {'refit': [], 'capped': []}
for g0 in GAMMA:
    dt = refit_dt(g0)
    for name, d in (('refit', dt), ('capped', np.clip(dt, -CAP, CAP))):
        r = evaluate(g0, d)
        r['gamma_0'] = float(g0)
        r['I'] = {}
        for delta in DELTAS:
            r['I'][delta] = {'J1': implaus(r['bmb'], t1, s1, delta, 1.0), 'J2': implaus(r['bin'], t2, s2, delta, 1.0),
                             'J3': implaus(r['d3'].ravel(), d3.ravel(), s3.ravel(), delta, 1.0),
                             'J4': implaus(r['j4'], t4, s4, delta, 1.0)}
        r['ratio3'] = float(np.nanmedian((r['d3'] / np.where(np.abs(d3) > 1, d3, np.nan)).ravel()))
        runs[name].append(r)
    print(f"   gamma_0 {g0:8.0f}: deltaT refit {dt.min():6.2f}..{dt.max():6.2f}; capped basins {int((np.abs(dt) > CAP).sum())}; "
          f"I(0.3) refit " + ' '.join(f"{k} {v:4.2f}" for k, v in runs['refit'][-1]['I'][0.3].items())
          + " | capped " + ' '.join(f"{k} {v:4.2f}" for k, v in runs['capped'][-1]['I'][0.3].items()), flush=True)

# ---- summaries -----------------------------------------------------------------------------------
summary = {}
lg = np.log(GAMMA)
for name, rr in runs.items():
    for delta in DELTAS:
        I = np.array([[r['I'][delta][k] for k in ('J1', 'J2', 'J3', 'J4')] for r in rr])
        nroy = np.all(I <= CUT, axis=1)
        logp = -0.5 * (I ** 2).sum(axis=1)
        w = np.exp(logp - logp.max())
        w /= w.sum()
        cdf = np.cumsum(w)
        q = [float(np.exp(np.interp(p, cdf, lg))) for p in (0.05, 0.5, 0.95)]
        summary[f'{name}_delta{delta}'] = {'nroy': [float(g) for g in GAMMA[nroy]], 'posterior_p05_p50_p95': q,
                                           'map': float(GAMMA[np.argmax(w)])}
        rng_txt = f'{GAMMA[nroy].min():.0f}-{GAMMA[nroy].max():.0f}' if nroy.any() else 'none'
        print(f"{name:7s} delta {delta}: not ruled out {rng_txt} ({int(nroy.sum())}/{GAMMA.size}); "
              f"posterior 5/50/95 {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}; MAP {GAMMA[np.argmax(w)]:.0f}")

print('\ncapped, delta 0.3: terms and diagnostics at selected gamma_0')
print('  gamma_0    I_J1  I_J2  I_J3  I_J4  warm-cold ratio  bin8 bin9 (obs 162 392)  PIG 2009/2012 (obs 72/29)')
k09 = [n for n, o in enumerate(obs4) if o[0] == 'pig' and o[1] in (2009, 2012)]
for r in runs['capped']:
    if np.any(np.isclose(r['gamma_0'], GAMMA[::4])) or r is runs['capped'][-1]:
        I = r['I'][0.3]
        print(f"  {r['gamma_0']:7.0f}  {I['J1']:5.2f} {I['J2']:5.2f} {I['J3']:5.2f} {I['J4']:5.2f}   {r['ratio3']:6.3f}"
              f"         {r['bin'][8]:5.0f} {r['bin'][9]:5.0f}            {r['j4'][k09[0]]:5.1f} / {r['j4'][k09[1]]:5.1f}")

# protocol Monte Carlo on the same grid (median-normalised, U[0,1] weights, targets sampled)
rng = np.random.default_rng(2)
for name in ('refit', 'capped'):
    rr = runs[name]
    B = np.stack([r['bmb'] for r in rr]); N = np.stack([r['bin'] for r in rr])
    D = np.stack([r['d3'] for r in rr]); V = np.stack([r['j4'] for r in rr])
    wbin = bf['BFRN_medians'].values[:nbin] / float(bf['BFRN_median'].values)
    best = []
    for _ in range(10):
        n = 10_000
        J1 = np.abs(B[None] - rng.normal(t1, s1, (n, nb))[:, None]).mean(2)
        J2 = (wbin * np.abs(N[None] - rng.normal(t2, s2, (n, nbin))[:, None])).mean(2)
        d3s = rng.normal(np.nan_to_num(d3), np.nan_to_num(s3), (n,) + d3.shape)
        ok = np.isfinite(d3)
        J3 = np.nanmean(np.where(ok, np.abs(np.nan_to_num(D)[None] - d3s[:, None]), np.nan).reshape(n, len(rr), -1), axis=2)
        J4 = np.abs(V[None] - rng.normal(t4, s4, (n, t4.size))[:, None]).mean(2)
        a = rng.uniform(0, 1, (n, 4))
        I = sum(a[:, i][:, None] * J / np.median(J) for i, J in enumerate((J1, J2, J3, J4)))
        best.append(GAMMA[np.argmin(I, axis=1)])
    best = np.concatenate(best)
    p = np.percentile(best, [5, 50, 95])
    summary[f'protocol_mc_{name}'] = p.tolist()
    print(f'protocol MC ({name}, J1-J4, 100k): 5/50/95 = {p[0]:.0f} / {p[1]:.0f} / {p[2]:.0f}')

json.dump({'gamma': GAMMA.tolist(), 'deltas': DELTAS, 'cap': CAP, 'cut': CUT, 'summary': summary,
           'runs': {n: [{'gamma_0': r['gamma_0'], 'dt': r['dt'].tolist(), 'bmb': r['bmb'].tolist(), 'bin': r['bin'].tolist(),
                         'j4': r['j4'].tolist(), 'ratio3': r['ratio3'], 'I': {str(k): v for k, v in r['I'].items()}}
                        for r in rr] for n, rr in runs.items()}},
          open(OUT, 'w'), indent=1)
print(f'\nsaved {OUT}\nDONE')
