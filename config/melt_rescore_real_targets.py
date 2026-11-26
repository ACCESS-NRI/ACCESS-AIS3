"""Re-score the SSA melt candidates with the real ISMIP7 J2 target and with J4 (worklog 6.29).

gamma_0=300 was chosen by J2 alone (J1 ~0 by construction once deltaT is refit per basin; J3 was
added later, 6.1). But that J2 target was synthetic: each basin's observed melt spread over the
buttressing bins in proportion to floating area (melt_ismip7_calibration.py,
ssa_melt_deltaT_refit_recalibrate.py). The real ISMIP7 term-2 target (meltobs/melt_target_term2_v3.nc)
puts 51% of the melt in bins 8-9 (the proxy: 33%), and 392 vs 174 Gt/yr in bin 9. A proxy that
assumes uniform melt per unit area favours the most uniform melt, i.e. small gamma_0 with large
deltaT. J4 (Amundsen interannual variability, ocean_observations_data/) was never used.

For the 9 per-basin-refit SSA candidates (models/deltaT_refit_state_ssa.json), melt is evaluated
with ISSM's ismip6 local formula in Python, as in ssa_melt_j3_refit.py (checked there against ISSM
solves to 3e-14 m/yr), on models/AIS3_melt_final_ssa.nc:
  J1        mean |basin BMB - Paolo/Davison/Adusumilli| (Gt/yr), Zhou climatology
  J2proxy   the old synthetic-target J2 (must reproduce the published J2 values: sanity check)
  J2        same BFRN-weighted form against melt_target_term2_v3.nc
  J3        from models/ssa_melt_j3_refit.json (7 ocean models, cold/warm)
  J4        mean |shelf BMB - observed| (Gt/yr) for Pine Island and Dotson under each observed
            year's Amundsen TF (Obs_<year>_TF.nc); J4min = PIG 2009 + 2012 (the protocol minimum)
Then the protocol's Monte Carlo (100,000 samples: weights U[0,1], targets drawn from N(mean, sigma),
terms normalised by their median over the converged candidates) for several term sets, and the
sigma-normalised misfits (each bin/year divided by its target uncertainty).
Read-only; writes models/melt_rescore_real_targets.json (refuses to overwrite).
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
OUT = f'{M}/melt_rescore_real_targets.json'
OC = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
           'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
J2_PUBLISHED = [115.443, 12.624, 45.529, 102.877, 151.365, 209.268, 240.196, 268.723, 345.259]
N_MC = 100_000
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


def tf_at_draft(tf_mesh, depths):
    d = -base
    k = np.clip(np.searchsorted(depths, d, side='right') - 1, 0, depths.size - 2)
    a2 = np.clip((d - depths[k]) / (depths[k + 1] - depths[k]), 0.0, 1.0)
    i = np.arange(nv)
    return (1.0 - a2) * tf_mesh[k, i] + a2 * tf_mesh[k + 1, i]


def melt_v(tf_draft, g0, dt):
    m = g0 * coef * np.maximum(tf_draft[elx] + dt[basin_el][:, None], 0.0) ** 2
    m[~el_float] = 0.0
    acc = np.zeros(nv)
    for c in range(3):
        np.add.at(acc, elx[:, c], m[:, c])
    return acc / np.maximum(n_conn, 1)


def gt(melt, mask):
    return float(np.nansum(melt[mask & floating] * rho_ice * varea[mask & floating]) / 1e12)


# targets and masks
obs = pd.read_csv(f'{OC}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv')
b_obs, b_sig = obs['BMR (Gt/yr)'].values, obs['BMR uncert (Gt/yr)'].values
t2 = xr.open_dataset(f'{OC}/meltobs/melt_target_term2_v3.nc')
bin_obs, bin_sig = t2['melt_mean'].values, t2['melt_mean_err'].values
bf = xr.open_dataset(f'{OC}/bfrns/BFRN_ismip8km_v2.nc')
bgrid = bf['BFRN_bins'].values
bxx, byy = np.meshgrid(bf['x'].values, bf['y'].values)
vbin = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), bgrid.ravel())(np.column_stack([mx, my]))
nbin = int(np.nanmax(bgrid[np.isfinite(bgrid)])) + 1
assert nbin == bin_obs.size, (nbin, bin_obs.size)
w_bin = bf['BFRN_medians'].values[:nbin] / float(bf['BFRN_median'].values)
area_bb = np.array([[varea[(basin_v == b) & floating & (vbin == k)].sum() for k in range(nbin)] for b in range(nb)])
bin_proxy = (b_obs[:, None] * area_bb / np.maximum(area_bb.sum(axis=1, keepdims=True), 1)).sum(axis=0)

sm = xr.open_dataset(f'{OC}/shelfmask/shelf_mask_ismip8km.nc')['shelf_mask'].values[0]
sxx, syy = np.meshgrid(xr.open_dataset(f'{OC}/shelfmask/shelf_mask_ismip8km.nc')['x'].values,
                       xr.open_dataset(f'{OC}/shelfmask/shelf_mask_ismip8km.nc')['y'].values)
vshelf = NearestNDInterpolator(np.column_stack([sxx.ravel(), syy.ravel()]), np.nan_to_num(sm).ravel())(np.column_stack([mx, my]))
shelf_ids = {}
for name, (lat, lon) in {'pig': (-75.1, -100.5), 'dotson': (-74.4, -112.4)}.items():
    px, py = pyissm.tools.general.ll_to_xy(np.array([lat]), np.array([lon]), -1)
    cand = np.where(np.nan_to_num(sm).ravel() > 0)[0]
    j = cand[np.argmin((sxx.ravel()[cand] - px[0]) ** 2 + (syy.ravel()[cand] - py[0]) ** 2)]
    sid = int(sm.ravel()[j])
    msk = (vshelf == sid) & floating
    shelf_ids[name] = sid
    print(f'{name}: shelf id {sid} at ({sxx.ravel()[j] / 1e3:.0f},{syy.ravel()[j] / 1e3:.0f}) km; '
          f'{varea[msk].sum() / 1e6:.0f} km^2 floating on the mesh; basins {np.unique(basin_v[msk]).tolist()}', flush=True)
t4 = xr.open_dataset(f'{OC}/ocean_observations_data/melt_observations_target_term4.nc')
years = [int(y) for y in t4['year'].values]
regions = [str(r) for r in t4['region'].values]

with open(f'{M}/deltaT_refit_state_ssa.json') as f:
    st = json.load(f)
j3s = {c['i']: c for c in json.load(open(f'{M}/ssa_melt_j3_refit.json'))['candidates']}
cands = []
for i_str, c in st['candidates'].items():
    i = int(i_str)
    cands.append({'i': i, 'gamma_0': float(c['gamma_0']), 'converged': bool(c['converged']),
                  'delta_t': np.asarray(c['history'][-1]['delta_t'], float),
                  'J3': j3s[i]['J3'], 'sens_ratio': j3s[i]['sens_ratio_median']})
cands.sort(key=lambda c: c['i'])

# present day
zm, zd = tf_on_mesh(ZHOU_TF)
zdraft = tf_at_draft(np.nan_to_num(zm, nan=0.0), zd)   # as the ISSM input (NaN -> 0)
for c in cands:
    m = melt_v(zdraft, c['gamma_0'], c['delta_t'])
    c['bmb'] = np.array([gt(m, basin_v == b) for b in range(nb)])
    c['bmb_bin'] = np.array([gt(m, vbin == k) for k in range(nbin)])
    c['J1'] = float(np.mean(np.abs(c['bmb'] - b_obs)))
    c['J2proxy'] = float(np.mean(w_bin * np.abs(c['bmb_bin'] - bin_proxy)))
    c['J2'] = float(np.mean(w_bin * np.abs(c['bmb_bin'] - bin_obs)))
    c['J2sig'] = float(np.mean(np.abs(c['bmb_bin'] - bin_obs) / bin_sig))
del zm

# J4
c4 = {(r, y): [] for r in regions for y in years}
for y in years:
    f = f'{OC}/ocean_observations_data/Obs_{y}_TF.nc'
    om, od = tf_on_mesh(f)
    odraft = tf_at_draft(om, od)
    for r in regions:
        tgt = float(t4['melt_rate'].sel(region=r, year=y))
        if not np.isfinite(tgt):
            continue
        msk = (vshelf == shelf_ids[r]) & floating
        cover = np.isfinite(odraft[msk]).mean()
        for c in cands:
            m = melt_v(np.nan_to_num(odraft, nan=0.0), c['gamma_0'], c['delta_t'])
            c.setdefault('j4', {})[f'{r}_{y}'] = gt(m, msk)
        print(f'   J4 {r} {y}: target {tgt:.1f} +- {float(t4["melt_rate_uncert"].sel(region=r, year=y)):.1f} Gt/yr; '
              f'TF coverage on shelf {cover:.2f}', flush=True)
    del om
keys4 = [k for k in cands[0]['j4']]
t4v = np.array([float(t4['melt_rate'].sel(region=k.split('_')[0], year=int(k.split('_')[1]))) for k in keys4])
t4s = np.array([float(t4['melt_rate_uncert'].sel(region=k.split('_')[0], year=int(k.split('_')[1]))) for k in keys4])
kmin = [keys4.index(k) for k in ('pig_2009', 'pig_2012')]
for c in cands:
    v = np.array([c['j4'][k] for k in keys4])
    c['J4'] = float(np.mean(np.abs(v - t4v)))
    c['J4min'] = float(np.mean(np.abs(v[kmin] - t4v[kmin])))
    c['J4sig'] = float(np.mean(np.abs(v - t4v) / t4s))
    c['pig_09_12'] = (float(v[kmin[0]]), float(v[kmin[1]]))

print('\n i  gamma_0 conv |   J1   J2proxy (published)    J2   J2/sig |    J3  sens | J4min   J4  J4/sig | PIG 2009 / 2012 (obs %.0f / %.0f)'
      % (t4v[kmin[0]], t4v[kmin[1]]))
ok_proxy = True
for c in cands:
    pub = J2_PUBLISHED[c['i']]
    ok_proxy &= abs(c['J2proxy'] - pub) / pub < 0.05
    print(f"{c['i']:2d} {c['gamma_0']:8.1f} {'y' if c['converged'] else 'n':>4} | {c['J1']:6.2f} {c['J2proxy']:7.1f} ({pub:7.1f}) {c['J2']:7.1f} {c['J2sig']:6.2f} | "
          f"{c['J3']:6.0f} {c['sens_ratio']:5.3f} | {c['J4min']:5.1f} {c['J4']:5.1f} {c['J4sig']:6.2f} | {c['pig_09_12'][0]:6.1f} / {c['pig_09_12'][1]:6.1f}")
print(f"J2proxy reproduces the published J2 within 5%: {ok_proxy}")
print('\nbin totals (Gt/yr), real target / proxy / candidates:')
print('  bin ' + ' '.join(f'{k:6d}' for k in range(nbin)))
print('  obs ' + ' '.join(f'{v:6.0f}' for v in bin_obs))
print('  prx ' + ' '.join(f'{v:6.0f}' for v in bin_proxy))
for c in cands:
    print(f"  {c['gamma_0']:5.0f}" + ' '.join(f'{v:6.0f}' for v in c['bmb_bin']))

# Monte Carlo, protocol form, converged candidates only
use = [c for c in cands if c['converged']]
g = np.array([c['gamma_0'] for c in use])
rng = np.random.default_rng(1)
BMB = np.stack([c['bmb'] for c in use])          # (nc, nb)
BIN = np.stack([c['bmb_bin'] for c in use])      # (nc, nbin)
J4V = np.stack([[c['j4'][k] for k in keys4] for c in use])
J3 = np.array([c['J3'] for c in use])


def sample_terms(n):
    ob = rng.normal(b_obs, b_sig, (n, nb))
    oi = rng.normal(bin_obs, bin_sig, (n, nbin))
    o4 = rng.normal(t4v, t4s, (n, len(keys4)))
    inc = rng.integers(0, 2, (n, len(keys4))).astype(bool)
    inc[inc.sum(axis=1) == 0, kmin[0]] = True
    j1 = np.abs(BMB[None] - ob[:, None]).mean(axis=2)
    j2 = (w_bin * np.abs(BIN[None] - oi[:, None])).mean(axis=2)
    d4 = np.abs(J4V[None] - o4[:, None])
    j4 = (d4 * inc[:, None]).sum(axis=2) / inc.sum(axis=1)[:, None]
    j3 = np.broadcast_to(J3, j1.shape)
    return {'J1': j1, 'J2': j2, 'J3': j3, 'J4': j4}


res = {}
chunks = [sample_terms(10_000) for _ in range(N_MC // 10_000)]
T = {k: np.concatenate([ch[k] for ch in chunks]) for k in chunks[0]}
med = {k: np.median(T[k]) for k in T}
print(f'\nMonte Carlo ({N_MC} samples; converged candidates {g.tolist()}): optimal gamma_0 5th/50th/95th pct, and the share each wins')
for combo in (('J1', 'J2'), ('J1', 'J2', 'J3'), ('J1', 'J2', 'J3', 'J4'), ('J2', 'J3', 'J4'), ('J3', 'J4'), ('J2', 'J4')):
    a = rng.uniform(0, 1, (N_MC, len(combo)))
    I = sum(a[:, n][:, None] * T[k] / med[k] for n, k in enumerate(combo))
    best = g[np.argmin(I, axis=1)]
    p = np.percentile(best, [5, 50, 95])
    share = {f'{v:.0f}': float(np.mean(best == v)) for v in g}
    res['+'.join(combo)] = {'p05_p50_p95': p.tolist(), 'share': share}
    print(f"  {'+'.join(combo):12s}: {p[0]:7.0f} {p[1]:7.0f} {p[2]:7.0f}   " + ', '.join(f'{k}: {v:.0%}' for k, v in share.items() if v >= 0.01))

json.dump({'shelf_ids': shelf_ids, 'j4_keys': keys4, 'bin_obs': bin_obs.tolist(), 'bin_proxy': bin_proxy.tolist(),
           'candidates': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in cands],
           'monte_carlo': res}, open(OUT, 'w'), indent=1)
print(f'\nsaved {OUT}\nDONE')
