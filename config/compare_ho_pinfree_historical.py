"""Pin-free HO historical run (_budd_nopins, worklog 6.28) vs the pinned HO run (_relaxfix) vs CPOM dH/dt.

Same fixed-coverage grounded mask (grounded and CPOM finite in every year 1996-2019), area weighting and
IMBIE2 basins as compare_ssa_historical_friction.py / plot_historical_relaxfix_comparison.py, so the numbers
sit next to the SSA tables (6.13, 6.33). HO thickness is the base layer of each run's per-chunk history
files (every integer year; 1995 from chunk 1's initial_thickness). Also reports thickness-change extremes.
Read-only; writes models/ho_pinfree_historical_scores.json (refuses to overwrite).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import glob
import json
import numpy as np
import netCDF4
import xarray as xr
import pyissm
from scipy.interpolate import NearestNDInterpolator

M = 'models'
OUT = f'{M}/ho_pinfree_historical_scores.json'
MIPKIT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc'
BASIN_FILE = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/imbie2/basin_numbers_ismip8km_v2.nc'
TAGS = {'HO pinned (relaxfix)': 'relaxfix', 'HO pin-free (budd_nopins)': 'budd_nopins'}
YEARS = np.arange(1996, 2020)
PERIODS = {'1996-2001': (1996, 2001), '2002-2013': (2002, 2013), '2014-2019': (2014, 2019)}
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')

d0 = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
x = np.asarray(d0['mesh'].variables['x'][:], float).ravel()
y = np.asarray(d0['mesh'].variables['y'][:], float).ravel()
el = np.asarray(d0['mesh'].variables['elements'][:]).astype(int) - 1
if el.shape[0] == 3:
    el = el.T
nv = x.size
tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.bincount(el.ravel(), weights=np.repeat(ea / 3, 3), minlength=nv)
ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
bel = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), ds_b['basinNumber'].values.ravel())(
    np.column_stack([tx.mean(1), ty.mean(1)])).astype(float)
basin = np.round(np.bincount(el.ravel(), weights=np.repeat(bel, 3), minlength=nv) /
                 np.maximum(np.bincount(el.ravel(), minlength=nv), 1)).astype(int)
# the SSA historical run's start masks define the fixed-coverage mask, as in the SSA comparison
hs = netCDF4.Dataset(f'{M}/AIS3_historical_1995_2019_SSA.nc')
il = np.asarray(hs['mask'].variables['ice_levelset'][:], float).ravel()
ol = np.asarray(hs['mask'].variables['ocean_levelset'][:], float).ravel()
hs.close()

mip = xr.open_dataset(MIPKIT)
oy = mip['cpom_dhdt_time'].dt.year.values
obs = np.empty((YEARS.size, nv), dtype=np.float32)
for k, yr in enumerate(YEARS):
    obs[k] = pyissm.data.interp.xr_to_mesh(mip[['dhdt_cpom']].isel(cpom_dhdt_time=int(np.where(oy == yr)[0][0])),
                                           'dhdt_cpom', x, y, x_var='x1km', y_var='y1km')
fixed = (il < 0) & (ol > 0) & np.all(np.isfinite(obs), axis=0)
print(f'fixed coverage: {fixed.sum()} vertices, {area[fixed].sum() / 1e12:.3f} Mkm^2', flush=True)
regions = {'domain': fixed}
for b in range(16):
    if (fixed & (basin == b)).sum() >= 50:
        regions[b] = fixed & (basin == b)


def amean(stk, m):
    return (stk[:, m] * area[m]).sum(axis=1) / area[m].sum()


def periods(series, m):
    tot = area[m].sum() * 917.0 / 1e12
    return {p: float(series[(YEARS >= a) & (YEARS <= b)].mean() * tot) for p, (a, b) in PERIODS.items()}


res = {'CPOM': {str(r): {'periods': periods(amean(obs, m), m)} for r, m in regions.items()}}
for label, tag in TAGS.items():
    H = {}
    for f in sorted(glob.glob(f'{M}/AIS3_historical_1995_2019_{tag}_history_*.npz')):
        h = np.load(f)
        if '1995-2001' in f:
            H[1995] = h['initial_thickness'].astype(float)
        for k, t in enumerate(h['times']):
            if abs(t - round(t)) < 1e-6:
                H[int(round(t))] = h['thickness'][k].astype(float)
    missing = [yr for yr in range(1995, 2020) if yr not in H]
    assert not missing, (tag, missing)
    dh = np.stack([H[yr] - H[yr - 1] for yr in YEARS])
    out = {str(r): {'periods': periods(amean(dh, m), m), 'rmse_yearly': float(np.sqrt(np.mean((amean(dh, m) - amean(obs, m)) ** 2)))}
           for r, m in regions.items()}
    tot = H[2019] - H[1995]
    out['extremes'] = {'max_abs_dH_1995_2019': float(np.abs(tot).max()), 'n_abs_dH_gt_300': int((np.abs(tot) > 300).sum()),
                       'n_abs_dH_gt_1000': int((np.abs(tot) > 1000).sum())}
    res[label] = out
    print(f'-- {label} done', flush=True)

print('\n== dH/dt on the fixed-coverage mask, Gt/yr (1996-2001 | 2002-2013 | 2014-2019), yearly RMSE (m/yr)')
for r in ['domain', 9, 14, 7, 0]:
    if str(r) not in res['CPOM']:
        continue
    for k in res:
        q = res[k][str(r)]
        print(f"  [{r}] {k:26s} " + ' | '.join(f"{q['periods'][p]:7.1f}" for p in PERIODS)
              + (f"   RMSE {q['rmse_yearly']:.4f}" if 'rmse_yearly' in q else ''))
print('\n== thickness change 1995-2019')
for k in TAGS:
    e = res[k]['extremes']
    print(f"  {k:26s} max |dH| {e['max_abs_dH_1995_2019']:.0f} m; |dH| > 300 m: {e['n_abs_dH_gt_300']}; > 1000 m: {e['n_abs_dH_gt_1000']}")
json.dump(res, open(OUT, 'w'), indent=1)
print(f'\nsaved {OUT}\nDONE')
