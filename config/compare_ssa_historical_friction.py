"""Score the SSA historical runs (friction law x melt calibration) against observed change, 1995-2019.

Runs (worklog 6.31-6.32), whichever exist:
  prod300           AIS3_historical_1995_2019_SSA.nc            (production melt gamma_0=300, Budd p=1, fixed N)
  hm4000            AIS3_historical_1995_2019_SSA_hm4000.nc     (new melt, Budd p=1, fixed N)
  hm4000_<variant>  ..._hm4000_{p1_flotN,p3_fixedN,p3_flotN}.nc (new melt, other friction laws)
All variants start from the same drag (drag-preserving rescale), so they differ only in their transient.
Observations (MIPKIT AntarcticaObsISMIP7-v1.2.nc):
  A  CPOM dH/dt, yearly 1996-2019, on the fixed-coverage grounded mask (grounded and CPOM finite in every
     year), as plot_historical_relaxfix_comparison.py: Gt/yr for 1996-2001 / 2002-2013 / 2014-2019, whole
     domain and per IMBIE2 basin; RMSE of the yearly area-mean series.
  B  speed change, NSIDC-0720 annual mosaics: ratio |v|(2019) / |v|(2007) (and 2019/2001) on grounded ice
     with both observed speeds > 100 m/yr, against the model's Vel at the same times; per basin median of
     log(model ratio / obs ratio), and the observed and modelled speed-up of Amundsen (basin 9) fast ice.
Reads only; writes models/ssa_historical_friction_scores.json (refuses to overwrite).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import json
import numpy as np
import netCDF4
import xarray as xr
import pyissm
from scipy.interpolate import NearestNDInterpolator

M = 'models'
OUT = f"{M}/ssa_historical_friction_scores{os.environ.get('AIS3_SCORE_TAG', '')}.json"
MIPKIT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc'
BASIN_FILE = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/imbie2/basin_numbers_ismip8km_v2.nc'
RUNS = {'prod300': 'AIS3_historical_1995_2019_SSA.nc', 'hm4000': 'AIS3_historical_1995_2019_SSA_hm4000.nc'}
for v in ('p1_flotN', 'p3_fixedN', 'p3_flotN', 'p5_flotN'):
    RUNS[f'hm4000_{v}'] = f'AIS3_historical_1995_2019_SSA_hm4000_{v}.nc'
# OCX 'vary' ocean pathway (worklog 6.34)
RUNS['prod300_ocxvary'] = 'AIS3_historical_1995_2019_SSA_ocxvary.nc'
RUNS['hm4000_ocxvary'] = 'AIS3_historical_1995_2019_SSA_hm4000_ocxvary.nc'
for v in ('p1_flotN', 'p3_fixedN', 'p3_flotN'):
    RUNS[f'hm4000_{v}_ocxvary'] = f'AIS3_historical_1995_2019_SSA_hm4000_{v}_ocxvary.nc'
RUNS['hm4000_frontgreene'] = 'AIS3_historical_1995_2019_SSA_hm4000_frontgreene.nc'  # observed front (worklog 6.37)
RUNS = {k: f'{M}/{f}' for k, f in RUNS.items() if os.path.exists(f'{M}/{f}')}
YEARS = np.arange(1996, 2020)
PERIODS = {'1996-2001': (1996, 2001), '2002-2013': (2002, 2013), '2014-2019': (2014, 2019)}
RHO_I = 917.0
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')
print('runs:', list(RUNS), flush=True)

d0 = netCDF4.Dataset(RUNS['prod300'])
x = np.asarray(d0['mesh'].variables['x'][:], float).ravel()
y = np.asarray(d0['mesh'].variables['y'][:], float).ravel()
el = np.asarray(d0['mesh'].variables['elements'][:]).astype(int) - 1
if el.shape[0] == 3:
    el = el.T
il = np.asarray(d0['mask'].variables['ice_levelset'][:], float).ravel()
ol = np.asarray(d0['mask'].variables['ocean_levelset'][:], float).ravel()
nv = x.size
tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.bincount(el.ravel(), weights=np.repeat(ea / 3, 3), minlength=nv)
ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
lk = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), ds_b['basinNumber'].values.ravel())
bel = lk(np.column_stack([tx.mean(1), ty.mean(1)])).astype(float)
basin = np.round(np.bincount(el.ravel(), weights=np.repeat(bel, 3), minlength=nv) /
                 np.maximum(np.bincount(el.ravel(), minlength=nv), 1)).astype(int)

mip = xr.open_dataset(MIPKIT)
oy = mip['cpom_dhdt_time'].dt.year.values
obs = np.empty((YEARS.size, nv), dtype=np.float32)
for k, yr in enumerate(YEARS):
    obs[k] = pyissm.data.interp.xr_to_mesh(mip[['dhdt_cpom']].isel(cpom_dhdt_time=int(np.where(oy == yr)[0][0])),
                                           'dhdt_cpom', x, y, x_var='x1km', y_var='y1km')
fixed = (il < 0) & (ol > 0) & np.all(np.isfinite(obs), axis=0)
print(f'fixed coverage: {fixed.sum()} vertices, {area[fixed].sum() / 1e12:.3f} Mkm^2', flush=True)
vt = mip['vel_time'].values
vyear = np.array([t.astype('datetime64[D]').astype(object).year + (t.astype('datetime64[D]').astype(object).timetuple().tm_yday - 1) / 365.25 for t in vt])


def obs_speed(i):
    vx = pyissm.data.interp.xr_to_mesh(mip[['vx_timeseries']].isel(vel_time=i), 'vx_timeseries', x, y)
    vy = pyissm.data.interp.xr_to_mesh(mip[['vy_timeseries']].isel(vel_time=i), 'vy_timeseries', x, y)
    return np.hypot(vx, vy)


EPOCHS = {2001: int(np.argmin(np.abs(vyear - 2001.1))), 2007: int(np.argmin(np.abs(vyear - 2007.1))),
          2019: int(np.argmin(np.abs(vyear - 2019.1)))}
vobs = {yr: obs_speed(i) for yr, i in EPOCHS.items()}
print('velocity epochs used:', {yr: round(float(vyear[i]), 2) for yr, i in EPOCHS.items()}, flush=True)
grounded = (il < 0) & (ol > 0)


def amean(stk, m):
    return (stk[:, m] * area[m]).sum(axis=1) / area[m].sum()


regions = {'domain': fixed}
for b in range(16):
    if (fixed & (basin == b)).sum() >= 50:
        regions[b] = fixed & (basin == b)


def period_gt(series, m):
    tot = area[m].sum() * RHO_I / 1e12
    return {p: float(series[(YEARS >= a) & (YEARS <= b)].mean() * tot) for p, (a, b) in PERIODS.items()}


res = {'obs': {}}
for rk, m in regions.items():
    s = amean(obs, m)
    res['obs'][str(rk)] = {'periods': period_gt(s, m)}

for name, path in RUNS.items():
    d = netCDF4.Dataset(path)
    r = d['results']['TransientSolution']
    t = np.asarray(r.variables['time'][:], float)
    H0 = np.asarray(d['geometry'].variables['thickness'][:], float).ravel()

    def at(var, yr, tol=0.06):
        i = int(np.argmin(np.abs(t - yr)))
        assert abs(t[i] - yr) < tol, (name, var, yr, t[i])
        return np.asarray(r.variables[var][i], float).ravel()

    Hy = {1995: H0}
    for yr in range(1996, 2020):
        Hy[yr] = at('Thickness', float(yr))
    dh = np.stack([Hy[yr] - Hy[yr - 1] for yr in YEARS])
    out = {}
    for rk, m in regions.items():
        s = amean(dh, m)
        out[str(rk)] = {'periods': period_gt(s, m), 'rmse_yearly': float(np.sqrt(np.mean((s - amean(obs, m)) ** 2)))}
    vel = {yr: at('Vel', float(vyear[i]), tol=0.1) for yr, i in EPOCHS.items()}  # 2019 mosaic is dated 2019.09, runs end 2019.0
    sp = {}
    for y0 in (2001, 2007):
        ok = grounded & (vobs[y0] > 100) & (vobs[2019] > 100) & np.isfinite(vobs[y0]) & np.isfinite(vobs[2019]) & (vel[y0] > 1)
        lr = np.log((vel[2019] / vel[y0]) / (vobs[2019] / vobs[y0]))
        sp[str(y0)] = {'n': int(ok.sum()), 'median_log_ratio_all': float(np.median(lr[ok])),
                       'rms_log_ratio_all': float(np.sqrt(np.mean(lr[ok] ** 2)))}
        for b in range(16):
            mb = ok & (basin == b)
            if mb.sum() >= 50:
                w = vobs[y0][mb] * area[mb]
                sp[str(y0)][f'b{b}'] = {'n': int(mb.sum()), 'obs_ratio_fw': float(np.sum(w * vobs[2019][mb] / vobs[y0][mb]) / w.sum()),
                                        'mod_ratio_fw': float(np.sum(w * vel[2019][mb] / vel[y0][mb]) / w.sum()),
                                        'median_log_ratio': float(np.median(lr[mb]))}
    out['speed'] = sp
    res[name] = out
    d.close()
    print(f'-- {name} done', flush=True)

runs = [k for k in res if k != 'obs']
print('\n== CPOM dH/dt, fixed coverage, Gt/yr (1996-2001 | 2002-2013 | 2014-2019) and yearly RMSE (m/yr)')
for rk in ['domain', 9, 14, 7, 0]:
    if str(rk) not in res['obs']:
        continue
    o = res['obs'][str(rk)]['periods']
    print(f"  [{rk}] CPOM            " + ' | '.join(f'{o[p]:7.1f}' for p in PERIODS))
    for k in runs:
        q = res[k][str(rk)]
        print(f"  [{rk}] {k:26s} " + ' | '.join(f"{q['periods'][p]:7.1f}" for p in PERIODS) + f"   RMSE {q['rmse_yearly']:.4f}")
print('\n== speed change 2007 -> 2019 on grounded ice (both obs > 100 m/yr): flux-weighted ratio obs / model, '
      'and median log(model/obs); overall RMS log ratio')
for k in runs:
    s = res[k]['speed']['2007']
    a9 = s.get('b9', {})
    print(f"  {k:26s} all: median log {s['median_log_ratio_all']:+.3f}, RMS {s['rms_log_ratio_all']:.3f} (n {s['n']}); "
          f"Amundsen b9 obs {a9.get('obs_ratio_fw', np.nan):.3f} model {a9.get('mod_ratio_fw', np.nan):.3f}")
print('\n== per basin, 2007 -> 2019: model/obs median log ratio')
bs = [b for b in range(16) if f'b{b}' in res[runs[0]]['speed']['2007']]
print('  run              ' + ' '.join(f'{b:>6d}' for b in bs))
for k in runs:
    print(f'  {k:26s} ' + ' '.join(f"{res[k]['speed']['2007'][f'b{b}']['median_log_ratio']:+6.3f}" for b in bs))
json.dump(res, open(OUT, 'w'), indent=1)
print(f'\nsaved {OUT}\nDONE')
