"""Relaxed-start HO historical run (_relaxfix, worklog 5e) vs old HO vs SSA vs CPOM dH/dt.

Same fixed-coverage grounded mask, area weighting and ISMIP7/IMBIE2 basin assignment as
check_ho_historical_timeseries_v2.py (worklog 4w/4x), so the numbers are directly comparable.
- relaxfix HO: yearly 1996-2019 from the per-chunk history files (base-layer thickness at
  every integer year; 1995 = chunk 1's initial_thickness).
- old (unrelaxed-start) HO: only 1996-2001 yearly, a 2002-2013 mean and 2014-2019 yearly
  survive. They are parsed from the v2 log, not recomputed (that needs two 160 GB loads).
- SSA historical and CPOM: recomputed as in v2.
Writes models/ho_relaxfix_historical_dhdt_{domain,basins}.png and the series to
models/ho_relaxfix_historical_dhdt_series.npz. Read-only otherwise.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import gc
import re
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm
from scipy.interpolate import NearestNDInterpolator

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
BASIN_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/'
              'imbie2/basin_numbers_ismip8km_v2.nc')
MIPKIT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc'
V2_LOG = '/g/data/au88/jh7060/ACCESS-AIS3/config/launch_check_ho_historical_timeseries_v2.pbs.out'
YEARS = np.arange(1996, 2020)
EARLY = np.arange(1996, 2002)
LATE = np.arange(2014, 2020)
MID = (YEARS >= 2002) & (YEARS <= 2013)
RHO = 917.0

# -- relaxfix HO: thickness at every integer year 1995-2019 (base layer = SSA mesh) ----------
H_year = {}
for c0, c1 in [(1995, 2001), (2001, 2007), (2007, 2013), (2013, 2019)]:
    h = np.load(f'{MODEL_DIR}/AIS3_historical_1995_2019_relaxfix_history_{c0}-{c1}.npz')
    if c0 == 1995:
        H_year[1995] = h['initial_thickness'].astype(np.float64)
    for t, H in zip(h['times'], h['thickness']):
        if abs(t - round(t)) < 1e-6:
            H_year[int(round(t))] = H.astype(np.float64)
missing = [y for y in range(1995, 2020) if y not in H_year]
assert not missing, f'relaxfix history missing years {missing}'
nvb = H_year[1995].size
hof = np.stack([H_year[y] - H_year[y - 1] for y in YEARS])
del H_year
gc.collect()

# -- SSA historical (as in v2) -----------------------------------------------------------------
print('-- SSA historical...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv = md.mesh.numberofvertices
assert nv == nvb, 'relaxfix base layer != SSA mesh'
x = np.asarray(md.mesh.x).ravel(); y = np.asarray(md.mesh.y).ravel()
el = np.asarray(md.mesh.elements).astype(int) - 1
ts0, dts = float(md.timestepping.start_time), float(md.timestepping.time_step)
ts = md.results.TransientSolution
if isinstance(ts, list):
    SS = np.stack([np.asarray(t.Thickness).ravel() for t in ts])
else:
    raw = np.asarray(ts.Thickness).ravel()
    SS = raw.reshape(raw.size // nv, nv)
TS = ts0 + dts * np.arange(1, SS.shape[0] + 1)
H0_s = np.asarray(md.geometry.thickness).ravel().copy()


def at(yr):
    if abs(yr - ts0) < 1e-6:
        return H0_s
    i = int(np.argmin(np.abs(TS - yr)))
    assert abs(TS[i] - yr) < 1e-6
    return SS[i]


ssa = np.stack([at(float(yr)) - at(float(yr - 1)) for yr in YEARS])
il = np.asarray(md.mask.ice_levelset).ravel(); ol = np.asarray(md.mask.ocean_levelset).ravel()
del md, SS, ts
gc.collect()

tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(nv)
for c in range(3):
    np.add.at(area, el[:, c], ea / 3.0)

print('-- CPOM 1996-2019...', flush=True)
mip = xr.open_dataset(MIPKIT)
oy = mip['cpom_dhdt_time'].dt.year.values
obs = np.empty((YEARS.size, nv), dtype=np.float32)
for k, yr in enumerate(YEARS):
    ti = int(np.where(oy == yr)[0][0])
    obs[k] = pyissm.data.interp.xr_to_mesh(mip[['dhdt_cpom']].isel(cpom_dhdt_time=ti), 'dhdt_cpom',
                                           x, y, x_var='x1km', y_var='y1km')
fixed = (il < 0) & (ol > 0) & np.all(np.isfinite(obs), axis=0)
print(f'   fixed coverage: {fixed.sum()} vertices, {area[fixed].sum() / 1e12:.3f} Mkm^2 (v2: 599285, 7.563)')

ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
lk = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), ds_b['basinNumber'].values.ravel())
bel = lk(np.column_stack([tx.mean(1), ty.mean(1)])).astype(float)
acc = np.zeros(nv); cnt = np.zeros(nv)
for c in range(3):
    np.add.at(acc, el[:, c], bel)
    np.add.at(cnt, el[:, c], 1)
basin = np.round(acc / np.maximum(cnt, 1)).astype(int)

# -- old HO from the v2 log --------------------------------------------------------------------
old = {}
txt = open(V2_LOG).read()
for blk in re.split(r'\n\[', txt)[1:]:
    head = blk.split(']')[0]
    key = 'domain' if head.startswith('whole domain') else (int(head.split()[-1]) if head.startswith('ISMIP7 basin') else None)
    if key is None:
        continue
    yrs = {}
    for m in re.finditer(r'^\s+(\d{4})\s+([+-]\d\.\d+|--)\s+', blk, re.M):
        if m.group(2) != '--':
            yrs[int(m.group(1))] = float(m.group(2))
    mm = re.search(r'2002-2013 mean: HO ([+-]\d\.\d+)', blk)
    old[key] = dict(yr=yrs, mid=float(mm.group(1)))
print(f'   old HO parsed from v2 log: domain + {sum(isinstance(k, int) for k in old)} basins')


def amr(stk, m):
    wv = area[m]
    return (np.atleast_2d(stk)[:, m] * wv).sum(axis=1) / wv.sum()


def stats(a, o):
    return np.sqrt(np.mean((a - o) ** 2)), (np.corrcoef(a, o)[0, 1] if a.std() > 0 and o.std() > 0 else np.nan)


def series(m, key):
    r = dict(hof=amr(hof, m), ssa=amr(ssa, m), obs=amr(obs, m), gt=area[m].sum() * RHO / 1e12)
    o = old.get(key)
    r['old_e'] = np.array([o['yr'].get(int(yy), np.nan) for yy in EARLY]) if o else np.full(6, np.nan)
    r['old_l'] = np.array([o['yr'].get(int(yy), np.nan) for yy in LATE]) if o else np.full(6, np.nan)
    r['old_m'] = o['mid'] if o else np.nan
    return r


def report(label, r):
    g = r['gt']
    print(f'\n[{label}]  m/yr, area-weighted, fixed coverage ({g / RHO * 1e12 / 1e12:.3f} Mkm^2)')
    print('   year   HOrelax    oldHO      SSA     CPOM')
    for i, yr in enumerate(YEARS):
        ov = r['old_e'][yr - 1996] if yr <= 2001 else (r['old_l'][yr - 2014] if yr >= 2014 else np.nan)
        os_ = f'{ov:+8.4f}' if np.isfinite(ov) else '     -- '
        print(f'   {yr} {r["hof"][i]:+8.4f} {os_} {r["ssa"][i]:+8.4f} {r["obs"][i]:+8.4f}')
    seg = lambda a, s: a[s].mean() * g
    e, l = YEARS <= 2001, YEARS >= 2014
    print(f'   Gt/yr      1996-2001  2002-2013  2014-2019  1996-2019')
    print(f'   HO relax  {seg(r["hof"], e):+9.1f}  {seg(r["hof"], MID):+9.1f}  {seg(r["hof"], l):+9.1f}  {r["hof"].mean() * g:+9.1f}')
    print(f'   old HO    {np.nanmean(r["old_e"]) * g:+9.1f}  {r["old_m"] * g:+9.1f}  {np.nanmean(r["old_l"]) * g:+9.1f}        --')
    print(f'   SSA       {seg(r["ssa"], e):+9.1f}  {seg(r["ssa"], MID):+9.1f}  {seg(r["ssa"], l):+9.1f}  {r["ssa"].mean() * g:+9.1f}')
    print(f'   CPOM      {seg(r["obs"], e):+9.1f}  {seg(r["obs"], MID):+9.1f}  {seg(r["obs"], l):+9.1f}  {r["obs"].mean() * g:+9.1f}')
    for nm in ('hof', 'ssa'):
        rm, co = stats(r[nm], r['obs'])
        print(f'   {"HO relax" if nm == "hof" else "SSA":8s} vs CPOM yearly 1996-2019: RMSE {rm:.4f} m/yr, corr {co:+.2f}, '
              f'bias {np.mean(r[nm] - r["obs"]):+.4f} m/yr')
    sel = np.isin(YEARS, np.r_[EARLY, LATE])
    ov = np.r_[r['old_e'], r['old_l']]
    if np.all(np.isfinite(ov)):
        print(f'   on the 12 yrs old HO has (1996-2001, 2014-2019): RMSE vs CPOM  old HO {np.sqrt(np.mean((ov - r["obs"][sel]) ** 2)):.4f}, '
              f'HO relax {np.sqrt(np.mean((r["hof"][sel] - r["obs"][sel]) ** 2)):.4f}, SSA {np.sqrt(np.mean((r["ssa"][sel] - r["obs"][sel]) ** 2)):.4f}')


dom = series(fixed, 'domain')
report('whole domain', dom)
res = {}
for b in range(16):
    m = fixed & (basin == b)
    if m.sum() >= 50:
        res[b] = series(m, b)
        report(f'ISMIP7 basin {b}', res[b])

print('\n=== Summary per basin: 1996-2019 mean Gt/yr and yearly RMSE vs CPOM (m/yr) ===')
print(f'{"b":>3} {"HOrelax":>8} {"SSA":>8} {"CPOM":>8} {"RMSE_HOr":>9} {"RMSE_SSA":>9} {"corr_HOr":>9} {"corr_SSA":>9}  closer')
nwin = 0
for b, r in res.items():
    g = r['gt']
    rh, ch = stats(r['hof'], r['obs']); rs, cs = stats(r['ssa'], r['obs'])
    nwin += rh < rs
    print(f'{b:3d} {r["hof"].mean() * g:+8.1f} {r["ssa"].mean() * g:+8.1f} {r["obs"].mean() * g:+8.1f} {rh:9.4f} {rs:9.4f} '
          f'{ch:+9.2f} {cs:+9.2f}  {"HO" if rh < rs else "SSA"}')
print(f'   HO relax has the lower yearly RMSE in {nwin} of {len(res)} basins')

np.savez(f'{MODEL_DIR}/ho_relaxfix_historical_dhdt_series.npz', years=YEARS,
         **{f'{k}_{f}': (np.asarray(v[f]) if not np.isscalar(v[f]) else np.array(v[f]))
            for k, v in [('domain', dom)] + [(f'b{b}', r) for b, r in res.items()]
            for f in ('hof', 'ssa', 'obs', 'old_e', 'old_l', 'old_m', 'gt')})


def draw(ax, r, title, legend=False):
    ax.plot(YEARS, r['obs'], 's-', ms=3, color='C1', label='CPOM')
    ax.plot(YEARS, r['ssa'], 'o-', ms=2.5, lw=1, color='C0', label='SSA')
    ax.plot(YEARS, r['hof'], 'o-', ms=3, lw=1.5, color='C2', label='HO relaxed start')
    ax.plot(EARLY, r['old_e'], 'x--', ms=4, lw=0.8, color='C3', label='old HO')
    ax.plot(LATE, r['old_l'], 'x--', ms=4, lw=0.8, color='C3')
    if np.isfinite(r['old_m']):
        ax.hlines(r['old_m'], 2001.5, 2013.5, colors='C3', linestyles=':', lw=1.5, label='old HO 2002-13 mean')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(fontsize=6)


fig, ax = plt.subplots(figsize=(11, 5.5))
draw(ax, dom, 'Whole domain, fixed coverage, area-weighted grounded dH/dt (m/yr)', legend=True)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_relaxfix_historical_dhdt_domain.png', dpi=140)
n = len(res)
fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(17, 2.9 * np.ceil(n / 4)), sharex=True)
axes = np.atleast_1d(axes).ravel()
for i, (a, (b, r)) in enumerate(zip(axes, res.items())):
    draw(a, r, f'ISMIP7 basin {b}', legend=(i == 0))
for a in axes[n:]:
    a.axis('off')
fig.suptitle('HO relaxed start (green) / old HO (red) / SSA (blue) / CPOM (orange): dH/dt per basin, m/yr', fontsize=10)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_relaxfix_historical_dhdt_basins.png', dpi=130)
print('\nSaved models/ho_relaxfix_historical_dhdt_domain.png, ..._basins.png, ..._series.npz')
print('DONE')
