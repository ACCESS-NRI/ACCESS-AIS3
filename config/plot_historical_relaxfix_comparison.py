"""1995-2019 dH/dt: relaxed-start HO (D2 rheology, `_relaxfix`) vs the old unrelaxed HO run vs SSA vs CPOM.

Same definitions as check_ho_historical_timeseries_v2.py (worklog 4s/4w/4x/5b): grounded vertices
with CPOM data in every year 1996-2019 (fixed coverage), area-weighted means of yearly thickness
change (m/yr), whole domain and per ISMIP7 basin.
- HO relaxfix: yearly 1996-2019 from its per-chunk history files (annual base-layer thickness) and
  the 1995 thickness stored in the first one.
- Old HO: parsed from that script's printed output (launch_check_ho_historical_timeseries_v2.pbs.out:
  yearly 1996-2001 and 2014-2019, 2002-2013 mean) -- its per-step history no longer exists in full.
- SSA and CPOM: recomputed here, and checked against the same printed output so the mask is the same.
Writes docs/figures/historical_dhdt_relaxfix_{domain,basins}.png and prints the numbers.
"""
import os
import re
import glob
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm
from scipy.interpolate import NearestNDInterpolator

MODEL_DIR = 'models'
FIG_DIR = 'docs/figures'
OLD_LOG = 'config/launch_check_ho_historical_timeseries_v2.pbs.out'
BASIN_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/'
              'imbie2/basin_numbers_ismip8km_v2.nc')
MIPKIT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc'
YEARS = np.arange(1996, 2020)
MID = (YEARS >= 2002) & (YEARS <= 2013)

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de'
C_SSA, C_OLD, C_NEW = '#2a78d6', '#eb6834', '#1baf7a'  # validated slots 1-3 (all-pairs, light)
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.labelcolor': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'axes.titlelocation': 'left',
})

# -- old HO (and SSA/CPOM for the consistency check) from the earlier printed output ------------
old = {}
txt = open(OLD_LOG).read().split('\n')
cur = None
for line in txt:
    m = re.match(r'\[(whole domain, fixed coverage|ISMIP7 basin (\d+))\]', line)
    if m:
        cur = 'domain' if m.group(2) is None else int(m.group(2))
        old[cur] = {'ho': {}, 'ssa': {}, 'obs': {}}
        continue
    if cur is None:
        continue
    m = re.match(r'\s+(\d{4})\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
    if m:
        yr = int(m.group(1))
        old[cur]['ho'][yr] = np.nan if m.group(2) == '--' else float(m.group(2))
        old[cur]['ssa'][yr], old[cur]['obs'][yr] = float(m.group(3)), float(m.group(4))
    m = re.match(r'\s+2002-2013 mean: HO (\S+)', line)
    if m:
        old[cur]['ho_mid'] = float(m.group(1))
print(f'parsed old-HO series for {len(old)} regions', flush=True)

# -- SSA historical (yearly) --------------------------------------------------------------------
print('-- SSA historical...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv = md.mesh.numberofvertices
x = np.asarray(md.mesh.x, dtype=float).ravel()
y = np.asarray(md.mesh.y, dtype=float).ravel()
el = np.asarray(md.mesh.elements).astype(int) - 1
ts = md.results.TransientSolution
steps = list(ts.steps) if hasattr(ts, 'steps') else (ts if isinstance(ts, list) else [ts])
st_t = np.array([float(s.time) for s in steps])
H0_s = np.asarray(md.geometry.thickness, dtype=float).ravel().copy()
il = np.asarray(md.mask.ice_levelset).ravel()
ol = np.asarray(md.mask.ocean_levelset).ravel()


def ssa_H(yr):
    if yr == 1995:
        return H0_s
    i = int(np.argmin(np.abs(st_t - yr)))
    assert abs(st_t[i] - yr) < 1e-6, yr
    return np.asarray(steps[i].Thickness, dtype=float).ravel()


ssa = np.stack([ssa_H(yr) - ssa_H(yr - 1) for yr in YEARS])
del md, ts, steps

# -- HO relaxfix (yearly, base layer from the history files) -------------------------------------
print('-- HO relaxfix...', flush=True)
Hn = {}
for f in sorted(glob.glob(f'{MODEL_DIR}/AIS3_historical_1995_2019_relaxfix_history_*.npz')):
    d = np.load(f)
    if '1995-2001' in f:
        Hn[1995] = d['initial_thickness'].astype(float)
    for k, t in enumerate(d['times']):
        if abs(t - round(t)) < 1e-6:
            Hn[int(round(t))] = d['thickness'][k].astype(float)
assert all(yr in Hn for yr in range(1995, 2020)), sorted(Hn)
assert Hn[1995].size == nv
new = np.stack([Hn[yr] - Hn[yr - 1] for yr in YEARS])

# -- geometry, CPOM, mask, basins -----------------------------------------------------------------
tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(nv)
for c in range(3):
    np.add.at(area, el[:, c], ea / 3.0)
print('-- CPOM...', flush=True)
mip = xr.open_dataset(MIPKIT)
oy = mip['cpom_dhdt_time'].dt.year.values
obs = np.empty((YEARS.size, nv), dtype=np.float32)
for k, yr in enumerate(YEARS):
    ti = int(np.where(oy == yr)[0][0])
    obs[k] = pyissm.data.interp.xr_to_mesh(mip[['dhdt_cpom']].isel(cpom_dhdt_time=ti), 'dhdt_cpom',
                                           x, y, x_var='x1km', y_var='y1km')
fixed = (il < 0) & (ol > 0) & np.all(np.isfinite(obs), axis=0)
print(f'   fixed coverage: {fixed.sum()} vertices, {area[fixed].sum() / 1e12:.3f} Mkm^2')
ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
lk = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), ds_b['basinNumber'].values.ravel())
bel = lk(np.column_stack([tx.mean(1), ty.mean(1)])).astype(float)
acc, cnt = np.zeros(nv), np.zeros(nv)
for c in range(3):
    np.add.at(acc, el[:, c], bel)
    np.add.at(cnt, el[:, c], 1)
basin = np.round(acc / np.maximum(cnt, 1)).astype(int)


def amean(stk, m):
    return (stk[:, m] * area[m]).sum(axis=1) / area[m].sum()


regions = {'domain': fixed}
for b in range(16):
    if (fixed & (basin == b)).sum() >= 50:
        regions[b] = fixed & (basin == b)

res = {}
worst = 0.0
for key, m in regions.items():
    r = {'ssa': amean(ssa, m), 'obs': amean(obs, m), 'new': amean(new, m), 'gt': area[m].sum() * 917.0 / 1e12}
    if key in old:  # same mask as the earlier run? SSA and CPOM must reproduce its printed values
        for name in ('ssa', 'obs'):
            ref = np.array([old[key][name][yr] for yr in YEARS])
            worst = max(worst, np.max(np.abs(np.round(r[name], 4) - ref)))
        r['old'] = np.array([old[key]['ho'][yr] for yr in YEARS])
        r['old_mid'] = old[key]['ho_mid']
    res[key] = r
print(f'consistency vs earlier output: max |SSA or CPOM difference| = {worst:.4f} m/yr (0.0001 = rounding)')
assert worst <= 0.00015, 'mask or method differs from the earlier comparison'

for key, r in res.items():
    g = r['gt']
    lab = 'whole domain, fixed coverage' if key == 'domain' else f'ISMIP7 basin {key}'
    rmse = lambda a: np.sqrt(np.nanmean((a - r['obs']) ** 2))
    corr = lambda a: np.corrcoef(a, r['obs'])[0, 1]
    print(f"\n[{lab}] Gt/yr means 1996-2001 | 2002-2013 | 2014-2019:")
    for nm, v in (('CPOM', r['obs']), ('SSA', r['ssa']), ('HO relaxfix', r['new'])):
        print(f"   {nm:12s} {v[:6].mean() * g:+8.1f} | {v[MID].mean() * g:+8.1f} | {v[-6:].mean() * g:+8.1f}"
              f"   RMSE vs CPOM {rmse(v):.4f} m/yr, corr {corr(v):+.2f}" if nm != 'CPOM' else
              f"   {nm:12s} {v[:6].mean() * g:+8.1f} | {v[MID].mean() * g:+8.1f} | {v[-6:].mean() * g:+8.1f}")
    if 'old' in r:
        o = r['old']
        print(f"   {'HO old':12s} {np.nanmean(o[:6]) * g:+8.1f} | {r['old_mid'] * g:+8.1f} | {np.nanmean(o[-6:]) * g:+8.1f}")


def draw(ax, r, title, legend=False):
    ax.plot(YEARS, r['obs'], color=INK, marker='s', ms=3, lw=1.2, label='CPOM (altimetry)')
    ax.plot(YEARS, r['ssa'], color=C_SSA, lw=1.4, label='SSA')
    if 'old' in r:
        ax.plot(YEARS, r['old'], color=C_OLD, lw=1.4, ls='-', label='HO, old (unrelaxed start)')
        ax.hlines(r['old_mid'], 2001.5, 2013.5, colors=C_OLD, linestyles='--', lw=1.4)
    ax.plot(YEARS, r['new'], color=C_NEW, lw=1.8, label='HO, relaxed start + D2')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(fontsize=7, loc='lower right')


os.makedirs(FIG_DIR, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 5))
draw(ax, res['domain'], 'Grounded ice with CPOM coverage every year (7.56 Mkm²), area-weighted', legend=True)
ax.set_ylabel('dH/dt (m/yr)')
ax.text(2007.5, res['domain']['old_mid'], 'old HO: 2002–2013 mean only', fontsize=7.5, color=INK2,
        ha='center', va='bottom')
fig.suptitle('Historical thickness change 1996–2019: relaxed-start HO vs old HO, SSA and CPOM',
             x=0.01, ha='left', fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/historical_dhdt_relaxfix_domain.png', dpi=170)

keys = [k for k in res if k != 'domain']
n = len(keys)
fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(17, 2.9 * np.ceil(n / 4)), sharex=True)
axes = np.atleast_1d(axes).ravel()
for i, (a, k) in enumerate(zip(axes, keys)):
    draw(a, res[k], f'ISMIP7 basin {k}', legend=(i == 0))
for a in axes[n:]:
    a.axis('off')
fig.suptitle('dH/dt per basin (m/yr): CPOM (black), SSA (blue), old HO (orange; dashed = 2002–2013 mean), '
             'relaxed HO (green)', x=0.01, ha='left', fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/historical_dhdt_relaxfix_basins.png', dpi=130)
print('\nwrote docs/figures/historical_dhdt_relaxfix_domain.png, ..._basins.png\nDONE')
