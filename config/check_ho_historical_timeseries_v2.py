"""HO vs SSA vs CPOM dH/dt, updated with the chunk-1 rerun (v2 of check_ho_historical_timeseries.py).

HO now has: yearly 1996-2001 (chunk-1 rerun, full 0.1 yr history), a 2001->2013 mean rate
(H2013 from the final checkpoint's geometry minus H2001 from the rerun's last step -- valid because
the rerun reproduces the original chunk 1 exactly, worklog 4x), and yearly 2014-2019 (chunk 4).
HO's 1995 thickness is the rerun's own geometry (identical to the relaxation input, 4y addendum).
SSA: yearly 1996-2019. CPOM: yearly 1996-2019, plus its own 2002-2013 mean for the HO segment.
Same fixed-coverage grounded mask and area weighting as 4s/4w/4x. These are all from the
UNRELAXED-start HO run; the relaxed-start rerun will supersede them.
Writes NEW plot files (*_v2.png); the earlier plots are left in place.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import gc
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
YEARS = np.arange(1996, 2020)
EARLY = np.arange(1996, 2002)
LATE = np.arange(2014, 2020)


def stack_base(ts, nv, nkeep):
    if isinstance(ts, list):
        return np.stack([np.asarray(t.Thickness).ravel()[:nkeep] for t in ts])
    raw = np.asarray(ts.Thickness).ravel()
    return raw.reshape(raw.size // nv, nv)[:, :nkeep]


def at(times, stack, h0, t0, yr):
    if abs(yr - t0) < 1e-6:
        return h0
    i = int(np.argmin(np.abs(times - yr)))
    assert abs(times[i] - yr) < 1e-6, f'no step at {yr}'
    return stack[i]


print('-- HO chunk-1 rerun (1995-2001)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_chunk1rerun.nc')
nv3, nvb = md.mesh.numberofvertices, int(md.mesh.numberofvertices2d)
t0, dt = float(md.timestepping.start_time), float(md.timestepping.time_step)
S1 = stack_base(md.results.TransientSolution, nv3, nvb)
T1 = t0 + dt * np.arange(1, S1.shape[0] + 1)
H1995_h = np.asarray(md.geometry.thickness).ravel()[:nvb].copy()
xh = np.asarray(md.mesh.x).ravel()[:nvb].copy()
ho_early = np.stack([at(T1, S1, H1995_h, t0, float(y)) - at(T1, S1, H1995_h, t0, float(y - 1)) for y in EARLY])
H2001_h = at(T1, S1, H1995_h, t0, 2001.0).copy()
del md, S1
gc.collect()

print('-- HO final checkpoint (chunk 4, 2013-2019)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019.nc')
t4, dt4 = float(md.timestepping.start_time), float(md.timestepping.time_step)
assert abs(t4 - 2013.0) < 1e-6
S4 = stack_base(md.results.TransientSolution, md.mesh.numberofvertices, nvb)
T4 = t4 + dt4 * np.arange(1, S4.shape[0] + 1)
H2013_h = np.asarray(md.geometry.thickness).ravel()[:nvb].copy()
ho_late = np.stack([at(T4, S4, H2013_h, t4, float(y)) - at(T4, S4, H2013_h, t4, float(y - 1)) for y in LATE])
ho_mid = (H2013_h - H2001_h) / 12.0
del md, S4
gc.collect()

print('-- SSA historical...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv2 = md.mesh.numberofvertices
x = np.asarray(md.mesh.x).ravel(); y = np.asarray(md.mesh.y).ravel()
assert nv2 == nvb and np.allclose(x, xh), 'HO base layer != SSA mesh'
el = np.asarray(md.mesh.elements).astype(int) - 1
ts0, dts = float(md.timestepping.start_time), float(md.timestepping.time_step)
SS = stack_base(md.results.TransientSolution, nv2, nv2)
TS = ts0 + dts * np.arange(1, SS.shape[0] + 1)
H0_s = np.asarray(md.geometry.thickness).ravel().copy()
ssa = np.stack([at(TS, SS, H0_s, ts0, float(y_)) - at(TS, SS, H0_s, ts0, float(y_ - 1)) for y_ in YEARS])
il = np.asarray(md.mask.ice_levelset).ravel(); ol = np.asarray(md.mask.ocean_levelset).ravel()
del md, SS
gc.collect()

tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(nv2)
for c in range(3):
    np.add.at(area, el[:, c], ea / 3.0)

print('-- CPOM 1996-2019...', flush=True)
mip = xr.open_dataset(MIPKIT)
oy = mip['cpom_dhdt_time'].dt.year.values
obs = np.empty((YEARS.size, nv2), dtype=np.float32)
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
acc = np.zeros(nv2); cnt = np.zeros(nv2)
for c in range(3):
    np.add.at(acc, el[:, c], bel)
    np.add.at(cnt, el[:, c], 1)
basin = np.round(acc / np.maximum(cnt, 1)).astype(int)
mid_obs_idx = np.isin(YEARS, np.arange(2002, 2014))


def amr(stk, m):
    wv = area[m]
    stk = np.atleast_2d(stk)
    return (stk[:, m] * wv).sum(axis=1) / wv.sum()


def series(m):
    return dict(ssa=amr(ssa, m), obs=amr(obs, m), ho_e=amr(ho_early, m), ho_m=float(amr(ho_mid, m)[0]),
                ho_l=amr(ho_late, m), obs_m=float(amr(obs[mid_obs_idx], m).mean()),
                ssa_m=float(amr(ssa[mid_obs_idx], m).mean()), gt=area[m].sum() * 917.0 / 1e12)


def report(label, r):
    g = r['gt']
    print(f'\n[{label}]  (m/yr; Gt/yr in brackets)')
    print('   year    HO       SSA      CPOM')
    for i, yr in enumerate(YEARS):
        h = r['ho_e'][yr - 1996] if yr <= 2001 else (r['ho_l'][yr - 2014] if yr >= 2014 else np.nan)
        hs = f'{h:+8.4f}' if np.isfinite(h) else '    --  '
        print(f'   {yr} {hs} {r["ssa"][i]:+8.4f} {r["obs"][i]:+8.4f}')
    print(f'   2002-2013 mean: HO {r["ho_m"]:+.4f} [{r["ho_m"] * g:+.1f}]  SSA {r["ssa_m"]:+.4f} '
          f'[{r["ssa_m"] * g:+.1f}]  CPOM {r["obs_m"]:+.4f} [{r["obs_m"] * g:+.1f}]')
    print(f'   1996-2001 mean: HO {r["ho_e"].mean() * g:+.1f}  SSA {r["ssa"][:6].mean() * g:+.1f}  '
          f'CPOM {r["obs"][:6].mean() * g:+.1f} Gt/yr;  2014-2019 mean: HO {r["ho_l"].mean() * g:+.1f}  '
          f'SSA {r["ssa"][-6:].mean() * g:+.1f}  CPOM {r["obs"][-6:].mean() * g:+.1f} Gt/yr')


def draw(ax, r, title, legend=False):
    ax.plot(YEARS, r['obs'], 's-', ms=3, color='C1', label='CPOM')
    ax.plot(YEARS, r['ssa'], 'o-', ms=2.5, lw=1, color='C0', label='SSA')
    ax.plot(EARLY, r['ho_e'], 'o-', ms=3, color='C3', label='HO yearly')
    ax.plot(LATE, r['ho_l'], 'o-', ms=3, color='C3')
    ax.hlines(r['ho_m'], 2001.5, 2013.5, colors='C3', linestyles='--', lw=2, label='HO 2002-2013 mean')
    ax.hlines(r['obs_m'], 2001.5, 2013.5, colors='C1', linestyles=':', lw=2, label='CPOM 2002-2013 mean')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(fontsize=6)


dom = series(fixed)
report('whole domain, fixed coverage', dom)
res = {}
for b in range(16):
    m = fixed & (basin == b)
    if m.sum() >= 50:
        res[b] = series(m)
        report(f'ISMIP7 basin {b}', res[b])

fig, ax = plt.subplots(figsize=(11, 5.5))
draw(ax, dom, 'Whole domain, fixed coverage, area-weighted dH/dt (m/yr) -- unrelaxed-start runs', legend=True)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_vs_ssa_historical_dhdt_domain_v2.png', dpi=140)
n = len(res)
fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(17, 2.9 * np.ceil(n / 4)), sharex=True)
axes = np.atleast_1d(axes).ravel()
for i, (a, (b, r)) in enumerate(zip(axes, res.items())):
    draw(a, r, f'ISMIP7 basin {b}', legend=(i == 0))
for a in axes[n:]:
    a.axis('off')
fig.suptitle('HO (red) / SSA (blue) / CPOM (orange) dH/dt per basin, m/yr; dashed = 2002-2013 means', fontsize=10)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_vs_ssa_historical_dhdt_basins_v2.png', dpi=130)
print('\nSaved models/ho_vs_ssa_historical_dhdt_domain_v2.png and models/ho_vs_ssa_historical_dhdt_basins_v2.png')
print('DONE')
