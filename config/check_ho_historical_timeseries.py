"""HO historical run vs CPOM, using only the HO output that survived the chunked run, with SSA
computed under identical definitions alongside it.

What survived for HO: H(1995) (AIS3_ho_relaxed.nc geometry), H(2013) (the final checkpoint's
geometry, i.e. chunk 4's initial condition -- solve() does not overwrite md.geometry), and chunk
4's full 0.1 yr history 2013.1-2019.0. H(2001)/H(2007) were overwritten (see worklog 4u/4v).

Compared quantities (per base vertex, then area-weighted over the SAME fixed-coverage grounded
mask used in check_ssa_historical_fixedcov_basins.py: grounded AND dhdt_cpom finite in all
years 1996-2019):
  * mean rate 1995->2013 = (H2013 - H1995) / 18, vs the mean of CPOM's 1996-2013 annual slices
  * yearly rate 2014..2019 = H(yr) - H(yr-1), vs CPOM's slice for that year
Whole domain and per ISMIP7 ocean basin (16; = Mouginot 18 with 8+9 and 16+17 merged).
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
OBS_YEARS = np.arange(1996, 2020)
LATE_YEARS = np.arange(2014, 2020)


def stack_thickness(ts, nv, n_keep):
    """(n_steps, n_keep) thickness from a TransientSolution list or the collapsed bare object."""
    if isinstance(ts, list):
        return np.stack([np.asarray(t.Thickness).ravel()[:n_keep] for t in ts])
    raw = np.asarray(ts.Thickness).ravel()
    return raw.reshape(raw.size // nv, nv)[:, :n_keep]


# ---------------- SSA (2D mesh, full 240-step history) ----------------
print('-- SSA: loading AIS3_historical_1995_2019_SSA.nc...', flush=True)
mds = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv2 = mds.mesh.numberofvertices
x2 = np.asarray(mds.mesh.x).ravel().copy()
y2 = np.asarray(mds.mesh.y).ravel().copy()
el2 = np.asarray(mds.mesh.elements).astype(int) - 1
t0 = float(mds.timestepping.start_time)
dts = float(mds.timestepping.time_step)
Hs = stack_thickness(mds.results.TransientSolution, nv2, nv2)
times_s = t0 + dts * np.arange(1, Hs.shape[0] + 1)
H0_s = np.asarray(mds.geometry.thickness).ravel().copy()   # 1995.0 initial condition


def H_at_s(yr):
    if abs(yr - t0) < 1e-6:
        return H0_s
    i = int(np.argmin(np.abs(times_s - yr)))
    assert abs(times_s[i] - yr) < 1e-6, f'no SSA step at {yr}'
    return Hs[i]


ssa_mean = (H_at_s(2013.0) - H_at_s(1995.0)) / 18.0
ssa_late = np.stack([H_at_s(float(y)) - H_at_s(float(y - 1)) for y in LATE_YEARS])
ssa_yearly_all = np.stack([H_at_s(float(y)) - H_at_s(float(y - 1)) for y in OBS_YEARS])
il_s = np.asarray(mds.mask.ice_levelset).ravel()
ol_s = np.asarray(mds.mask.ocean_levelset).ravel()
del mds, Hs
gc.collect()

# ---------------- HO (3D mesh; base layer only) ----------------
print('-- HO: loading AIS3_historical_1995_2019.nc (final checkpoint, chunk 4)...', flush=True)
mdh = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019.nc')
nv3 = mdh.mesh.numberofvertices
nvb = int(mdh.mesh.numberofvertices2d)
on_base = np.asarray(mdh.mesh.vertexonbase).astype(bool).ravel()
assert on_base[:nvb].all() and on_base.sum() == nvb, 'expected base-layer vertices to be the first nv2d'
xh = np.asarray(mdh.mesh.x).ravel()[:nvb]
yh = np.asarray(mdh.mesh.y).ravel()[:nvb]
print(f'   HO: {nv3} 3D vertices, {nvb} per layer ({nv3 / nvb:.2f} layers); '
      f'timestepping {mdh.timestepping.start_time}->{mdh.timestepping.final_time}')
same_mesh = (nvb == nv2) and np.allclose(xh, x2) and np.allclose(yh, y2)
print(f'   HO base layer identical to SSA mesh (same vertices, same order): {same_mesh}')
if not same_mesh:
    raise RuntimeError('HO base mesh differs from SSA mesh -- this script assumes a shared 2D '
                       'mesh for a like-for-like comparison; extend it before using the output.')

H2013_h = np.asarray(mdh.geometry.thickness).ravel()[:nvb].copy()
th = float(mdh.timestepping.start_time)
dth = float(mdh.timestepping.time_step)
Hh = stack_thickness(mdh.results.TransientSolution, nv3, nvb)
times_h = th + dth * np.arange(1, Hh.shape[0] + 1)
print(f'   chunk-4 steps: {Hh.shape[0]}, {times_h[0]:.2f}..{times_h[-1]:.2f}')
d_first = Hh[0] - H2013_h
print(f'   sanity: H(first step) - geometry.thickness: mean {d_first.mean():+.4f} m, '
      f'p99 |.| {np.percentile(np.abs(d_first), 99):.3f} m (small => geometry is the 2013 state)')
il_h = np.asarray(mdh.mask.ice_levelset).ravel()[:nvb].copy()
ol_h = np.asarray(mdh.mask.ocean_levelset).ravel()[:nvb].copy()


def H_at_h(yr):
    if abs(yr - th) < 1e-6:
        return H2013_h
    i = int(np.argmin(np.abs(times_h - yr)))
    assert abs(times_h[i] - yr) < 1e-6, f'no HO step at {yr}'
    return Hh[i]


ho_late = np.stack([H_at_h(float(y)) - H_at_h(float(y - 1)) for y in LATE_YEARS])
del mdh, Hh
gc.collect()

print('-- HO: loading AIS3_ho_relaxed.nc for H(1995)...', flush=True)
mdr = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_relaxed.nc')
H1995_h = np.asarray(mdr.geometry.thickness).ravel()[:nvb].copy()
del mdr
gc.collect()
ho_mean = (H2013_h - H1995_h) / 18.0

# ---------------- geometry, masks, obs ----------------
tri_x, tri_y = x2[el2], y2[el2]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
area = np.zeros(nv2)
for c in range(3):
    np.add.at(area, el2[:, c], elem_area / 3.0)

grounded = (il_s < 0) & (ol_s > 0)
print(f'   grounded mask agreement SSA-final vs HO-2013 state: '
      f'{100 * np.mean(grounded == ((il_h < 0) & (ol_h > 0))):.2f}% of vertices')

print('-- Interpolating CPOM dhdt for 1996-2019...', flush=True)
mip = xr.open_dataset(MIPKIT)
oy = mip['cpom_dhdt_time'].dt.year.values
obs = np.empty((OBS_YEARS.size, nv2), dtype=np.float32)
for k, yr in enumerate(OBS_YEARS):
    ti = int(np.where(oy == yr)[0][0])
    obs[k] = pyissm.data.interp.xr_to_mesh(mip[['dhdt_cpom']].isel(cpom_dhdt_time=ti), 'dhdt_cpom',
                                           x2, y2, x_var='x1km', y_var='y1km')
fixed = grounded & np.all(np.isfinite(obs), axis=0)
print(f'   fixed coverage: {fixed.sum()} vertices, {area[fixed].sum() / 1e12:.3f} Mkm^2')
obs_mean = obs[np.isin(OBS_YEARS, np.arange(1996, 2014))].mean(axis=0)
obs_late = obs[np.isin(OBS_YEARS, LATE_YEARS)]

ds_b = xr.open_dataset(BASIN_FILE)
bg = ds_b['basinNumber'].values
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
lk = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), bg.ravel())
bel = lk(np.column_stack([tri_x.mean(1), tri_y.mean(1)])).astype(float)
acc = np.zeros(nv2); cnt = np.zeros(nv2)
for c in range(3):
    np.add.at(acc, el2[:, c], bel)
    np.add.at(cnt, el2[:, c], 1)
basin = np.round(acc / np.maximum(cnt, 1)).astype(int)
rho = 917.0


def am(v, m):
    return float((v[m] * area[m]).sum() / area[m].sum())


def am_rows(stk, m):
    return np.array([am(r, m) for r in stk])


def report(label, m):
    a = area[m].sum()
    gt = a * rho / 1e12
    r = dict(area=a / 1e12,
             ho95=am(ho_mean, m), ssa95=am(ssa_mean, m), obs95=am(obs_mean, m),
             hoL=am_rows(ho_late, m), ssaL=am_rows(ssa_late, m), obsL=am_rows(obs_late, m))
    print(f'\n[{label}]  area {r["area"]:.3f} Mkm^2')
    print(f'   1995-2013 mean (m/yr):  HO {r["ho95"]:+.4f}  SSA {r["ssa95"]:+.4f}  obs {r["obs95"]:+.4f}'
          f'   (Gt/yr: HO {r["ho95"] * gt:+.1f}  SSA {r["ssa95"] * gt:+.1f}  obs {r["obs95"] * gt:+.1f})')
    print(f'   2014-2019 mean (m/yr):  HO {r["hoL"].mean():+.4f}  SSA {r["ssaL"].mean():+.4f}  '
          f'obs {r["obsL"].mean():+.4f}'
          f'   (Gt/yr: HO {r["hoL"].mean() * gt:+.1f}  SSA {r["ssaL"].mean() * gt:+.1f}  '
          f'obs {r["obsL"].mean() * gt:+.1f})')
    print('   yearly ' + '  '.join(f'{y}: HO {h:+.4f} SSA {s:+.4f} obs {o:+.4f}'
                                   for y, h, s, o in zip(LATE_YEARS, r['hoL'], r['ssaL'], r['obsL'])))
    return r


print('\n=========== FIXED-COVERAGE, AREA-WEIGHTED ===========')
dom = report('whole domain', fixed)
basins = {}
for b in range(16):
    m = fixed & (basin == b)
    if m.sum() >= 50:
        basins[b] = report(f'ISMIP7 basin {b}', m)

print('\n=== Compact per-basin table (Gt/yr) ===')
print(f'{"b":>2} {"area":>6} | {"HO 95-13":>9} {"SSA 95-13":>9} {"obs 96-13":>9} | '
      f'{"HO 14-19":>9} {"SSA 14-19":>9} {"obs 14-19":>9}')
for b, r in basins.items():
    gt = r['area'] * 1e12 * rho / 1e12
    print(f'{b:2d} {r["area"]:6.3f} | {r["ho95"] * gt:+9.1f} {r["ssa95"] * gt:+9.1f} {r["obs95"] * gt:+9.1f} | '
          f'{r["hoL"].mean() * gt:+9.1f} {r["ssaL"].mean() * gt:+9.1f} {r["obsL"].mean() * gt:+9.1f}')
print('NOTE: 6 yearly points (2014-2019) are too few for a meaningful correlation; compare levels.')

print('\n-- Plotting...', flush=True)


def draw(ax, r, m, title):
    ssa_all = am_rows(ssa_yearly_all, m)
    obs_all = am_rows(obs, m)
    ax.plot(OBS_YEARS, obs_all, 's-', ms=3, color='C1', label='obs (CPOM)')
    ax.plot(OBS_YEARS, ssa_all, '-', lw=1, color='C0', alpha=0.45, label='SSA yearly')
    ax.hlines(r['obs95'], 1995.5, 2013.0, colors='C1', linestyles=':', lw=2)
    ax.hlines(r['ssa95'], 1995.5, 2013.0, colors='C0', linestyles='--', lw=2, label='SSA 1995-2013 mean')
    ax.hlines(r['ho95'], 1995.5, 2013.0, colors='C3', linestyles='--', lw=2, label='HO 1995-2013 mean')
    ax.plot(LATE_YEARS, r['hoL'], 'o-', ms=3, color='C3', label='HO yearly')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


fig, ax = plt.subplots(figsize=(10, 5.5))
draw(ax, dom, fixed, 'Whole domain, fixed coverage, area-weighted dH/dt (m/yr)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_vs_ssa_historical_dhdt_domain.png', dpi=140)

n = len(basins)
fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(17, 2.9 * np.ceil(n / 4)), sharex=True)
axes = np.atleast_1d(axes).ravel()
for ax, (b, r) in zip(axes, basins.items()):
    draw(ax, r, fixed & (basin == b), f'ISMIP7 basin {b}')
for ax in axes[n:]:
    ax.axis('off')
axes[0].legend(fontsize=6)
fig.suptitle('HO (red) vs SSA (blue) vs CPOM (orange): fixed-coverage area-weighted dH/dt per basin '
             '(m/yr); dashed = 1995-2013 means', fontsize=10)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_vs_ssa_historical_dhdt_basins.png', dpi=130)
print('Saved models/ho_vs_ssa_historical_dhdt_domain.png and models/ho_vs_ssa_historical_dhdt_basins.png')
print('DONE')
