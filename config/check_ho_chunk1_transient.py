"""HO start-up transient test, 1996-2001, from the chunk-1 rerun (AIS3_historical_1995_2019_chunk1rerun.nc,
full 0.1 yr history), alongside SSA under identical definitions -- the HO counterpart of worklog 4t.

1. Reproducibility: recompute the statistics check_historical_chunk1.py printed for the ORIGINAL chunk 1
   (whose output was overwritten) on the full 3D arrays, and print them next to the originals.
2. Yearly dH/dt 1996-2001, fixed-coverage grounded mask (grounded AND dhdt_cpom finite 1996-2019, as in
   4s/4w), area-weighted: HO, SSA, CPOM, whole domain and per ISMIP7 basin.
3. Split each model's dH/dt into SMB (the model's own md.smb.mass_balance, average of year and previous
   year as in 4t) and the dynamic residual dH/dt - SMB. A start-up transient shows as a residual that
   changes quickly in the first years; compare its size in HO vs SSA.
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
YEARS = np.arange(1996, 2002)
FIX_YEARS = np.arange(1996, 2020)


def full_stack(ts, nv, field):
    if isinstance(ts, list):
        return np.stack([np.asarray(getattr(t, field)).ravel() for t in ts])
    raw = np.asarray(getattr(ts, field)).ravel()
    return raw.reshape(raw.size // nv, nv)


def smb_by_year(md, n_keep):
    mb = np.asarray(md.smb.mass_balance)
    return dict(zip(mb[-1, :].astype(int), [mb[:n_keep, j] for j in range(mb.shape[1])]))


# ---------------- HO rerun ----------------
print('-- HO: loading AIS3_historical_1995_2019_chunk1rerun.nc...', flush=True)
mdh = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_chunk1rerun.nc')
nv3 = mdh.mesh.numberofvertices
nvb = int(mdh.mesh.numberofvertices2d)
assert np.asarray(mdh.mesh.vertexonbase).astype(bool).ravel()[:nvb].all()
th = float(mdh.timestepping.start_time)
dth = float(mdh.timestepping.time_step)
print(f'   timestepping {th}->{mdh.timestepping.final_time}, dt={dth}')
Hh3 = full_stack(mdh.results.TransientSolution, nv3, 'Thickness')
times_h = th + dth * np.arange(1, Hh3.shape[0] + 1)
print(f'   {Hh3.shape[0]} steps {times_h[0]:.2f}..{times_h[-1]:.2f}')
H0h3 = np.asarray(mdh.geometry.thickness).ravel().copy()
d0 = Hh3[0] - H0h3
print(f'   sanity: H(first step) - geometry.thickness mean {d0.mean():+.4f} m, p99 |.| '
      f'{np.percentile(np.abs(d0), 99):.3f} m (small => geometry is the 1995 state)')

print('\n=== Reproducibility vs ORIGINAL chunk 1 (full 3D arrays, same definitions as check_historical_chunk1.py) ===')
dH3 = Hh3[-1] - H0h3
orig_p = {0: -1544.68, 1: -77.79, 5: -30.92, 25: -8.65, 50: -0.30, 75: 8.51, 95: 44.92, 99: 121.73, 100: 1012.11}
for p, o in orig_p.items():
    print(f'   p{p:5.1f}: rerun {np.nanpercentile(dH3, p):9.2f} m   original {o:9.2f} m')
print(f'   mean|dH|: rerun {np.nanmean(np.abs(dH3)):.2f} m   original 16.71 m')
ol0_3 = np.asarray(mdh.mask.ocean_levelset).ravel()
ol1_3 = full_stack(mdh.results.TransientSolution, nv3, 'MaskOceanLevelset')[-1]
g0, g1 = ol0_3 >= 0, ol1_3 >= 0
print(f'   newly floating: rerun {int((g0 & ~g1).sum())}   original 130380')
print(f'   newly grounded: rerun {int((~g0 & g1).sum())}   original 484350')
del ol1_3, dH3

Hh = Hh3[:, :nvb].copy()
H0h = H0h3[:nvb].copy()
del Hh3, H0h3
smb_h = smb_by_year(mdh, nvb)
xh = np.asarray(mdh.mesh.x).ravel()[:nvb].copy()
yh = np.asarray(mdh.mesh.y).ravel()[:nvb].copy()
del mdh
gc.collect()


def Hh_at(yr):
    if abs(yr - th) < 1e-6:
        return H0h
    i = int(np.argmin(np.abs(times_h - yr)))
    assert abs(times_h[i] - yr) < 1e-6
    return Hh[i]


# ---------------- SSA ----------------
print('\n-- SSA: loading AIS3_historical_1995_2019_SSA.nc...', flush=True)
mds = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv2 = mds.mesh.numberofvertices
x2 = np.asarray(mds.mesh.x).ravel().copy()
y2 = np.asarray(mds.mesh.y).ravel().copy()
assert nv2 == nvb and np.allclose(x2, xh) and np.allclose(y2, yh), 'HO base layer != SSA mesh'
el2 = np.asarray(mds.mesh.elements).astype(int) - 1
ts0 = float(mds.timestepping.start_time)
dts = float(mds.timestepping.time_step)
Hs = full_stack(mds.results.TransientSolution, nv2, 'Thickness')
times_s = ts0 + dts * np.arange(1, Hs.shape[0] + 1)
H0s = np.asarray(mds.geometry.thickness).ravel().copy()
smb_s = smb_by_year(mds, nv2)
il = np.asarray(mds.mask.ice_levelset).ravel()
ol = np.asarray(mds.mask.ocean_levelset).ravel()
del mds
gc.collect()


def Hs_at(yr):
    if abs(yr - ts0) < 1e-6:
        return H0s
    i = int(np.argmin(np.abs(times_s - yr)))
    assert abs(times_s[i] - yr) < 1e-6
    return Hs[i]


print(f'   SSA vs HO initial (1995) thickness on base layer: mean diff {np.mean(H0h - H0s):+.3f} m, '
      f'p99 |.| {np.percentile(np.abs(H0h - H0s), 99):.2f} m')

ho_rate = np.stack([Hh_at(float(y)) - Hh_at(float(y - 1)) for y in YEARS])
ssa_rate = np.stack([Hs_at(float(y)) - Hs_at(float(y - 1)) for y in YEARS])
ho_smb = np.stack([0.5 * (smb_h[y] + smb_h[y - 1]) for y in YEARS])
ssa_smb = np.stack([0.5 * (smb_s[y] + smb_s[y - 1]) for y in YEARS])
del Hh, Hs
gc.collect()

# ---------------- masks, obs, basins ----------------
tri_x, tri_y = x2[el2], y2[el2]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
area = np.zeros(nv2)
for c in range(3):
    np.add.at(area, el2[:, c], elem_area / 3.0)
grounded = (il < 0) & (ol > 0)

print('-- Interpolating CPOM for the fixed-coverage mask (1996-2019)...', flush=True)
mip = xr.open_dataset(MIPKIT)
oy = mip['cpom_dhdt_time'].dt.year.values
finite_all = np.ones(nv2, dtype=bool)
obs = {}
for yr in FIX_YEARS:
    ti = int(np.where(oy == yr)[0][0])
    o = pyissm.data.interp.xr_to_mesh(mip[['dhdt_cpom']].isel(cpom_dhdt_time=ti), 'dhdt_cpom',
                                      x2, y2, x_var='x1km', y_var='y1km')
    finite_all &= np.isfinite(o)
    if yr in YEARS:
        obs[yr] = o
fixed = grounded & finite_all
obs_rate = np.stack([obs[y] for y in YEARS])
print(f'   fixed coverage: {fixed.sum()} vertices, {area[fixed].sum() / 1e12:.3f} Mkm^2')

ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
lk = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), ds_b['basinNumber'].values.ravel())
bel = lk(np.column_stack([tri_x.mean(1), tri_y.mean(1)])).astype(float)
acc = np.zeros(nv2); cnt = np.zeros(nv2)
for c in range(3):
    np.add.at(acc, el2[:, c], bel)
    np.add.at(cnt, el2[:, c], 1)
basin = np.round(acc / np.maximum(cnt, 1)).astype(int)


def am(stk, m):
    w = area[m]
    return (stk[:, m] * w).sum(axis=1) / w.sum()


def block(label, m):
    gt = area[m].sum() * 917.0 / 1e12
    h, s, o = am(ho_rate, m), am(ssa_rate, m), am(obs_rate, m)
    hr, sr = h - am(ho_smb, m), s - am(ssa_smb, m)
    print(f'\n[{label}]  area {area[m].sum() / 1e12:.3f} Mkm^2   (x{gt:.0f} -> Gt/yr)')
    print(f'   {"year":>4} {"HO":>8} {"SSA":>8} {"obs":>8} | {"HO resid":>9} {"SSA resid":>9}')
    for y, a, b, c, d, e in zip(YEARS, h, s, o, hr, sr):
        print(f'   {y:4d} {a:+8.4f} {b:+8.4f} {c:+8.4f} | {d:+9.4f} {e:+9.4f}')
    print(f'   resid change 1996->2001: HO {hr[-1] - hr[0]:+.4f}  SSA {sr[-1] - sr[0]:+.4f} m/yr;  '
          f'1996-2001 mean Gt/yr: HO {h.mean() * gt:+.1f}  SSA {s.mean() * gt:+.1f}  obs {o.mean() * gt:+.1f}')
    return h, s, o, hr, sr


print('\n=========== FIXED-COVERAGE, AREA-WEIGHTED, 1996-2001 ===========')
dom = block('whole domain', fixed)
res = {}
for b in range(16):
    m = fixed & (basin == b)
    if m.sum() >= 50:
        res[b] = block(f'ISMIP7 basin {b}', m)

print('\n-- Plotting...', flush=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
h, s, o, hr, sr = dom
ax[0].plot(YEARS, h, 'o-', color='C3', label='HO'); ax[0].plot(YEARS, s, 'o-', color='C0', label='SSA')
ax[0].plot(YEARS, o, 's-', color='C1', label='obs (CPOM)'); ax[0].axhline(0, color='gray', lw=0.5)
ax[0].set_title('dH/dt, fixed coverage, area-weighted (m/yr)'); ax[0].legend(fontsize=8)
ax[1].plot(YEARS, hr, 'o-', color='C3', label='HO'); ax[1].plot(YEARS, sr, 'o-', color='C0', label='SSA')
ax[1].set_title('Dynamic residual dH/dt - SMB (m/yr)'); ax[1].legend(fontsize=8)
for a in ax:
    a.set_xlabel('Year')
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_chunk1_transient_domain.png', dpi=140)

n = len(res)
fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(17, 2.8 * np.ceil(n / 4)), sharex=True)
axes = np.atleast_1d(axes).ravel()
for a, (b, (h, s, o, hr, sr)) in zip(axes, res.items()):
    a.plot(YEARS, h, 'o-', ms=3, color='C3', label='HO')
    a.plot(YEARS, s, 'o-', ms=3, color='C0', label='SSA')
    a.plot(YEARS, o, 's-', ms=3, color='C1', label='obs')
    a.axhline(0, color='gray', lw=0.5)
    a.set_title(f'ISMIP7 basin {b}', fontsize=8)
    a.tick_params(labelsize=7)
for a in axes[n:]:
    a.axis('off')
axes[0].legend(fontsize=6)
fig.suptitle('1996-2001 dH/dt per basin (m/yr), fixed coverage, area-weighted', fontsize=10)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_chunk1_transient_basins.png', dpi=130)
print('Saved models/ho_chunk1_transient_domain.png and models/ho_chunk1_transient_basins.png')
print('DONE')
