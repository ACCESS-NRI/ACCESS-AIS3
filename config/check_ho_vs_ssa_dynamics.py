"""Why does HO lose ~2x more grounded ice than SSA from the same starting geometry (worklog 4x)?

Read-only diagnosis from existing output, at the start of the historical run (first saved step,
t=1995.1) and over its first year, over grounded ice, whole domain and per ISMIP7 basin:
  * speed: observed (md.inversion.vel_obs of the SSA model), SSA, HO surface / base / depth-average
    -- does HO flow faster, and is it sliding (base) or internal deformation (surface - base)?
  * rheology: HO depth-averaged rheology_B vs SSA rheology_B (flow ~ B^-n, n=3: 10% softer B gives
    ~37% faster deformation)
  * first-year dynamic loss (dH/dt - SMB) HO vs SSA, mapped.
HO = chunk-1 rerun (AIS3_historical_1995_2019_chunk1rerun.nc, full 0.1 yr history); SSA =
AIS3_historical_1995_2019_SSA.nc. Same 2D mesh (HO base layer == SSA mesh, checked in 4w/4x).
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


def step(ts, nv, field, idx):
    if isinstance(ts, list):
        return np.asarray(getattr(ts[idx], field), dtype=float).ravel().copy()
    raw = np.asarray(getattr(ts, field), dtype=float).ravel()
    return raw.reshape(raw.size // nv, nv)[idx].copy()


def smb_at(md, n_keep, year):
    mb = np.asarray(md.smb.mass_balance)
    j = int(np.where(mb[-1, :].astype(int) == year)[0][0])
    return mb[:n_keep, j].astype(float)


# ---------------- HO ----------------
print('-- HO: loading AIS3_historical_1995_2019_chunk1rerun.nc...', flush=True)
mdh = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_chunk1rerun.nc')
nv3 = mdh.mesh.numberofvertices
nvb = int(mdh.mesh.numberofvertices2d)
nl = nv3 // nvb
t0h, dth = float(mdh.timestepping.start_time), float(mdh.timestepping.time_step)
i1996 = int(round((1996.0 - t0h) / dth)) - 1   # step ending at t=1996.0
vel3 = step(mdh.results.TransientSolution, nv3, 'Vel', 0).reshape(nl, nvb)
H1996_h = step(mdh.results.TransientSolution, nv3, 'Thickness', i1996)[:nvb]
H0_h = np.asarray(mdh.geometry.thickness, dtype=float).ravel()[:nvb].copy()
z = np.asarray(mdh.mesh.z, dtype=float).ravel().reshape(nl, nvb)
span = z[-1] - z[0]
sig = np.where(np.abs(span)[None, :] > 1e-9, (z - z[0]) / np.where(np.abs(span) > 1e-9, span, 1.0)[None, :],
               np.linspace(0, 1, nl)[:, None])
# trapezoidal depth average in sigma (relative layer spacing), robust to stale absolute z
w = np.diff(sig, axis=0)
vel_davg_h = (0.5 * (vel3[1:] + vel3[:-1]) * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)
vel_surf_h, vel_base_h = vel3[-1].copy(), vel3[0].copy()
B3 = np.asarray(mdh.materials.rheology_B, dtype=float).ravel()
B_davg_h = ((0.5 * (B3.reshape(nl, nvb)[1:] + B3.reshape(nl, nvb)[:-1]) * w).sum(axis=0)
            / np.maximum(w.sum(axis=0), 1e-12)) if B3.size == nv3 else np.full(nvb, np.nan)
print(f'   HO: {nl} layers; rheology_B size {B3.size}; friction class {type(mdh.friction).__name__}; '
      f'z-vs-geometry base mismatch max {np.abs(z[0] - np.asarray(mdh.geometry.base).ravel()[:nvb]).max():.2f} m')
dyn_h = (H1996_h - H0_h) - 0.5 * (smb_at(mdh, nvb, 1995) + smb_at(mdh, nvb, 1996))
del mdh, vel3, B3, z
gc.collect()

# ---------------- SSA ----------------
print('-- SSA: loading AIS3_historical_1995_2019_SSA.nc...', flush=True)
mds = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv2 = mds.mesh.numberofvertices
assert nv2 == nvb
x = np.asarray(mds.mesh.x).ravel()
y = np.asarray(mds.mesh.y).ravel()
el = np.asarray(mds.mesh.elements).astype(int) - 1
t0s, dts = float(mds.timestepping.start_time), float(mds.timestepping.time_step)
j1996 = int(round((1996.0 - t0s) / dts)) - 1
vel_s = step(mds.results.TransientSolution, nv2, 'Vel', 0)
H1996_s = step(mds.results.TransientSolution, nv2, 'Thickness', j1996)
H0_s = np.asarray(mds.geometry.thickness, dtype=float).ravel().copy()
B_s = np.asarray(mds.materials.rheology_B, dtype=float).ravel()
vel_obs = np.asarray(mds.inversion.vel_obs, dtype=float).ravel()
il = np.asarray(mds.mask.ice_levelset).ravel()
ol = np.asarray(mds.mask.ocean_levelset).ravel()
dyn_s = (H1996_s - H0_s) - 0.5 * (smb_at(mds, nv2, 1995) + smb_at(mds, nv2, 1996))
print(f'   SSA friction class {type(mds.friction).__name__}; rheology_B size {B_s.size}; '
      f'1995 thickness HO-SSA max|d| {np.abs(H0_h - H0_s).max():.4f} m')
del mds
gc.collect()

tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(nv2)
for c in range(3):
    np.add.at(area, el[:, c], ea / 3.0)
ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
lk = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), ds_b['basinNumber'].values.ravel())
bel = lk(np.column_stack([tx.mean(1), ty.mean(1)])).astype(float)
acc = np.zeros(nv2); cnt = np.zeros(nv2)
for c in range(3):
    np.add.at(acc, el[:, c], bel)
    np.add.at(cnt, el[:, c], 1)
basin = np.round(acc / np.maximum(cnt, 1)).astype(int)

gr = (il < 0) & (ol > 0) & np.isfinite(vel_obs) & (vel_obs > 0)
print(f'   grounded with observed speed: {gr.sum()} vertices, {area[gr].sum() / 1e12:.3f} Mkm^2')


def am(v, m):
    return float((v[m] * area[m]).sum() / area[m].sum())


def med_ratio(a, b, m):
    ok = m & (b > 1.0)   # ratios only where the denominator speed is > 1 m/yr
    return float(np.median(a[ok] / b[ok])) if ok.any() else np.nan


def row(label, m):
    gt = area[m].sum() * 917.0 / 1e12
    print(f'{label:>7} {area[m].sum() / 1e12:6.3f} | {am(vel_obs, m):7.1f} {am(vel_s, m):7.1f} '
          f'{am(vel_davg_h, m):7.1f} {am(vel_surf_h, m):7.1f} {am(vel_base_h, m):7.1f} | '
          f'{med_ratio(vel_davg_h, vel_s, m):6.2f} {med_ratio(vel_surf_h, vel_obs, m):6.2f} '
          f'{med_ratio(vel_s, vel_obs, m):6.2f} {med_ratio(vel_base_h, vel_surf_h, m):6.2f} | '
          f'{np.nanmedian(B_davg_h[m] / B_s[m]):6.3f} | {am(dyn_h, m) * gt:+8.1f} {am(dyn_s, m) * gt:+8.1f}')


print('\n=== Grounded ice (with observed speed), area-weighted means; ratios are medians over vertices ===')
print(f'{"":>7} {"Mkm2":>6} | {"obs":>7} {"SSA":>7} {"HOdavg":>7} {"HOsurf":>7} {"HObase":>7} | '
      f'{"HOd/SSA":>6} {"HOs/obs":>6} {"SSA/obs":>6} {"HOb/HOs":>6} | {"B_HO/SSA":>6} | '
      f'{"HOdyn":>8} {"SSAdyn":>8}')
print(' (speeds m/yr at t=1995.1; dyn = first-year dH/dt - SMB integrated, Gt/yr)')
row('all', gr)
for b in range(16):
    m = gr & (basin == b)
    if m.sum() >= 50:
        row(f'b{b}', m)

print('\n-- Plotting maps...', flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
sel = gr


def mapit(ax, v, title, cmap, vmin, vmax):
    sc = ax.scatter(x[sel] / 1e3, y[sel] / 1e3, c=v[sel], s=0.3, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=10); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.04)


d = dyn_h - dyn_s
lim = np.nanpercentile(np.abs(d[sel]), 98)
mapit(axes[0], d, 'HO - SSA first-year dynamic dH/dt (m/yr)\n(negative = HO exports more ice)', 'RdBu', -lim, lim)
ratio = np.log10(np.maximum(vel_davg_h, 1e-3) / np.maximum(vel_s, 1e-3))
mapit(axes[1], ratio, 'log10(HO depth-avg speed / SSA speed) at 1995.1', 'PuOr_r', -0.5, 0.5)
mapit(axes[2], B_davg_h / B_s, 'HO depth-avg B / SSA B (<1 = HO softer)', 'PuOr', 0.7, 1.3)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_vs_ssa_dynamics_maps.png', dpi=130)
print('Saved models/ho_vs_ssa_dynamics_maps.png')
print('DONE')
