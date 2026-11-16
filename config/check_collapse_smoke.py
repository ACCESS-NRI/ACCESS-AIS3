"""Where did the levelset-collapse smoke test (worklog 6.12, inner job 180862912) go wrong?

Reads the partial outbin of sub-solve 2019.0 (3 saved steps before the NaN) against the
projection initial state, and the 2100 collapse mask that the smoke applied (SMOKE_SHIFT=81).
Read-only; prints a summary and writes models/collapse_smoke_check.png.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm

ROOT = '/g/data/au88/jh7060/ACCESS-AIS3'
RUN = os.environ.get('CHECK_RUN', 'AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2019-2020_smoke_sub2019.0_failed_nan_20261009')
OUTBIN = f'{ROOT}/execution_SSA/{RUN}/' + RUN.split('_failed')[0] + '.outbin'
MASK = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/CESM2-WACCM/ssp585/fracture/v2.1/'
        'ice_shelf_collapse_mask_cesm2waccm_ssp585_ismip7_8km-v2.1.nc')

md = pyissm.model.io.load_model(f'{ROOT}/models/AIS3_proj_ssa_init.nc')
x = np.asarray(md.mesh.x).ravel()
y = np.asarray(md.mesh.y).ravel()
el = np.asarray(md.mesh.elements).astype(int) - 1
H0 = np.asarray(md.geometry.thickness, float).ravel()
il0 = np.asarray(md.mask.ice_levelset, float).ravel()
ol0 = np.asarray(md.mask.ocean_levelset, float).ravel()
v0 = np.asarray(md.initialization.vel, float).ravel()
md.results.TransientSolution = []
md = pyissm.model.execute.load_results_from_disk(md, OUTBIN)
ts = md.results.TransientSolution
steps = list(ts.steps) if hasattr(ts, 'steps') else (ts if isinstance(ts, list) else [ts])
print(f'{len(steps)} saved steps: ' + ', '.join(f'{float(s.time):.2f}' for s in steps))

c = xr.open_dataset(MASK, decode_times=False)
gx, gy = c.x.values, c.y.values
ix = np.rint((x - gx[0]) / (gx[1] - gx[0])).astype(int)
iy = np.rint((y - gy[0]) / (gy[1] - gy[0])).astype(int)
on = c['mask'].sel(time=2100).values[iy, ix] == 1
removed = on & (il0 < 0) & (ol0 < 0)        # approx. (nearest cell, no iceberg pass)
keep = (il0 < 0) & ~removed
# vertices of kept ice that share an element with a removed vertex = new front
nf = np.zeros(x.size, bool)
e_mix = el[np.any(removed[el], axis=1) & np.any(keep[el], axis=1)]
nf[e_mix.ravel()] = True
nf &= keep
print(f'removed ~{removed.sum()} vertices; new-front kept vertices {nf.sum()} '
      f'(grounded {int((nf & (ol0 > 0)).sum())}, floating {int((nf & (ol0 < 0)).sum())})')
try:
    lat, lon = pyissm.tools.general.xy_to_ll(x, y, -1)
except Exception:
    lat = lon = np.full(x.size, np.nan)

for s in steps:
    vel = np.asarray(s.Vel, float).ravel()
    H = np.asarray(s.Thickness, float).ravel()
    ok = keep & np.isfinite(vel)
    top = np.argsort(-np.where(ok, vel, -1))[:8]
    print(f'\nt={float(s.time):.2f}: kept ice max vel {vel[ok].max():.3e} m/yr; >20 km/yr: {(vel[ok] > 2e4).sum()}, '
          f'>100 km/yr: {(vel[ok] > 1e5).sum()}; new-front median {np.median(vel[nf]):.0f}, p99 {np.percentile(vel[nf], 99):.0f}; '
          f'kept-ice H range {H[keep].min():.1f}..{H[keep].max():.1f}; |dH|max {np.abs(H - H0)[keep].max():.1f}')
    for i in top:
        print(f'   x {x[i] / 1e3:8.1f} y {y[i] / 1e3:8.1f} lat {lat[i]:7.2f} lon {lon[i]:7.2f} vel {vel[i]:.3e} v0 {v0[i]:7.0f} '
              f'H0 {H0[i]:6.0f} H {H[i]:7.1f} {"G" if ol0[i] > 0 else "F"} newfront {nf[i]}')

s = steps[-1]
vel = np.asarray(s.Vel, float).ravel()
i0 = int(np.argsort(-np.where(keep & np.isfinite(vel), vel, -1))[0])
R = 200e3
w = (np.abs(x - x[i0]) < R) & (np.abs(y - y[i0]) < R)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, v, t in [(ax[0], np.log10(np.maximum(v0, 1)), 'log10 vel start'),
                (ax[1], np.log10(np.maximum(vel, 1)), f'log10 vel t={float(s.time):.1f}')]:
    sc = a.scatter(x[w & keep] / 1e3, y[w & keep] / 1e3, c=v[w & keep], s=2, vmin=0, vmax=5)
    a.scatter(x[w & removed] / 1e3, y[w & removed] / 1e3, c='lightgrey', s=1)
    a.plot(x[i0] / 1e3, y[i0] / 1e3, 'rx')
    a.set_aspect('equal')
    a.set_title(t + ' (grey: removed)')
    plt.colorbar(sc, ax=a)
plt.tight_layout()
plt.savefig(f"{ROOT}/models/collapse_smoke_check{os.environ.get('CHECK_TAG', '')}.png", dpi=120)
print('saved models/collapse_smoke_check.png')
