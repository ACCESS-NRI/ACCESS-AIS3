"""Locate the largest thickness changes in the D2 HO relaxation (worklog 5d: max|dH| 1155 m vs 868 m
with the old B) and compare with the old-B relaxation at the same places.

dH = relaxed end-state thickness (md.geometry after adopt_final_state) minus the relaxation input
(AIS3_melt_gamma_tuning.nc geometry). Base layer only (thickness is uniform down each column).
Read-only; writes a printout and models/ho_relaxed_d2_extremes.png.
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


def last_step_base(md, field, nvb):
    ts = md.results.TransientSolution
    nv = md.mesh.numberofvertices
    if isinstance(ts, list):
        a = np.asarray(getattr(ts[-1], field), float).ravel()
    else:
        raw = np.asarray(getattr(ts, field), float).ravel()
        a = raw.reshape(raw.size // nv, nv)[-1]
    return a


print('-- relaxation input (AIS3_melt_gamma_tuning.nc)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_melt_gamma_tuning.nc')
nv3 = md.mesh.numberofvertices
nvb = int(md.mesh.numberofvertices2d)
nl = nv3 // nvb
x = np.asarray(md.mesh.x).ravel()[:nvb].copy()
y = np.asarray(md.mesh.y).ravel()[:nvb].copy()
el2 = np.asarray(md.mesh.elements2d).astype(int) - 1
H0 = np.asarray(md.geometry.thickness, float).ravel()[:nvb].copy()
bed = np.asarray(md.geometry.bed, float).ravel()[:nvb].copy()
ol0 = np.asarray(md.mask.ocean_levelset, float).ravel()[:nvb].copy()
vel_obs = np.asarray(md.inversion.vel_obs, float).ravel()[-nvb:].copy()   # surface layer
del md
gc.collect()

res = {}
for tag, fname in [('D2', 'AIS3_ho_relaxed.nc'), ('oldB', 'AIS3_ho_relaxed_oldB.nc')]:
    print(f'-- {tag}: {fname}...', flush=True)
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/{fname}')
    assert md.mesh.numberofvertices == nv3
    H1 = np.asarray(md.geometry.thickness, float).ravel()[:nvb].copy()
    ol1 = np.asarray(md.mask.ocean_levelset, float).ravel()[:nvb].copy()
    vel = last_step_base(md, 'Vel', nvb)[-nvb:].copy()   # surface layer of the last step
    B = np.asarray(md.materials.rheology_B, float).ravel().reshape(nl, nvb).mean(axis=0)
    res[tag] = dict(dH=H1 - H0, H1=H1, ol1=ol1, vel=vel, Bmean=B)
    del md
    gc.collect()

ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
basin = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]),
                              ds_b['basinNumber'].values.ravel())(np.column_stack([x, y])).astype(int)
try:
    lat, lon = pyissm.tools.general.xy_to_ll(x, y, -1)
except Exception:
    lat = lon = None


def state(o):
    return np.where(o > 0, 'G', 'F')


d2, ob = res['D2'], res['oldB']
for tag in ('D2', 'oldB'):
    a = np.abs(res[tag]['dH'])
    print(f'\n[{tag}] mean|dH| {a.mean():.3f} m, max {a.max():.2f} m; vertices with |dH| > 100 m: '
          f'{(a > 100).sum()}, > 300 m: {(a > 300).sum()}, > 500 m: {(a > 500).sum()}')

print('\n=== 25 largest |dH| in the D2 relaxation (state G=grounded F=floating at start->end) ===')
print(f'{"x km":>8} {"y km":>8} {"lat":>7} {"lon":>7} {"bas":>3} {"H0":>7} {"dH_D2":>8} {"dH_oldB":>8} '
      f'{"st":>4} {"bed":>7} {"vel_D2":>7} {"vel_old":>7} {"v_obs":>7} {"Bratio":>6}')
idx = np.argsort(-np.abs(d2['dH']))[:25]
for i in idx:
    la = f'{lat[i]:7.2f}' if lat is not None else '    nan'
    lo = f'{lon[i]:7.2f}' if lon is not None else '    nan'
    print(f'{x[i] / 1e3:8.1f} {y[i] / 1e3:8.1f} {la} {lo} {basin[i]:3d} {H0[i]:7.1f} {d2["dH"][i]:+8.1f} '
          f'{ob["dH"][i]:+8.1f} {state(ol0[i])}->{state(d2["ol1"][i])} {bed[i]:7.0f} {d2["vel"][i]:7.0f} '
          f'{ob["vel"][i]:7.0f} {vel_obs[i]:7.0f} {d2["Bmean"][i] / ob["Bmean"][i]:6.3f}')

print('\n=== |dH| > 300 m by basin and start state (D2 | oldB) ===')
for tag in ('D2', 'oldB'):
    big = np.abs(res[tag]['dH']) > 300
    rows = []
    for b in np.unique(basin[big]):
        m = big & (basin == b)
        rows.append(f'b{b}: {m.sum()} (G {int((m & (ol0 > 0)).sum())}, F {int((m & (ol0 <= 0)).sum())})')
    print(f'   {tag}: ' + ', '.join(rows))

i0 = int(idx[0])
R = 150e3
win = (np.abs(x - x[i0]) < R) & (np.abs(y - y[i0]) < R)
print(f'\nMax-|dH| vertex: x={x[i0] / 1e3:.1f} km, y={y[i0] / 1e3:.1f} km, basin {basin[i0]}'
      + (f', lat {lat[i0]:.2f}, lon {lon[i0]:.2f}' if lat is not None else ''))
print(f'   window +-150 km: {win.sum()} vertices; |dH|>100 m: D2 {(np.abs(d2["dH"][win]) > 100).sum()}, '
      f'oldB {(np.abs(ob["dH"][win]) > 100).sum()}')

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
lim = max(np.nanpercentile(np.abs(d2['dH'][win]), 99), 50)
for ax, v, title, cmap, vmin, vmax in [
    (axes[0], d2['dH'], 'dH over 1-yr relaxation, D2 (m)', 'RdBu', -lim, lim),
    (axes[1], ob['dH'], 'dH over 1-yr relaxation, old B (m)', 'RdBu', -lim, lim),
    (axes[2], d2['Bmean'] / ob['Bmean'], 'column-mean B: D2 / old', 'PuOr', 0.5, 1.5),
]:
    sc = ax.scatter(x[win] / 1e3, y[win] / 1e3, c=v[win], s=3, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.tricontour(x / 1e3, y / 1e3, el2, ol0, levels=[0], colors='k', linewidths=0.6)
    ax.set_xlim((x[i0] - R) / 1e3, (x[i0] + R) / 1e3)
    ax.set_ylim((y[i0] - R) / 1e3, (y[i0] + R) / 1e3)
    ax.plot(x[i0] / 1e3, y[i0] / 1e3, 'kx', ms=10)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, fraction=0.045)
fig.suptitle(f'Around the D2 max-|dH| vertex (x) -- black line: starting grounding line; basin {basin[i0]}',
             fontsize=10)
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/ho_relaxed_d2_extremes.png', dpi=130)
print('Saved models/ho_relaxed_d2_extremes.png')
print('DONE')
