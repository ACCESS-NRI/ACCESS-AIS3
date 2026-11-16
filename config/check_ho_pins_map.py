"""Where does the HO model hold velocity fixed (worklog 6.20)? The HO projection start
(AIS3_proj_ho_relaxfix_init.nc) carries spcvx/spcvy = observed velocity on ~176k columns; SSA has none.

Pins split into three groups:
  boundary  ice vertices on the mesh boundary (ais_0.1.py ~568)
  nonice    ice_levelset >= 0 (Schoof third wave)
  onice     everything else, i.e. the thin-ice and C-at-lower-bound waves on grounded or floating ice
Per ISMIP basin and LARMIP-3 sector: grounded area, pinned grounded area, pinned area within 50 km
of the grounding line, and the pinned share of the obs-speed x H flux within 50 km of the GL.
Read-only (base layer only); prints tables, writes models/ho_pins_map.png and models/ho_pins_by_basin.csv.
"""
import os
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pandas as pd
import netCDF4
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

M = 'models'
BASIN_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/'
              'imbie2/basin_numbers_ismip8km_v2.nc')

s = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
x = np.asarray(s['mesh']['x'][:], float).ravel()
y = np.asarray(s['mesh']['y'][:], float).ravel()
el = np.asarray(s['mesh']['elements'][:]).astype(int) - 1
s.close()
nvb = x.size
tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(nvb)
for c in range(3):
    np.add.at(area, el[:, c], ea / 3.0)

d = netCDF4.Dataset(f'{M}/AIS3_proj_ho_relaxfix_init.nc')
x3 = np.asarray(d['mesh']['x'][:nvb], float).ravel()
assert np.allclose(x3, x), 'HO base layer != SSA mesh'
spx = np.asarray(d['stressbalance']['spcvx'][:nvb], float).ravel()
spx_top = np.asarray(d['stressbalance']['spcvx'][-nvb:], float).ravel()
il = np.asarray(d['mask']['ice_levelset'][:nvb], float).ravel()
ol = np.asarray(d['mask']['ocean_levelset'][:nvb], float).ravel()
H = np.asarray(d['geometry']['thickness'][:nvb], float).ravel()
onb = np.asarray(d['mesh']['vertexonboundary'][:nvb]).ravel().astype(bool)
vobs = np.asarray(d['inversion']['vel_obs'][:nvb], float).ravel()
d.close()
assert np.array_equal(np.isfinite(spx), np.isfinite(spx_top)), 'pins differ between base and surface layer'

ice = il < 0
gr = ice & (ol > 0)
pin = np.isfinite(spx)
grp = np.where(~pin, '', np.where(ice & onb, 'boundary', np.where(~ice, 'nonice', 'onice')))
for g in ('boundary', 'nonice', 'onice'):
    m = grp == g
    print(f'{g:9s}: {int(m.sum()):7d} columns; grounded-ice area {area[m & gr].sum() / 1e6:10,.0f} km2, '
          f'floating {area[m & ice & ~gr].sum() / 1e6:8,.0f} km2, median H {np.median(H[m]) if m.any() else np.nan:.0f} m')

fl = ice & (ol < 0)
glv = np.zeros(nvb, bool)
glv[el[np.any(gr[el], axis=1) & np.any(fl[el], axis=1)].ravel()] = True
glv &= gr
dist_gl = cKDTree(np.column_stack([x[glv], y[glv]])).query(np.column_stack([x, y]))[0]
near = gr & (dist_gl < 50e3)


def sample(path, var=None):
    ds = xr.open_dataset(path)
    v = var or next(k for k in ds.data_vars if ds[k].ndim == 2)
    mx, my = np.meshgrid(ds['x'].values, ds['y'].values)
    mv = ds[v].values.astype(float)
    ok = np.isfinite(mv) & (mv > 0)
    return np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mv[ok])(x, y)).astype(int)


rows = []
for label, ids in (('ismip', sample(BASIN_FILE)), ('larmip', sample('assets/larmip3/LARMIP3_sectors-1km.nc'))):
    for b in np.unique(ids[gr]):
        m = ids == b
        flux = vobs * H * area ** 0.5
        rows.append(dict(set=label, basin=int(b),
                         grounded_km2=area[gr & m].sum() / 1e6,
                         pinned_grounded_pct=100 * area[gr & m & (grp == 'onice')].sum() / area[gr & m].sum(),
                         pinned_near_gl_pct=100 * area[near & m & (grp == 'onice')].sum() / max(area[near & m].sum(), 1),
                         pinned_flux_near_gl_pct=100 * flux[near & m & (grp == 'onice')].sum() / max(flux[near & m].sum(), 1),
                         pinned_shelf_pct=100 * area[fl & m & pin].sum() / max(area[fl & m].sum(), 1)))
df = pd.DataFrame(rows)
pd.set_option('display.width', 160)
print(df.round(1).to_string(index=False))
df.to_csv(f'{M}/ho_pins_by_basin.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x[ice][::7] / 1e3, y[ice][::7] / 1e3, s=0.1, c='0.85', rasterized=True)
ax.scatter(x[glv] / 1e3, y[glv] / 1e3, s=0.1, c='0.4', rasterized=True)
for g, c in (('onice', '#d95f02'), ('nonice', '#1b9e77'), ('boundary', '#7570b3')):
    m = grp == g
    ax.scatter(x[m] / 1e3, y[m] / 1e3, s=0.3, c=c, label=f'{g} ({int(m.sum())})', rasterized=True)
ax.set_aspect('equal'); ax.legend(markerscale=20, loc='lower left')
ax.set_title('HO velocity pins (spcvx = observed), grey line = 2019 grounding line')
plt.tight_layout(); plt.savefig(f'{M}/ho_pins_map.png', dpi=130)
print(f'wrote {M}/ho_pins_map.png, {M}/ho_pins_by_basin.csv')
print('DONE')
