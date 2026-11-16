"""Why is the SSA LARMIP response 4-8x below LARMIP-2 (worklog 6.14)?

For the Weddell (r1) and Amundsen (r4) 8 m/yr step runs against CTRL, at years 50, 100, 200:
  - shelf: thickness of the sector's initially floating ice, exp vs ctrl; fraction at <= 2 m
  - grounding line: grounded area lost in the sector (exp - ctrl)
  - grounding-line flux (domain scalar, exp - ctrl: only this sector is forced)
  - grounded speed-up in the sector by distance from the initial grounding line
  - friction: the PRESCRIBED effective pressure N (friction.coupling = 3, fixed in time) against
    the overburden-minus-water-pressure N that the thinned ice would have, near the grounding line
Read-only; prints a summary and writes models/larmip_response_check.png.
"""
import os
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import glob
import numpy as np
import netCDF4
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

M = 'models'
RHO_I, RHO_W, G = 917.0, 1023.0, 9.81
YEARS = [50, 100, 200]
T0 = 2019.0

d = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
x = np.asarray(d['mesh']['x'][:]).ravel(); y = np.asarray(d['mesh']['y'][:]).ravel()
el = np.asarray(d['mesh']['elements'][:]).astype(int) - 1
bed = np.asarray(d['geometry']['bed'][:]).ravel()
H0 = np.asarray(d['geometry']['thickness'][:]).ravel()
il = np.asarray(d['mask']['ice_levelset'][:]).ravel()
ol0 = np.asarray(d['mask']['ocean_levelset'][:]).ravel()
fc = np.asarray(d['friction']['coefficient'][:]).ravel()
Np = np.asarray(d['friction']['effective_pressure'][:]).ravel()
print('friction attrs:', {a: d['friction'].getncattr(a) for a in d['friction'].ncattrs()})
nv = x.size
tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(nv)
for c in range(3):
    np.add.at(area, el[:, c], ea / 3.0)

ds = xr.open_dataset('assets/larmip3/LARMIP3_sectors-1km.nc')
var = next(v for v in ds.data_vars if ds[v].ndim == 2)
mx, my = np.meshgrid(ds['x'].values, ds['y'].values)
mv = ds[var].values.astype(float)
ok = np.isfinite(mv)
sector = np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mv[ok])(x, y)).astype(int)

ice = il < 0
gr0 = ice & (ol0 > 0)
fl0 = ice & (ol0 < 0)
# initial grounding line: grounded vertices sharing an element with a floating one
glv = np.zeros(nv, bool)
mixed = el[np.any(gr0[el], axis=1) & np.any(fl0[el], axis=1)]
glv[mixed.ravel()] = True
glv &= gr0
dist_gl = cKDTree(np.column_stack([x[glv], y[glv]])).query(np.column_stack([x, y]))[0]


def load_at(prefix, year):
    t = T0 + year
    for f in sorted(glob.glob(f'{M}/{prefix}_*_history.npz')):
        if 'smoke' in f:
            continue
        h = np.load(f)
        times = h['times']
        i = np.where(np.abs(times - t) < 1e-6)[0]
        if i.size:
            i = int(i[0])
            return {k: h[k][i].astype(np.float64) for k in ('thickness', 'ocean_levelset', 'vel')}, \
                {k[7:]: float(h[k][i]) for k in h.files if k.startswith('scalar_')}
    raise FileNotFoundError(f'{prefix} t={t}')


ctrl = {yr: load_at('AIS3_proj_ssa_ctrl', yr) for yr in YEARS}
v0 = np.asarray(d['initialization']['vel'][:]).ravel()
res = {}
for r, name in [(1, 'Weddell'), (4, 'Amundsen')]:
    sec = sector == r
    print(f'\n=================== r{r} {name} ===================')
    print(f'   sector: initially floating ice {area[fl0 & sec].sum() / 1e6:,.0f} km2 (mean H {np.average(H0[fl0 & sec], weights=area[fl0 & sec]):.0f} m), '
          f'grounded {area[gr0 & sec].sum() / 1e6:,.0f} km2')
    gl_s = gr0 & sec & (dist_gl < 20e3)
    Nover0 = np.maximum(RHO_I * G * H0 - RHO_W * G * np.maximum(-bed, 0), 0)
    print(f'   friction near GL (<20 km, grounded): prescribed N median {np.median(Np[gl_s]) / 1e6:.3f} MPa; '
          f'overburden-flotation N at start median {np.median(Nover0[gl_s]) / 1e6:.3f} MPa; fc median {np.median(fc[gl_s]):.3g}')
    res[r] = {}
    for yr in YEARS:
        e, es = load_at(f'AIS3_proj_ssa_larmip_r{r}_step', yr)
        c, cs = ctrl[yr]
        fs = fl0 & sec
        He, Hc = e['thickness'], c['thickness']
        w = area[fs]
        dH_shelf = np.average(He[fs] - Hc[fs], weights=w)
        thin2 = np.sum(w[He[fs] <= 2.0]) / w.sum()
        gr_e = (e['ocean_levelset'] > 0) & ice & sec
        gr_c = (c['ocean_levelset'] > 0) & ice & sec
        dGA = (area[gr_c].sum() - area[gr_e].sum()) / 1e6
        dGLF = es['GroundinglineMassFlux'] - cs['GroundinglineMassFlux']
        dVAF = es['IceVolumeAboveFloatation'] - cs['IceVolumeAboveFloatation']
        line = (f'   y{yr:3d}: shelf dH(exp-ctrl) {dH_shelf:+7.1f} m (exp mean H {np.average(He[fs], weights=w):.0f}, ctrl {np.average(Hc[fs], weights=w):.0f}); '
                f'shelf area at <=2 m {100 * thin2:5.1f}%; grounded area lost {dGA:8,.0f} km2; '
                f'GL flux +{dGLF:7.1f} Gt/yr; VAF {-dVAF * RHO_I / 1028 / 3.625e14 * 1000:+6.1f} mm SLE')
        print(line)
        bands = [(0, 10e3), (10e3, 50e3), (50e3, 150e3), (150e3, 400e3)]
        s = []
        for a, b in bands:
            m = gr0 & sec & (e['ocean_levelset'] > 0) & (dist_gl >= a) & (dist_gl < b)
            ratio = np.average(e['vel'][m], weights=area[m]) / np.average(c['vel'][m], weights=area[m]) if m.any() else np.nan
            s.append(f'{a / 1e3:.0f}-{b / 1e3:.0f} km x{ratio:.2f}')
        print('          grounded speed exp/ctrl by distance from initial GL: ' + ', '.join(s))
        still = gr0 & sec & (e['ocean_levelset'] > 0) & (dist_gl < 20e3)
        Nover = np.maximum(RHO_I * G * He - RHO_W * G * np.maximum(-bed, 0), 0)
        print(f'          still-grounded <20 km from initial GL: H change exp-ctrl median {np.median(He[still] - Hc[still]):+.1f} m; '
              f'N if flotation-based: median {np.median(Nover[still]) / 1e6:.3f} MPa (start {np.median(Nover0[still]) / 1e6:.3f}) '
              f'vs prescribed {np.median(Np[still]) / 1e6:.3f} MPa (fixed); frac with flotation N < 0.5x prescribed '
              f'{np.mean(Nover[still] < 0.5 * Np[still]):.2f}')
        res[r][yr] = (He - Hc, e['vel'] / np.maximum(c['vel'], 1.0), sec)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for row, r in enumerate((1, 4)):
    dH, vr, sec = res[r][200]
    m = sec & ice
    for col, (v, ttl, cm, lim) in enumerate([(dH, 'H exp-ctrl at y200 (m)', 'RdBu', 400),
                                             (np.log2(vr), 'log2 speed exp/ctrl at y200', 'PuOr_r', 2)]):
        ax = axes[row, col]
        sc = ax.scatter(x[m] / 1e3, y[m] / 1e3, c=v[m], s=0.3, cmap=cm, vmin=-lim, vmax=lim)
        ax.scatter(x[glv & sec] / 1e3, y[glv & sec] / 1e3, c='k', s=0.2)
        ax.set_aspect('equal'); ax.set_title(f'r{r} {ttl}  (black: initial GL)', fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.04)
plt.tight_layout()
plt.savefig(f'{M}/larmip_response_check.png', dpi=110)
print('\nSaved models/larmip_response_check.png')
print('DONE')
