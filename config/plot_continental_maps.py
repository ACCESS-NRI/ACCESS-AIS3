"""Antarctic maps of ice-thickness change for the SSA projection campaign and LARMIP.

  docs/figures/maps_larmip_dH.png       LARMIP step runs minus control at year 200 (2219), one panel per
                                        sector, with that sector's outline and the run's grounding line
  docs/figures/maps_projections_dH.png  2100 and 2300: control drift (vs 2019), and SSP runs (CESM2-WACCM)
                                        minus control at the same year, with grounding lines

Thickness and grounded/floating mask come from the per-chunk history files (annual, SSA mesh).
Colour: diverging blue <-> red with a neutral grey midpoint (red = thinning) on a symmetric log scale
(linear within +-5 m, logarithmic to +-1000 m), so metre-scale grounded change and km-scale shelf loss both show.
Run as a batch job (reads ~20 history files, renders 3.2M triangles per panel).
"""
import os
import re
import glob
import numpy as np
import netCDF4
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

ROOT = '/g/data/au88/jh7060/ACCESS-AIS3'
MODEL_DIR = f'{ROOT}/models'
FIG_DIR = f'{ROOT}/docs/figures'
LARMIP_MASK = f'{ROOT}/assets/larmip3/LARMIP3_sectors-1km.nc'
SURFACE, INK, INK2, OCEAN = '#fcfcfb', '#0b0b0b', '#52514e', '#e8eef5'
# Diverging: blue (thickening) <-> neutral grey <-> red (thinning); blue steps from the reference
# palette's blue ramp, red arm mirrored in step count.
DIV = LinearSegmentedColormap.from_list(
    'dH', ['#7a1414', '#c03030', '#e86a5f', '#f5b8ae', '#f0efec', '#9ec5f4', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'text.color': INK,
                     'axes.titlelocation': 'left'})

nc = netCDF4.Dataset(f'{MODEL_DIR}/AIS3_proj_ssa_init.nc')
x = np.asarray(nc['mesh']['x'][:], dtype=float).ravel()
y = np.asarray(nc['mesh']['y'][:], dtype=float).ravel()
el = np.asarray(nc['mesh']['elements'][:]).astype(int) - 1
H0 = np.asarray(nc['geometry']['thickness'][:], dtype=float).ravel()
il0 = np.asarray(nc['mask']['ice_levelset'][:], dtype=float).ravel()
ol0 = np.asarray(nc['mask']['ocean_levelset'][:], dtype=float).ravel()
nc.close()
ice_el = (il0[el] < 0).all(axis=1)          # draw only elements inside the (fixed) ice front
tri = mtri.Triangulation(x / 1e3, y / 1e3, el, mask=~ice_el)
tri_all = mtri.Triangulation(x / 1e3, y / 1e3, el)


def state(exp, year):
    """(thickness, ocean_levelset) at an integer year from the experiment's history files."""
    if year == 2019:
        return H0, ol0
    for f in glob.glob(f'{MODEL_DIR}/AIS3_proj_ssa_{exp}_*_history.npz'):
        m = re.fullmatch(rf'AIS3_proj_ssa_{re.escape(exp)}_(\d+)-(\d+)_history\.npz', os.path.basename(f))
        if not m or not (int(m.group(1)) < year <= int(m.group(2))):
            continue
        d = np.load(f)
        t = d['times']
        k = int(np.argmin(np.abs(t - year)))
        assert abs(t[k] - year) < 1e-6, (exp, year, t[k])
        return d['thickness'][k].astype(float), d['ocean_levelset'][k].astype(float)
    raise FileNotFoundError(f'{exp} {year}')


def panel(ax, dH, ol, title, norm, ol_ref=None):
    ax.set_facecolor(OCEAN)
    pc = ax.tripcolor(tri, dH, cmap=DIV, norm=norm, shading='gouraud', rasterized=True)
    ax.tricontour(tri_all, ol, levels=[0], colors=INK, linewidths=0.35)
    if ol_ref is not None:
        ax.tricontour(tri_all, ol_ref, levels=[0], colors=INK2, linewidths=0.35, linestyles=':')
    ax.set_aspect('equal')
    ax.set_xlim(-2750, 2800)
    ax.set_ylim(-2300, 2350)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, fontweight='bold')
    return pc


def limit(fields):
    v = np.concatenate([np.abs(f[np.isfinite(f) & (il0 < 0)]) for f in fields])
    lim = np.percentile(v, 99.5)
    for nice in (5, 10, 20, 25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000):
        if lim <= nice:
            return nice
    return float(np.ceil(lim / 500) * 500)


NORM = SymLogNorm(linthresh=5, linscale=0.6, vmin=-1000, vmax=1000, base=10)
TICKS = [-1000, -100, -10, 0, 10, 100, 1000]


def colorbar(fig, pc, rect):
    cb = fig.colorbar(pc, cax=fig.add_axes(rect), orientation='horizontal', extend='both', ticks=TICKS)
    cb.ax.set_xticklabels([f'{t:+d}' if t else '0' for t in TICKS])
    cb.set_label('ice-thickness change (m, symmetric log scale); red = thinning, blue = thickening', color=INK)
    cb.ax.tick_params(colors=INK2)


os.makedirs(FIG_DIR, exist_ok=True)

# -- LARMIP ------------------------------------------------------------------------------------------
names = {1: 'Weddell', 2: 'East Antarctica', 3: 'Ross', 4: 'Amundsen', 5: 'Peninsula'}
Hc, olc = state('ctrl', 2219)
fields = {}
for r in names:
    H, ol = state(f'larmip_r{r}_step', 2219)
    fields[r] = (H - Hc, ol)
lim = limit([f[0] for f in fields.values()])
norm = NORM
ds = xr.open_dataset(LARMIP_MASK)
sx, sy = np.meshgrid(ds['x'].values[::4], ds['y'].values[::4])
sector = ds['z'].values[::4, ::4]
fig, axs = plt.subplots(1, 5, figsize=(20, 4.9))
fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.2, wspace=0.03)
for ax, r in zip(axs, names):
    dH, ol = fields[r]
    pc = panel(ax, dH, ol, f'{r}  {names[r]}', norm, ol_ref=olc)
    ax.contour(sx / 1e3, sy / 1e3, (sector == r).astype(float), levels=[0.5], colors=INK2, linewidths=0.8,
               linestyles='--')
colorbar(fig, pc, [0.25, 0.1, 0.5, 0.035])
fig.suptitle('LARMIP-3, 8 m/yr extra melt in one sector: thickness change at year 200 (2219), minus control.  '
             'Solid line: grounding line of the run; dotted: control; dashed: sector outline',
             x=0.005, y=0.985, ha='left', fontsize=11, fontweight='bold')
fig.savefig(f'{FIG_DIR}/maps_larmip_dH.png', dpi=170, bbox_inches='tight')
print('wrote maps_larmip_dH.png, colour limit', lim, flush=True)

# -- projections ---------------------------------------------------------------------------------------
rows = []
for yr, exps in ((2100, ['ssp126_cesm2_ref_ref', 'ssp370_cesm2_ref_ref', 'ssp585_cesm2_ref_ref', 'ssp585_cesm2_high_ref']),
                 (2300, ['ssp126_cesm2_ref_ref', 'ssp585_cesm2_ref_ref', 'ssp585_cesm2_high_ref', 'ssp585_cesm2_ref_smbonly'])):
    Hc, olc = state('ctrl', yr)
    row = [(f'Control drift {yr} − 2019', Hc - H0, olc, ol0)]
    for e in exps:
        H, ol = state(e, yr)
        label = e.replace('_cesm2', '').replace('_ref_ref', '').replace('_high_ref', ' high melt') \
                 .replace('_ref_smbonly', ' SMB only').upper().replace('SSP', 'SSP').replace(' HIGH MELT', ', high melt') \
                 .replace(' SMB ONLY', ', SMB only')
        row.append((f'{label} {yr}, minus control', H - Hc, ol, olc))
    rows.append(row)
lim = limit([f[1] for row in rows for f in row])
norm = NORM
fig, axs = plt.subplots(2, 5, figsize=(20, 9.2))
fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.11, wspace=0.03, hspace=0.1)
for i, row in enumerate(rows):
    for j, (title, dH, ol, olr) in enumerate(row):
        pc = panel(axs[i, j], dH, ol, title, norm, ol_ref=olr)
colorbar(fig, pc, [0.25, 0.05, 0.5, 0.018])
fig.suptitle('ACCESS-ISSM (AIS3, SSA), CESM2-WACCM forcing: ice-thickness change.  Control panels: change since 2019; '
             'others: minus control in the same year.  Solid: grounding line; dotted: reference',
             x=0.005, y=0.985, ha='left', fontsize=11, fontweight='bold')
fig.savefig(f'{FIG_DIR}/maps_projections_dH.png', dpi=150, bbox_inches='tight')
print('wrote maps_projections_dH.png, colour limit', lim)
print('DONE')
