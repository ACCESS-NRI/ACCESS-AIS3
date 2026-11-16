"""Map MIPKIT's 18 Mouginot/Rignot IMBIE drainage basins onto the 16 ISMIP7 ocean basins.

Cross-tabulates at MIPKIT's native 500 m grid: each 500 m cell gets its ISMIP7 8 km basin by
nearest-cell lookup, then areas are summed per (Mouginot, ISMIP7) pair, restricted to ice using
MIPKIT's own BedMachine mask. Grounded ice is the primary table (it is what the dH/dt
comparisons use); all ice (grounded + floating) is reported for context.

Outputs: printed tables, models/basin_crosstab_mouginot18_vs_ismip7ocean16.csv (grounded km^2),
models/basin_mapping_mouginot18_vs_ismip7ocean16.png.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
MIPKIT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc'
ISMIP7_BASINS = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/'
                 'imbie2/basin_numbers_ismip8km_v2.nc')

print('-- Loading MIPKIT mouginot_basins and mask_bedmachine (500 m)...', flush=True)
ds = xr.open_dataset(MIPKIT)
x5 = ds['x'].values
y5 = ds['y'].values
dx5 = abs(float(x5[1] - x5[0]))
dy5 = abs(float(y5[1] - y5[0]))
cell_km2 = dx5 * dy5 / 1e6
print(f'   grid {y5.size}x{x5.size}, cell {dx5:.0f} m x {dy5:.0f} m = {cell_km2:.3f} km^2; '
      f'y runs {y5[0]:.0f} -> {y5[-1]:.0f}')
mou = ds['mouginot_basins'].values
mask = ds['mask_bedmachine'].values
print(f'   mask_bedmachine attrs: {ds["mask_bedmachine"].attrs}')
mv, mc = np.unique(mask, return_counts=True)
print('   mask_bedmachine values/counts: ' + ', '.join(f'{int(v)}:{c}' for v, c in zip(mv, mc)))
# BedMachine convention: 0 ocean, 1 ice-free land, 2 grounded ice, 3 floating ice, 4 Lake Vostok.
# Verify against the attrs printed above before trusting these tables.
grounded = (mask == 2) | (mask == 4)
floating = mask == 3
ice = grounded | floating

print('-- Loading ISMIP7 ocean basins (8 km) and projecting onto the 500 m grid...', flush=True)
ds8 = xr.open_dataset(ISMIP7_BASINS)
b8 = ds8['basinNumber'].values
x8 = ds8['x'].values
y8 = ds8['y'].values
ix8 = np.clip(np.rint((x5 - x8[0]) / (x8[1] - x8[0])).astype(int), 0, x8.size - 1)
iy8 = np.clip(np.rint((y5 - y8[0]) / (y8[1] - y8[0])).astype(int), 0, y8.size - 1)
b8_on5 = b8[np.ix_(iy8, ix8)].astype(int)

mou_ids = np.unique(mou[ice])
mou_ids = mou_ids[mou_ids > 0]
ism_ids = np.arange(int(b8.max()) + 1)
print(f'   Mouginot basin IDs on ice: {mou_ids.tolist()}  ({mou_ids.size})')
print(f'   Mouginot value on ice outside 1..18 (unassigned): '
      f'{int(((mou[ice] < 1) | (mou[ice] > 18)).sum())} cells')


def crosstab(sel):
    m = mou[sel].astype(int)
    b = b8_on5[sel]
    code = m * 100 + b
    vals, counts = np.unique(code, return_counts=True)
    tab = pd.DataFrame(0.0, index=mou_ids, columns=ism_ids)
    for v, c in zip(vals, counts):
        mi, bi = v // 100, v % 100
        if mi in tab.index and bi in tab.columns:
            tab.loc[mi, bi] += c * cell_km2
    return tab


tab_g = crosstab(grounded)
tab_i = crosstab(ice)
out_csv = f'{MODEL_DIR}/basin_crosstab_mouginot18_vs_ismip7ocean16.csv'
tab_g.round(1).to_csv(out_csv)
print(f'   saved grounded cross-tab (km^2) to {out_csv}')


def fmt_parts(row, total, thresh=1.0):
    parts = [(int(k), 100 * v / total) for k, v in row.items() if total > 0 and 100 * v / total >= thresh]
    parts.sort(key=lambda p: -p[1])
    return ', '.join(f'{k}:{p:.1f}%' for k, p in parts)


print('\n=== Each Mouginot basin -> ISMIP7 ocean basins (GROUNDED ice, shares >= 1%) ===')
print(f'{"mou":>4} {"grounded km2":>13} {"dominant":>9} {"share":>7}   split')
clean = 0
for m in tab_g.index:
    row = tab_g.loc[m]
    tot = row.sum()
    dom = int(row.idxmax())
    share = 100 * row.max() / tot if tot > 0 else np.nan
    clean += share >= 95
    print(f'{m:4d} {tot:13.0f} {dom:9d} {share:6.1f}%   {fmt_parts(row, tot)}')
print(f'   Mouginot basins with >=95% of grounded area in ONE ISMIP7 basin: {clean} of {len(tab_g)}')

print('\n=== Each ISMIP7 ocean basin <- Mouginot basins (GROUNDED ice, shares >= 1%) ===')
print(f'{"ism":>4} {"grounded km2":>13}   composition')
for b in tab_g.columns:
    col = tab_g[b]
    tot = col.sum()
    print(f'{b:4d} {tot:13.0f}   {fmt_parts(col, tot)}')

print('\n=== All ice (grounded + floating): dominant ISMIP7 basin per Mouginot basin ===')
for m in tab_i.index:
    row = tab_i.loc[m]
    tot = row.sum()
    print(f'{m:4d} all-ice {tot:10.0f} km2 -> {int(row.idxmax())} ({100 * row.max() / tot:.1f}%)')

print('\n-- Plotting (subsampled 8x for display)...', flush=True)
s = 8
mou_s = mou[::s, ::s].astype(float)
b8_s = b8_on5[::s, ::s].astype(float)
ice_s = ice[::s, ::s]
mou_s[~ice_s] = np.nan
b8_s[~ice_s] = np.nan
xs, ys = x5[::s] / 1e3, y5[::s] / 1e3
ext = [xs.min(), xs.max(), ys.min(), ys.max()]
origin = 'lower' if y5[1] > y5[0] else 'upper'

fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
cmap = plt.get_cmap('tab20')
for ax, arr, title, ids in [
    (axes[0], mou_s, 'MIPKIT Mouginot/Rignot drainage basins (18), ice only', mou_ids),
    (axes[1], b8_s, 'ISMIP7 ocean basins (16), ice only', ism_ids),
]:
    ax.imshow(np.where(np.isfinite(arr), arr % 20, np.nan), extent=ext, origin=origin,
              cmap=cmap, vmin=0, vmax=19, interpolation='nearest')
    X, Y = np.meshgrid(xs, ys)
    for i in ids:
        m = arr == i
        if m.sum() > 20:
            ax.text(X[m].mean(), Y[m].mean(), str(int(i)), ha='center', va='center',
                    fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

ax = axes[2]
ax.imshow(np.where(np.isfinite(b8_s), b8_s % 20, np.nan), extent=ext, origin=origin,
          cmap=cmap, vmin=0, vmax=19, interpolation='nearest', alpha=0.55)
mou_fill = np.where(np.isfinite(mou_s), mou_s, -1)
edge = np.zeros_like(mou_fill, dtype=bool)
edge[:, 1:] |= mou_fill[:, 1:] != mou_fill[:, :-1]
edge[1:, :] |= mou_fill[1:, :] != mou_fill[:-1, :]
edge &= np.isfinite(mou_s)
ax.imshow(np.where(edge, 1.0, np.nan), extent=ext, origin=origin, cmap='Greys', vmin=0, vmax=1,
          interpolation='nearest')
ax.set_title('ISMIP7 16 (colour) with Mouginot 18 boundaries (black)', fontsize=10)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out_png = f'{MODEL_DIR}/basin_mapping_mouginot18_vs_ismip7ocean16.png'
plt.savefig(out_png, dpi=130)
print(f'Saved {out_png}')
print('DONE')
