"""Two figures, in the style of plot_projections.py (same validated colour slots, one y-axis per panel):

  docs/figures/projections_collapse.png   ISMIP7 ice-shelf collapse run (ssp585, CESM2-WACCM) vs the same
                                          scenario without collapse: (a) sea level minus control to 2300,
                                          (b) floating ice area, (c) map of the year each vertex was removed
  docs/figures/projections_ho_vs_ssa.png  HO vs SSA to 2100: (a) ssp585 minus control, (b) control drift

Sea level is from volume above flotation, ISMIP convention (as export_larmip.py / plot_projections.py),
mm, positive = rise. SSA series come from the cache models/proj_ssa_sle_series.csv (plot_projections.py
--proj); HO series are computed here from the HO base layer (the SSA mesh), each track from its own
2019 initial state. Run as a batch job (reads the HO and collapse history files).
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import plot_projections as pp  # style (rcParams), colour slots, end_labels

M = pp.MODEL_DIR
SLOTS, INK, INK2, GRID = pp.SLOTS, pp.INK, pp.INK2, pp.GRID
SSA_COLOR, ALT_COLOR = SLOTS[0], SLOTS[1]

nc = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
x = np.asarray(nc['mesh']['x'][:], dtype=float).ravel()
y = np.asarray(nc['mesh']['y'][:], dtype=float).ravel()
el = np.asarray(nc['mesh']['elements'][:]).astype(int) - 1
bed = np.asarray(nc['geometry']['bed'][:], dtype=float).ravel()
il0 = np.asarray(nc['mask']['ice_levelset'][:], dtype=float).ravel()
ol0 = np.asarray(nc['mask']['ocean_levelset'][:], dtype=float).ravel()
ri = float(nc['materials'].getncattr('rho_ice'))
rw = float(nc['materials'].getncattr('rho_water'))
nc.close()
nv = x.size
tx, ty = x[el], y[el]
ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(nv)
for c in range(3):
    np.add.at(area, el[:, c], ea / 3.0)


def vaf(H, ol):
    return np.sum(np.where(ol > 0, np.maximum(H + np.minimum(bed, 0.0) * rw / ri, 0.0), 0.0) * area)


def files(prefix):
    return [f for f in sorted(glob.glob(f'{M}/{prefix}_*_history.npz'))
            if re.fullmatch(rf'{re.escape(prefix)}_\d+-\d+_history\.npz', os.path.basename(f))]


def scalar(prefix, key):
    t, v = [], []
    for f in files(prefix):
        h = np.load(f)
        t += list(h['times']); v += list(h[key])
    t, v = np.array(t), np.array(v)
    keep = np.abs(t - np.round(t)) < 1e-6  # yearly outputs only (collapse sub-solves also save their first step)
    u, i = np.unique(np.round(t[keep]), return_index=True)
    return u.astype(int), v[keep][i]


def ho_sle(exp):
    """Yearly SLE (mm) of an HO run from its base layer, relative to the HO 2019 initial state."""
    d = netCDF4.Dataset(f'{M}/AIS3_proj_ho_relaxfix_init.nc')
    H0 = np.asarray(d['geometry']['thickness'][:nv], dtype=float).ravel()
    O0 = np.asarray(d['mask']['ocean_levelset'][:nv], dtype=float).ravel()
    d.close()
    v0 = vaf(H0, O0)
    yrs, s = [2019], [0.0]
    for f in files(f'AIS3_proj_ho_relaxfix_{exp}'):
        h = np.load(f)
        th, ol = h['thickness'], h['ocean_levelset']
        assert th.shape[1] == nv, 'HO history is not on the base layer'
        for k, t in enumerate(h['times']):
            if abs(t - round(t)) < 1e-6:
                yrs.append(int(round(t)))
                s.append(-(vaf(th[k].astype(float), ol[k].astype(float)) - v0) * ri / 1028.0 / 3.625e14 * 1e3)
        print(f'   {os.path.basename(f)} read', flush=True)
    yrs, s = np.array(yrs), np.array(s)
    u, i = np.unique(yrs, return_index=True)
    return u, s[i]


sle = pd.read_csv(pp.SLE_CACHE)
yr = sle['year'].values
assert 'ssp585_cesm2_ref_ref_collapse' in sle, 'rerun plot_projections.py --proj first (collapse run missing)'

# ---------------------------------------------------------------- collapse figure
ref = sle['ssp585_cesm2_ref_ref'].values - sle['ctrl'].values
col = sle['ssp585_cesm2_ref_ref_collapse'].values - sle['ctrl'].values
tf_r, fa_r = scalar('AIS3_proj_ssa_ssp585_cesm2_ref_ref', 'scalar_FloatingArea')
tf_c, fa_c = scalar('AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse', 'scalar_FloatingArea')
when = np.full(nv, np.nan)
for f in files('AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse'):
    h = np.load(f)
    if 'collapsed_at' in h.files:
        when = np.fmin(when, h['collapsed_at'])
print(f'collapse: {int(np.isfinite(when).sum())} vertices removed, years {np.nanmin(when):.0f}-{np.nanmax(when):.0f}', flush=True)

fig = plt.figure(figsize=(15.5, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05])
ax = fig.add_subplot(gs[0])
for v, lab, c in ((ref, 'no collapse', SSA_COLOR), (col, 'with ISMIP7 collapse mask', ALT_COLOR)):
    ok = np.isfinite(v)
    ax.plot(yr[ok], v[ok], color=c, label=lab)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlim(2019, 2300)
ax.set_title('a  SSP5-8.5 (CESM2-WACCM), minus control')
ax.set_ylabel('sea-level contribution (mm)')
ax.set_xlabel('year')
ax.legend(loc='upper left')
pp.end_labels(ax, [(ref[np.isfinite(ref)][-1], f'{ref[np.isfinite(ref)][-1]:+.0f}'),
                   (col[np.isfinite(col)][-1], f'{col[np.isfinite(col)][-1]:+.0f}')], 2300, dx=3)

ax = fig.add_subplot(gs[1])
ax.plot(tf_r, fa_r / 1e9, color=SSA_COLOR, label='no collapse')
ax.plot(tf_c, fa_c / 1e9, color=ALT_COLOR, label='with collapse')
ax.set_xlim(2019, 2300)
ax.set_ylim(0, None)
ax.set_title('b  Floating ice area')
ax.set_ylabel('thousand km$^2$')
ax.set_xlabel('year')
ax.legend(loc='lower left')

ax = fig.add_subplot(gs[2])
tri = mtri.Triangulation(x / 1e3, y / 1e3, el)
ax.tricontour(tri, il0, levels=[0], colors=INK2, linewidths=0.5)
ax.tricontour(tri, np.where(il0 < 0, ol0, -1.0), levels=[0], colors=GRID, linewidths=0.5)
m = np.isfinite(when)
sc = ax.scatter(x[m] / 1e3, y[m] / 1e3, c=when[m], s=0.4, cmap='Oranges', vmin=2019, vmax=2300, linewidths=0,
                rasterized=True)
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s_ in ax.spines.values():
    s_.set_visible(False)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
cb.set_label('year removed', color=INK2)
ax.set_title('c  Where and when shelves collapse')
ax.text(0.01, 0.01, 'grey line: 2019 ice front; light line: 2019 grounding line', transform=ax.transAxes,
        fontsize=8, color=INK2)
fig.suptitle('ACCESS-AIS3 (SSA): ISMIP7 ice-shelf collapse (fracture path C, mask v2.1)',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.tight_layout(rect=(0, 0, 0.97, 0.94))
out = f'{pp.FIG_DIR}/projections_collapse.png'
fig.savefig(out, dpi=200)
print('wrote', out, flush=True)

# ---------------------------------------------------------------- HO vs SSA figure
print('-- HO series...', flush=True)
yh, ho_ctrl = ho_sle('ctrl')
yh2, ho_585 = ho_sle('ssp585_cesm2_ref_ref')
assert np.array_equal(yh, yh2)
ssa_ctrl = sle['ctrl'].values
ssa_585 = sle['ssp585_cesm2_ref_ref'].values
sel = yr <= 2100

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axs[0]
ax.plot(yr[sel], (ssa_585 - ssa_ctrl)[sel], color=SSA_COLOR, label='SSA')
ax.plot(yh, ho_585 - ho_ctrl, color=ALT_COLOR, label='HO (Blatter-Pattyn)')
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlim(2019, 2100)
ax.set_title('a  SSP5-8.5 (CESM2-WACCM) minus control')
ax.set_ylabel('sea-level contribution (mm)')
pp.end_labels(ax, [((ssa_585 - ssa_ctrl)[yr == 2100][0], f'SSA {(ssa_585 - ssa_ctrl)[yr == 2100][0]:+.0f}'),
                   ((ho_585 - ho_ctrl)[-1], f'HO {(ho_585 - ho_ctrl)[-1]:+.0f}')], 2100, dx=1.5)
ax.legend(loc='lower left')
ax = axs[1]
ax.plot(yr[sel], ssa_ctrl[sel], color=SSA_COLOR, label='SSA')
ax.plot(yh, ho_ctrl, color=ALT_COLOR, label='HO (Blatter-Pattyn)')
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlim(2019, 2100)
ax.set_title('b  Control run drift (constant 1995-2014 climate)')
pp.end_labels(ax, [(ssa_ctrl[yr == 2100][0], f'SSA {ssa_ctrl[yr == 2100][0]:+.0f}'),
                   (ho_ctrl[-1], f'HO {ho_ctrl[-1]:+.0f}')], 2100, dx=1.5)
ax.legend(loc='lower left')
for a in axs:
    a.set_xlabel('year')
fig.suptitle('ACCESS-AIS3: higher-order (HO) vs SSA projections, each from its own 2019 state',
             x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.tight_layout(rect=(0, 0, 0.95, 0.93))
out = f'{pp.FIG_DIR}/projections_ho_vs_ssa.png'
fig.savefig(out, dpi=200)
print('wrote', out)
print('HO ssp585-ctrl 2050/2100:', (ho_585 - ho_ctrl)[yh == 2050][0], (ho_585 - ho_ctrl)[-1],
      '| SSA:', (ssa_585 - ssa_ctrl)[yr == 2050][0], (ssa_585 - ssa_ctrl)[yr == 2100][0])
print('collapse-ref 2100/2200/2300:', *[f'{(col - ref)[yr == Y][0]:+.1f}' for Y in (2100, 2200, 2300)])
print('DONE')
