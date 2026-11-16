"""Figures for the SSA projection campaign and the LARMIP submission.

  python plot_projections.py --larmip   -> docs/figures/larmip_responses.png (reads the LARMIP CSV)
  python plot_projections.py --proj     -> docs/figures/projections_ssa.png (computes sea level from the
                                           history files; run as a batch job, ~300 GB of reads; caches
                                           models/proj_ssa_sle_series.csv)

Sea level is from volume above flotation (ISMIP convention, as export_larmip.py), mm, positive =
rise. Colours: validated categorical slots in fixed order (dataviz skill); labels in text ink.
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/g/data/au88/jh7060/ACCESS-AIS3'
MODEL_DIR = f'{ROOT}/models'
FIG_DIR = f'{ROOT}/docs/figures'
LARMIP_CSV = f'{ROOT}/docs/larmip/AIS3_ssa_larmip_sle.csv'
SLE_CACHE = f'{MODEL_DIR}/proj_ssa_sle_series.csv'

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']  # validated (light): adjacent pass
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 1.6,
    'lines.solid_capstyle': 'round', 'legend.frameon': False, 'legend.fontsize': 9,
})


def end_labels(ax, items, x, min_gap_frac=0.045, dx=3):
    """Label line ends (text ink), nudged apart vertically so they never overlap.
    items: list of (y_end, text)."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    ys = [items[i][0] for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for k, i in enumerate(order):
        ax.annotate(items[i][1], (x, items[i][0]), xytext=(x + dx, ys[k]), textcoords='data',
                    va='center', ha='left', fontsize=8.5, color=INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=GRID, lw=0.6) if abs(ys[k] - items[i][0]) > gap * 0.3 else None)


def larmip_figure():
    d = pd.read_csv(LARMIP_CSV, comment='#')
    yr = d['year'].values
    ctrl = d['ctrl'].values * 1e3
    names = {1: 'Weddell', 2: 'East Antarctica', 3: 'Ross', 4: 'Amundsen', 5: 'Peninsula'}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={'width_ratios': [1, 1.25, 1.25]})

    ax = axs[0]
    ax.plot(yr, ctrl, color=INK2)
    ax.set_title('a  Control')
    ax.set_xlabel('model year (0 = 2019)')
    ax.set_ylabel('sea-level contribution (mm)')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(0, 200)
    ax.annotate(f'{ctrl[-1]:.0f} mm', (yr[-1], ctrl[-1]), xytext=(4, 0), textcoords='offset points',
                ha='left', va='center', fontsize=8.5, color=INK2, annotation_clip=False)

    resp = {k: {r: (d[f'larmip_r{r}_{k}'].values * 1e3 - ctrl) for r in names} for k in ('step', 'ramp')}
    top = max(v.max() for k in resp for v in resp[k].values())
    for ax, k, title in ((axs[1], 'step', 'b  8 m/yr extra melt from year 0, minus control'),
                         (axs[2], 'ramp', 'c  Extra melt ramped 0 to 8 m/yr, minus control')):
        for r in names:
            ax.plot(yr, resp[k][r], color=SLOTS[r - 1], label=f'{r} {names[r]}')
        ax.set_ylim(-0.03 * top, 1.08 * top)
        ax.set_xlim(0, 200)
        ax.set_title(title)
        ax.set_xlabel('model year (0 = 2019)')
        ax.axhline(0, color=INK2, lw=0.8)
        end_labels(ax, [(resp[k][r][-1], f'{names[r]} {resp[k][r][-1]:.0f}') for r in names], 200)
    axs[1].set_ylabel('sea-level contribution (mm)')
    axs[2].axvspan(0, 100, color='#f0efec', zorder=0, lw=0)
    axs[2].text(50, 1.02 * top, 'melt ramps up', ha='center', va='top', fontsize=8.5, color=INK2)
    axs[2].text(150, 1.02 * top, 'held at 8 m/yr', ha='center', va='top', fontsize=8.5, color=INK2)
    axs[1].legend(loc='upper left', title='LARMIP-3 sector', title_fontsize=9)
    fig.suptitle('ACCESS-ISSM (AIS3, SSA) LARMIP-3 experiments: sea-level contribution from volume above flotation',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    out = f'{FIG_DIR}/larmip_responses.png'
    fig.savefig(out, dpi=200)
    print('wrote', out)


def sle_series():
    """Yearly sea level (mm) per experiment, 2019..2300, from the history files (cached)."""
    import netCDF4
    nc = netCDF4.Dataset(f'{MODEL_DIR}/AIS3_proj_ssa_init.nc')
    x = np.asarray(nc['mesh']['x'][:], dtype=float).ravel()
    y = np.asarray(nc['mesh']['y'][:], dtype=float).ravel()
    elx = np.asarray(nc['mesh']['elements'][:]).astype(int) - 1
    bed = np.asarray(nc['geometry']['bed'][:], dtype=float).ravel()
    H0 = np.asarray(nc['geometry']['thickness'][:], dtype=float).ravel()
    ol0 = np.asarray(nc['mask']['ocean_levelset'][:], dtype=float).ravel()
    ri = float(nc['materials'].getncattr('rho_ice'))
    rw = float(nc['materials'].getncattr('rho_water'))
    nc.close()
    tx, ty = x[elx], y[elx]
    ea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
    area = np.zeros(x.size)
    for c in range(3):
        np.add.at(area, elx[:, c], ea / 3.0)

    def vaf(H, ol):
        return np.sum(np.where(ol > 0, np.maximum(H + np.minimum(bed, 0.0) * rw / ri, 0.0), 0.0) * area)

    v0 = vaf(H0, ol0)
    exps = sorted({re.sub(r'_\d+-\d+_history\.npz$', '', os.path.basename(f))[len('AIS3_proj_ssa_'):]
                   for f in glob.glob(f'{MODEL_DIR}/AIS3_proj_ssa_*_history.npz')
                   if '_smoke' not in f and '_larmip_' not in f and re.search(r'_\d+-\d+_history\.npz$', f)})
    years = np.arange(2019, 2301)
    out = pd.DataFrame({'year': years})
    for exp in exps:
        s = np.full(years.size, np.nan)
        s[0] = 0.0
        for f in sorted(glob.glob(f'{MODEL_DIR}/AIS3_proj_ssa_{exp}_*_history.npz')):
            if not re.fullmatch(rf'AIS3_proj_ssa_{re.escape(exp)}_\d+-\d+_history\.npz', os.path.basename(f)):
                continue
            d = dict(np.load(f))
            for k, t in enumerate(d['times']):
                if abs(t - round(t)) < 1e-6 and 2020 <= round(t) <= 2300:
                    s[int(round(t)) - 2019] = -(vaf(d['thickness'][k].astype(float), d['ocean_levelset'][k].astype(float))
                                               - v0) * ri / 1028.0 / 3.625e14 * 1e3
        out[exp] = s
        print(f'{exp}: last year {years[np.isfinite(s)][-1]}', flush=True)
    out.to_csv(SLE_CACHE, index=False)
    print('wrote', SLE_CACHE)
    return out


def projection_figure(s):
    yr = s['year'].values
    c = s['ctrl'].values

    def rel(e):
        return s[e].values - c if e in s else None

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.35, 1.1, 1]})

    # a: scenario (colour) x melt (line style), CESM2-WACCM
    ax = axs[0]
    labels = []
    for i, scen in enumerate(('ssp126', 'ssp370', 'ssp585')):
        for melt, ls in (('ref', '-'), ('high', '--')):
            e = f'{scen}_cesm2_{melt}_ref'
            v = rel(e)
            if v is None or not np.isfinite(v).any():
                continue
            ok = np.isfinite(v)
            ax.plot(yr[ok], v[ok], color=SLOTS[i], ls=ls,
                    label=f"{scen.upper()}{'' if melt == 'ref' else ', high melt'}")
            labels.append((v[ok][-1], yr[ok][-1], f"{scen}{' high' if melt == 'high' else ''} {v[ok][-1]:+.0f}"))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(2019, 2300)
    ax.set_title('a  CESM2-WACCM: scenario and melt sensitivity')
    ax.set_ylabel('sea-level contribution minus control (mm)')
    ax.legend(loc='lower left', fontsize=8.5)
    for yend, xend, txt in labels:
        ax.annotate(txt, (xend, yend), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8,
                    color=INK2, annotation_clip=False)

    # b: attribution, ssp585 CESM2
    ax = axs[1]
    comb, ocn, smb = rel('ssp585_cesm2_ref_ref'), rel('ssp585_cesm2_ref_oceanonly'), rel('ssp585_cesm2_ref_smbonly')
    for v, lab, col, ls in ((comb, 'ocean + SMB forcing', INK, '-'), (ocn, 'ocean forcing only', SLOTS[0], '-'),
                            (smb, 'SMB forcing only', SLOTS[1], '-')):
        if v is not None:
            ok = np.isfinite(v)
            ax.plot(yr[ok], v[ok], color=col, ls=ls, label=lab)
    if ocn is not None and smb is not None:
        ok = np.isfinite(ocn + smb)
        ax.plot(yr[ok], (ocn + smb)[ok], color=INK2, ls=':', lw=1.4, label='sum of single-forcing runs')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(2019, 2300)
    ax.set_title('b  SSP5-8.5 (CESM2-WACCM): attribution')
    ax.legend(loc='lower left', fontsize=8.5)

    # c: GCM spread at SSP3-7.0, to 2100
    ax = axs[2]
    gcms = [('cesm2', 'CESM2-WACCM'), ('mri', 'MRI-ESM2-0'), ('access', 'ACCESS-CM2'), ('canesm5', 'CanESM5'),
            ('gfdl', 'GFDL-ESM4'), ('ipsl', 'IPSL-CM6A-LR'), ('mpi', 'MPI-ESM1-2-HR')]
    items = []
    for tag, name in gcms:
        v = rel(f'ssp370_{tag}_ref_ref')
        if v is None:
            continue
        ok = np.isfinite(v) & (yr <= 2100)
        if not ok.any():
            continue
        ax.plot(yr[ok], v[ok], color=INK2, lw=1.2)
        if yr[ok][-1] == 2100:
            items.append((v[ok][-1], f'{name} {v[ok][-1]:+.0f}'))
        else:
            items.append((v[ok][-1], f'{name} (to {yr[ok][-1]})'))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(2019, 2100)
    ax.set_title('c  SSP3-7.0: seven GCMs (reference melt)')
    end_labels(ax, [(yv, t) for yv, t in items], 2100, dx=2)
    for a in axs:
        a.set_xlabel('year')
    fig.suptitle('ACCESS-ISSM (AIS3, SSA) projections: sea-level contribution relative to the constant-climate control',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.94, 0.95))
    out = f'{FIG_DIR}/projections_ssa.png'
    fig.savefig(out, dpi=200)
    print('wrote', out)


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    if '--larmip' in sys.argv:
        larmip_figure()
    if '--proj' in sys.argv:
        projection_figure(sle_series())
    if '--proj-cached' in sys.argv:
        projection_figure(pd.read_csv(SLE_CACHE))
