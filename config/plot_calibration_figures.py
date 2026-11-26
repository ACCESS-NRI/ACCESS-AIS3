"""Figures for worklog 6.29-6.35: melt calibration, SSA friction laws, pin-free HO historical.

Reads the saved score files (no model runs):
  models/melt_history_match.json, models/melt_rescore_real_targets.json        -> docs/figures/melt_calibration_hm.png
  models/ssa_historical_friction_scores.json                                   -> docs/figures/ssa_historical_friction.png
  models/ho_pinfree_historical_scores.json + the HO history files               -> docs/figures/ho_pinfree_historical.png
Palette: categorical slots 1-4 of the dataviz reference (validated light: CVD worst adjacent dE 9.1; slots 3-4
below 3:1 contrast, so every series is also direct-labelled). Observations are drawn in ink, not a series hue.
"""
import os
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import glob
import json
import numpy as np
import netCDF4
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M, FIG = 'models', 'docs/figures'
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#9a9893', '#e4e3de'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.labelcolor': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'axes.titlelocation': 'left',
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'font.size': 9.5, 'lines.linewidth': 2,
})


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', va='center', fontsize=8.5, color=INK2)
    ax.plot([x], [y], 'o', ms=4, color=color, zorder=5)


# ---------------------------------------------------------------- 1. melt calibration
hm = json.load(open(f'{M}/melt_history_match.json'))
rs = json.load(open(f'{M}/melt_rescore_real_targets.json'))
g = np.array(hm['gamma'])
cap = hm['runs']['capped']
I = {k: np.array([r['I']['0.3'][k] for r in cap]) for k in ('J1', 'J2', 'J3', 'J4')}
nroy = hm['summary']['capped_delta0.3']['nroy']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1.15, 1]})
a1.axvspan(min(nroy), max(nroy), color=GRID, alpha=0.6, lw=0, zorder=0)
a1.axhline(3, color=INK2, lw=1, ls=(0, (4, 3)))
a1.text(g[-1], 3.15, 'rule-out threshold', ha='right', va='bottom', fontsize=8, color=INK2)
names = {'J1': 'J1 basin melt', 'J2': 'J2 buttressing bins', 'J3': 'J3 warm-cold change', 'J4': 'J4 Amundsen years'}
for (k, c), dy in zip((('J1', S1), ('J2', S2), ('J3', S3), ('J4', S4)), (8, -8, 0, 0)):
    a1.plot(g, I[k], color=c, label=names[k])
    j = int(np.argmin(np.abs(g - 2.0e4))) if k == 'J3' else len(g) - 1
    label_end(a1, g[j], I[k][j], names[k], c, dy)
for gv, txt in ((300, 'old: 300'), (4000, 'adopted: 4,000')):
    a1.axvline(gv, color=INK, lw=1)
    a1.text(gv, 9.6, txt, rotation=90, ha='right', va='top', fontsize=8.5, color=INK)
a1.set_xscale('log'); a1.set_ylim(0, 10); a1.set_xlim(250, 3.2e5)
a1.set_xlabel('gamma_0 (m/yr)'); a1.set_ylabel('misfit / uncertainty (RMS)')
a1.set_title('Misfit per term vs gamma_0\n(deltaT capped at +-2 C, 30% structural error)')
a1.text(np.sqrt(min(nroy) * max(nroy)), 4.3, 'not ruled out', ha='center', fontsize=8.5, color=INK2)
a1.legend(loc='upper right', fontsize=8)

bins = np.arange(10)
obs, prox = np.array(rs['bin_obs']), np.array(rs['bin_proxy'])
ref = hm['runs']['refit']
r300 = ref[int(np.argmin(np.abs(g - 300)))]
r4k = ref[int(np.argmin(np.abs(g - 4000)))]
w = 0.26
a2.bar(bins - w, prox, w * 0.92, color=MUTED, label='old J2 target (area-proportional proxy)')
a2.bar(bins, r300['bin'], w * 0.92, color=S2, label='model, gamma_0 = 300')
a2.bar(bins + w, r4k['bin'], w * 0.92, color=S1, label=f"model, gamma_0 = {r4k['gamma_0']:,.0f}")
obs_err = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/meltobs/'
                          'melt_target_term2_v3.nc')['melt_mean_err'].values
a2.errorbar(bins, obs, yerr=obs_err, fmt='D', ms=6, color=INK, ecolor=INK2, elinewidth=1, capsize=2, label='ISMIP7 target (observed)', zorder=6)
a2.set_xticks(bins); a2.set_xlabel('buttressing bin (0 = passive ... 9 = strongest)')
a2.set_ylabel('melt (Gt/yr)'); a2.set_title('Melt per buttressing bin\n(real target vs the old proxy)')
a2.legend(loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig(f'{FIG}/melt_calibration_hm.png', dpi=150)
plt.close(fig)
print('saved melt_calibration_hm.png')

# ---------------------------------------------------------------- 2. SSA friction laws
fr = json.load(open(f'{M}/ssa_historical_friction_scores.json'))
runs = [('hm4000', 'p=1, fixed N (production law)', S1), ('hm4000_p1_flotN', 'p=1, flotation N', S2),
        ('hm4000_p3_fixedN', 'p=3, fixed N', S3), ('hm4000_p3_flotN', 'p=3, flotation N (LARMIP main)', S4)]
short = {'hm4000': 'p=1 fixed N', 'hm4000_p1_flotN': 'p=1 flot. N', 'hm4000_p3_fixedN': 'p=3 fixed N',
         'hm4000_p3_flotN': 'p=3 flot. N'}
periods = ['1996-2001', '2002-2013', '2014-2019']
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={'width_ratios': [1, 1, 0.8]})
for ax, reg, title in ((axs[0], 'domain', 'Whole domain (fixed-coverage grounded mask)'), (axs[1], '9', 'Amundsen (basin 9)')):
    xo = np.arange(3)
    ax.plot(xo, [fr['obs'][reg]['periods'][p] for p in periods], 'D-', color=INK, ms=7, lw=1.5, label='CPOM (observed)', zorder=6)
    for n, (k, lab, c) in enumerate(runs):
        if k in fr:
            ax.plot(xo + (n - 1.5) * 0.06, [fr[k][reg]['periods'][p] for p in periods], 'o-', color=c, ms=6, lw=1.5, label=lab)
    if 'prod300' in fr:
        ax.plot(xo, [fr['prod300'][reg]['periods'][p] for p in periods], ':', color=MUTED, lw=1.5, label='old melt (gamma_0 300), p=1 fixed N')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xo); ax.set_xticklabels(periods); ax.set_ylabel('grounded mass change (Gt/yr)'); ax.set_title(title)
axs[0].legend(loc='lower right', fontsize=7.5)
ax = axs[2]
yv = np.arange(len(runs))
ax.axvline(fr[runs[0][0]]['speed']['2007']['b9']['obs_ratio_fw'], color=INK, lw=1.5)
ax.text(fr[runs[0][0]]['speed']['2007']['b9']['obs_ratio_fw'], -0.4, 'observed +18% ', fontsize=8.5, color=INK, va='center', ha='right')
for n, (k, lab, c) in enumerate(runs):
    if k in fr:
        v = fr[k]['speed']['2007']['b9']['mod_ratio_fw']
        ax.plot([1, v], [n, n], color=c, lw=2)
        ax.plot([v], [n], 'o', color=c, ms=8)
        ax.text(1.0, n + 0.2, f'{short[k]}: {100 * (v - 1):+.1f}%', fontsize=8, color=INK2, va='bottom')
ax.axvline(1, color=INK2, lw=0.8)
ax.set_yticks([]); ax.set_xlim(0.96, 1.22); ax.set_ylim(-0.5, len(runs) - 0.1)
ax.set_xlabel('speed 2019 / 2007, fast grounded ice (flux-weighted)'); ax.set_title('Amundsen speed-up')
fig.suptitle('SSA historical 1995-2019 with four friction laws (new melt, constant climatology ocean)', x=0.01, ha='left', fontsize=11.5, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIG}/ssa_historical_friction.png', dpi=150)
plt.close(fig)
print('saved ssa_historical_friction.png')

# ---------------------------------------------------------------- 3. pin-free HO
ho = json.load(open(f'{M}/ho_pinfree_historical_scores.json'))
P, F = 'HO pinned (relaxfix)', 'HO pin-free (budd_nopins)'
fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.45)
for n, (reg, title) in enumerate((('domain', 'Whole domain'), ('9', 'Amundsen'), ('14', 'Filchner-Ronne'), ('7', 'Ross'))):
    ax = fig.add_subplot(gs[0, n])
    xo = np.arange(3)
    ax.plot(xo, [ho['CPOM'][reg]['periods'][p] for p in periods], 'D-', color=INK, ms=7, lw=1.5, label='CPOM', zorder=6)
    ax.plot(xo - 0.05, [ho[P][reg]['periods'][p] for p in periods], 'o-', color=S2, ms=6, lw=1.5, label='pinned')
    ax.plot(xo + 0.05, [ho[F][reg]['periods'][p] for p in periods], 'o-', color=S1, ms=6, lw=1.5, label='pin-free')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xo); ax.set_xticklabels(['96-01', '02-13', '14-19'])
    ax.set_title(f"{title}\nyearly RMSE {ho[P][reg]['rmse_yearly']:.4f} -> {ho[F][reg]['rmse_yearly']:.4f} m/yr", fontsize=9.5)
    if n == 0:
        ax.set_ylabel('grounded mass change (Gt/yr)')
        ax.legend(loc='lower left', fontsize=8)
d = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
x = np.asarray(d['mesh'].variables['x'][:]) / 1e3
y = np.asarray(d['mesh'].variables['y'][:]) / 1e3


def series(tag, k):
    out = {}
    for f in sorted(glob.glob(f'{M}/AIS3_historical_1995_2019_{tag}_history_*.npz')):
        h = np.load(f)
        if '1995-2001' in f:
            out[1995.0] = float(h['initial_thickness'][k])
        for i, t in enumerate(h['times']):
            out[float(t)] = float(h['thickness'][i][k])
    t = np.array(sorted(out))
    return t, np.array([out[v] for v in t])


sites = (((-2328, 1051), 'Peninsula (-2328, 1051) km'), ((-1179, -1049), '(-1179, -1049) km'),
         ((-443, -532), 'Ross (-443, -532) km'), ((-608, 304), 'Weddell (-608, 304) km'))
for n, ((X, Y), title) in enumerate(sites):
    ax = fig.add_subplot(gs[1, n])
    k = int(np.argmin((x - X) ** 2 + (y - Y) ** 2))
    for tag, c, lab in (('relaxfix', S2, 'pinned'), ('budd_nopins', S1, 'pin-free')):
        t, h = series(tag, k)
        ax.plot(t, h, color=c, label=lab)
        ax.text(t[-1], h[-1], f' {lab}', fontsize=8, color=INK2, va='center')
    ax.set_title(title, fontsize=9.5); ax.set_xlim(1995, 2023.5)
    if n == 0:
        ax.set_ylabel('ice thickness (m)')
        ax.legend(loc='upper left', fontsize=8)
fig.suptitle('HO historical 1995-2019: pinned (Schoof + pins) vs pin-free (SSA Budd friction) vs CPOM; '
             'bottom: thickness at the former spike sites', x=0.01, ha='left', fontsize=11.5, fontweight='bold')
fig.savefig(f'{FIG}/ho_pinfree_historical.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved ho_pinfree_historical.png')
