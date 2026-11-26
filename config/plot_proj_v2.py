"""Figures for the v2 projection set (docs/projection_experiments.md, worklog 6.35), from whatever has run so far.

Sea level is ISSM IceVolumeAboveFloatation as mm SLE (+ = rise), each run minus its own ctrl at matching times
(series from proj_v2_sle_table.py). October = the 2026-10 set (gamma_0 300, linear sliding, fixed N): grey reference.
Writes docs/figures/projections_v2_sensitivity.png and projections_v2_scenarios.png (with a date stamp in the
title; earlier versions are kept as *_<date>.png if a file of that name already exists).

usage: python plot_proj_v2.py      (login node; reads only scalar series)
"""
import datetime
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proj_v2_sle_table as T  # noqa: E402

OUT = '/g/data/au88/jh7060/ACCESS-AIS3/docs/figures'
TODAY = datetime.date.today().isoformat()
# Reference categorical palette (dataviz skill), fixed slot order; validated light: all checks pass, contrast WARN on
# aqua/yellow/magenta -> every line is direct-labelled at its end.
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300',
         violet='#6250d6', red='#e34948')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
V2 = '_coupledN_p3_hm4000'
OCT = dict(color=INK2, ls=(0, (5, 3)), lw=1.4)   # October reference


def rel(exp, ctrl):
    """(years, mm SLE of exp minus ctrl) on exp's output times; None if either has no history."""
    a, b = T.series(exp), T.series(ctrl)
    if a is None or b is None:
        return None
    (ta, va), (tb, vb) = a, b
    keep = ta <= tb[-1] + 1e-6
    if keep.sum() < 2:
        return None
    sa = -(va - va[0]) * T.K
    sb = -(vb - vb[0]) * T.K
    return ta[keep], sa[keep] - np.interp(ta[keep], tb, sb)


def absolute(exp):
    s = T.series(exp)
    return None if s is None else (s[0], -(s[1] - s[1][0]) * T.K)


def panel(ax, lines, title, ylabel=True, legend_loc='upper left'):
    """lines: list of (label, series or None, style dict). Draws, direct-labels line ends, adds the legend."""
    ends = []
    for label, s, st in lines:
        if s is None:
            continue
        t, y = s
        ax.plot(t, y, label=label, solid_capstyle='round', **{'lw': 2.0, **st})
        ends.append([y[-1], t[-1], label, st.get('color', INK)])
    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    ax.set_xlim(2019, 2300)
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=6)
    if ylabel:
        ax.set_ylabel('mm SLE, minus own control', color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    # direct labels at line ends, nudged apart vertically (text in ink, a short colour tick for identity)
    if ends:
        lo, hi = ax.get_ylim()
        gap = 0.055 * (hi - lo)
        ends.sort()
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for (y0, t0, label, col), yl in zip(ends, ys):
            ax.annotate(label, xy=(t0, y0), xytext=(t0 + 4, yl), fontsize=7.5, color=INK, va='center',
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', color=col, lw=1.2, shrinkA=0, shrinkB=2))
    ax.legend(loc=legend_loc, fontsize=7.5, frameon=False, labelcolor=INK, handlelength=2.4)


def save(fig, name):
    path = f'{OUT}/{name}.png'
    if os.path.exists(path):
        made = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
        old = f'{OUT}/{name}_{made}.png'
        if made != TODAY and not os.path.exists(old):  # keep one copy per earlier day; same-day reruns overwrite
            os.rename(path, old)
    fig.savefig(path, dpi=160, facecolor=SURF)
    print('saved', path)


plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.facecolor': SURF})

# ---- Figure 1: configuration sensitivity -------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 8.6), facecolor=SURF)
plt.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.07, wspace=0.32, hspace=0.32)
panel(axs[0, 0], [
    ('October', rel('ssp585_cesm2_ref_ref', 'ctrl'), OCT),
    ('v2', rel('ssp585_cesm2_ref_ref' + V2, 'ctrl' + V2), dict(color=C['blue'])),
    ('v2 + SMB feedback', rel('ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000', 'ctrl_coupledN_p3_smbfb-cesm2_hm4000'),
     dict(color=C['orange'])),
    ('v2, gamma_0 8000', rel('ssp585_cesm2_ref_ref_coupledN_p3_hm8000', 'ctrl_coupledN_p3_hm8000'), dict(color=C['aqua'])),
    ('v2 + SEM1', rel('ssp585_cesm2_ref_ref_coupledN_p3_hm4000sem1', 'ctrl_coupledN_p3_hm4000sem1'),
     dict(color=C['yellow'])),
    ('v2 + SEM1 + feedback', rel('ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000sem1',
                                 'ctrl_coupledN_p3_smbfb-cesm2_hm4000sem1'), dict(color=C['magenta'])),
    ('v2 drift-corrected', rel('ssp585_cesm2_ref_ref_coupledN_p3_driftcorr_hm4000', 'ctrl_coupledN_p3_driftcorr_hm4000'),
     dict(color=INK, ls=(0, (1, 1.5)), lw=1.6)),
], 'a  ssp585 CESM2-WACCM: physics and melt choices')
panel(axs[0, 1], [
    ('October', rel('ssp585_cesm2_ref_ref', 'ctrl'), OCT),
    ('October + collapse', rel('ssp585_cesm2_ref_ref_collapse', 'ctrl'), dict(color=INK2, ls=(0, (1, 1.5)), lw=1.6)),
    ('v2', rel('ssp585_cesm2_ref_ref' + V2, 'ctrl' + V2), dict(color=C['blue'])),
    ('v2 + collapse', rel('ssp585_cesm2_ref_ref_collapse' + V2, 'ctrl' + V2), dict(color=C['violet'])),
], 'b  ssp585 CESM2-WACCM: ISMIP7 ice-shelf collapse', ylabel=False)
panel(axs[1, 0], [
    ('October', rel('ssp585_mri_ref_ref', 'ctrl'), OCT),
    ('v2', rel('ssp585_mri_ref_ref' + V2, 'ctrl' + V2), dict(color=C['blue'])),
    ('v2 + SMB feedback', rel('ssp585_mri_ref_ref_coupledN_p3_smbfb_hm4000', 'ctrl_coupledN_p3_smbfb-mri_hm4000'),
     dict(color=C['orange'])),
    ('v2 ocean only', rel('ssp585_mri_ref_oceanonly' + V2, 'ctrl' + V2), dict(color=C['green'])),
    ('v2 SMB only', rel('ssp585_mri_ref_smbonly' + V2, 'ctrl' + V2), dict(color=C['red'])),
], 'c  ssp585 MRI-ESM2-0: attribution')
panel(axs[1, 1], [
    ('October', absolute('ctrl'), OCT),
    ('v2', absolute('ctrl' + V2), dict(color=C['blue'])),
    ('v2 + SMB feedback (CESM2)', absolute('ctrl_coupledN_p3_smbfb-cesm2_hm4000'), dict(color=C['orange'])),
    ('v2, gamma_0 8000', absolute('ctrl_coupledN_p3_hm8000'), dict(color=C['aqua'])),
    ('v2 + SEM1', absolute('ctrl_coupledN_p3_hm4000sem1'), dict(color=C['yellow'])),
    ('v2 drift-corrected', absolute('ctrl_coupledN_p3_driftcorr_hm4000'), dict(color=INK, ls=(0, (1, 1.5)), lw=1.6)),
], 'd  Control runs: drift since 2019 (mm SLE)', ylabel=False, legend_loc='lower left')
fig.suptitle(f'ACCESS-AIS3 SSA projections, v2 set (gamma_0 4000 melt + coupledN_p3 sliding) vs October   '
             f'[runs in progress, {TODAY}]', x=0.06, ha='left', fontsize=12.5, color=INK)
fig.text(0.06, 0.935, 'Ice volume above flotation as sea-level equivalent; + = rise. Lines end where a run (or its '
         'control) currently ends.', fontsize=9, color=INK2)
save(fig, 'projections_v2_sensitivity')

# ---- Figure 2: scenarios under v2 ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURF)
plt.subplots_adjust(left=0.06, right=0.88, top=0.84, bottom=0.11, wspace=0.32)
for k, (gcm, tag) in enumerate((('CESM2-WACCM', 'cesm2'), ('MRI-ESM2-0', 'mri'))):
    panel(axs[k], [
        ('October ssp585', rel(f'ssp585_{tag}_ref_ref', 'ctrl'), OCT),
        ('October ssp126', rel(f'ssp126_{tag}_ref_ref', 'ctrl'), dict(color=INK2, ls=(0, (1, 1.5)), lw=1.6)),
        ('v2 ssp126', rel(f'ssp126_{tag}_ref_ref' + V2, 'ctrl' + V2), dict(color=C['blue'])),
        ('v2 ssp534-over', rel(f'ssp534-over_{tag}_ref_ref' + V2, 'ctrl' + V2), dict(color=C['orange'])),
        ('v2 ssp585', rel(f'ssp585_{tag}_ref_ref' + V2, 'ctrl' + V2), dict(color=C['aqua'])),
    ], f"{'ab'[k]}  {gcm}", ylabel=(k == 0))
fig.suptitle(f'ACCESS-AIS3 SSA projections: scenarios under v2   [runs in progress, {TODAY}]', x=0.06, ha='left',
             fontsize=12.5, color=INK)
fig.text(0.06, 0.905, 'Each run minus its own control; October (grey) for reference. ssp534-over: overshoot, '
         'emissions peak ~2040 then decline.', fontsize=9, color=INK2)
save(fig, 'projections_v2_scenarios')
