"""LARMIP-3 analysis: our two sets against LARMIP-2, and a linear-response-theory (LRT) test.

Inputs
  docs/larmip/AIS3_ssa_larmip_sle.csv              standard set (Budd p=q=1, prescribed N)
  docs/larmip/AIS3_ssa_larmip_sle_coupledN_p3.csv  N coupled to flotation, Budd p=3 (drag-preserving rescale)
  assets/larmip2/<model>_BM08_R<n>.dat              LARMIP-2 8 m/yr step responses (m SLE, yearly),
                                                    downloaded from pik-potsdam.de/~anders/larmip/Experiments
Our responses are experiment minus control (ISMIP-convention VAF), mm SLE.

Region mapping (assumed, as in worklog 6.14): LARMIP-2 R1 East Antarctica, R2 Ross, R3 Amundsen,
R4 Weddell, R5 Peninsula; LARMIP-3 sectors 1 Weddell, 2 East Antarctica, 3 Ross, 4 Amundsen,
5 Peninsula. LARMIP-3 sector masks are not the LARMIP-2 regions, so the comparison is indicative.

LRT: the ramp (0 -> 8 m/yr linearly over 100 yr, then held) is a sum of 0.08 m/yr steps switched on
at mid-year j+0.5 (j = 0..99), so a linear system predicts
    S_ramp(t) = sum_j (0.08/8) * S_step(t - j - 0.5)       (S_step interpolated linearly).
Writes docs/figures/larmip_vs_larmip2.png, docs/figures/larmip_lrt_test.png and prints the tables.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/g/data/au88/jh7060/ACCESS-AIS3'
SETS = {'standard': f'{ROOT}/docs/larmip/AIS3_ssa_larmip_sle.csv',
        'coupledN_p3': f'{ROOT}/docs/larmip/AIS3_ssa_larmip_sle_coupledN_p3.csv'}
SUFFIX = {'standard': '', 'coupledN_p3': '_coupledN_p3'}
SECTORS = {1: 'Weddell', 2: 'East Antarctica', 3: 'Ross', 4: 'Amundsen', 5: 'Peninsula'}
L2_REGION = {1: 4, 2: 1, 3: 2, 4: 3, 5: 5}   # our sector -> LARMIP-2 region number
YRS = (50, 100, 200)

SURFACE, INK, INK2, GRID, L2C = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de', '#c9c8c2'
C_STD, C_P3 = '#2a78d6', '#eb6834'   # validated categorical slots 1-2
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'axes.titlelocation': 'left'})

# -- our responses (mm, exp - ctrl), indexed by model year 0..200 --------------------------------
ours = {}
for name, path in SETS.items():
    d = pd.read_csv(path, comment='#').set_index('year') * 1e3
    s = SUFFIX[name]
    ctrl = d[f'ctrl{s}']
    ours[name] = {k: {r: (d[f'larmip_r{r}_{k}{s}'] - ctrl).values for r in SECTORS} for k in ('step', 'ramp')}

# -- LARMIP-2 ---------------------------------------------------------------------------------------
l2 = {}
for f in glob.glob(f'{ROOT}/assets/larmip2/*_BM08_R[1-5].dat'):
    model, reg = f.split('/')[-1][:-len('_BM08_R1.dat')], int(f[-5])
    v = np.loadtxt(f) * 1e3
    l2.setdefault(reg, {})[model] = np.concatenate([[0.0], v])   # prepend year 0
models = sorted(next(iter(l2.values())))
print(f'LARMIP-2: {len(models)} models: {", ".join(models)}')


def l2_at(reg, yr):
    return np.array([l2[reg][m][yr] for m in models if l2[reg][m].size > yr])


# -- table: ours vs LARMIP-2 --------------------------------------------------------------------------
print('\n8 m/yr step, response minus control (mm SLE)')
print(f"{'sector':16s}" + ''.join(f"| y{y}: std   p3  L2 med [p5-p95] rank(p3) " for y in YRS))
for r, nm in SECTORS.items():
    row = f'{nm:16s}'
    for y in YRS:
        v = l2_at(L2_REGION[r], y)
        sd, p3 = ours['standard']['step'][r][y], ours['coupledN_p3']['step'][r][y]
        rank = 1 + int((v < p3).sum()) if np.isfinite(p3) else 0
        row += (f"| {sd:6.1f} {p3:6.1f} {np.median(v):6.1f} [{np.percentile(v, 5):5.1f}-{np.percentile(v, 95):5.1f}] "
                f"{rank:2d}/{v.size + 1} ")
    print(row)
print('rank = position of the coupledN_p3 response among the LARMIP-2 models plus ours (1 = smallest)')
tot = {k: sum(ours[k]['step'][r][200] for r in SECTORS) for k in ours}
print(f"sum of the five sectors at y200: standard {tot['standard']:.0f} mm, coupledN_p3 {tot['coupledN_p3']:.0f} mm; "
      f"LARMIP-2 median of per-model sums {np.median([sum(l2[g][m][200] for g in range(1, 6)) for m in models]):.0f} mm")


# -- LRT test ---------------------------------------------------------------------------------------------
def lrt_ramp(step):
    t = np.arange(step.size, dtype=float)
    pred = np.zeros_like(t)
    for j in range(100):
        tj = t - (j + 0.5)
        pred += (0.08 / 8.0) * np.interp(tj, t, step, left=0.0)
    return pred


print('\nLinear response test: simulated ramp vs prediction from the step run (mm; ratio sim/pred)')
lrt = {}
for name in ours:
    for r, nm in SECTORS.items():
        sim = ours[name]['ramp'][r]
        pred = lrt_ramp(ours[name]['step'][r])
        lrt[(name, r)] = (sim, pred)
        cells = []
        for y in (100, 150, 200):
            s_, p_ = sim[y], pred[y]
            cells.append(f'y{y} {s_:6.1f} vs {p_:6.1f} ({s_ / p_:.2f})' if np.isfinite(s_) and p_ > 0.05 else
                         f'y{y}   --   vs {p_:6.1f}')
        print(f'  {name:11s} {nm:16s} ' + ' | '.join(cells))

# -- figure 1: step responses vs LARMIP-2 --------------------------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(20, 4.6), sharex=True)
yr = np.arange(201)
for ax, (r, nm) in zip(axs, SECTORS.items()):
    for i, m in enumerate(models):
        v = l2[L2_REGION[r]][m]
        ax.plot(np.arange(v.size), v, color=L2C, lw=0.9, label='LARMIP-2 models (17)' if i == 0 else None)
    med = np.median(np.stack([l2[L2_REGION[r]][m][:201] for m in models]), axis=0)
    ax.plot(yr, med, color=INK2, lw=1.4, ls='--', label='LARMIP-2 median')
    ax.plot(yr, ours['standard']['step'][r], color=C_STD, lw=2, label='AIS3 standard')
    ax.plot(yr, ours['coupledN_p3']['step'][r], color=C_P3, lw=2, label='AIS3 coupled N, p = 3')
    ax.set_title(f'{r}  {nm}', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, max(np.percentile([l2[L2_REGION[r]][m][200] for m in models], 90) * 1.1, 10))
    ax.set_xlabel('model year')
axs[0].set_ylabel('sea-level contribution (mm)')
axs[0].legend(fontsize=8, loc='upper left')
fig.suptitle('LARMIP 8 m/yr step: AIS3 (LARMIP-3 sectors, minus control) vs LARMIP-2 models (matching regions; '
             'upper 10% of LARMIP-2 clipped)', x=0.01, ha='left', fontsize=11, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f'{ROOT}/docs/figures/larmip_vs_larmip2.png', dpi=160)

# -- figure 2: LRT ---------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(20, 4.6), sharex=True)
for ax, (r, nm) in zip(axs, SECTORS.items()):
    for name, col, lab in (('standard', C_STD, 'standard'), ('coupledN_p3', C_P3, 'coupled N, p = 3')):
        sim, pred = lrt[(name, r)]
        ax.plot(yr, sim, color=col, lw=2, label=f'{lab}: ramp run')
        ax.plot(yr, pred, color=col, lw=1.4, ls='--', label=f'{lab}: LRT from step')
    ax.axvspan(0, 100, color='#f0efec', zorder=0, lw=0)
    ax.set_title(f'{r}  {nm}', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 200)
    ax.set_xlabel('model year')
axs[0].set_ylabel('sea-level contribution (mm)')
axs[0].legend(fontsize=8, loc='upper left')
fig.suptitle('Ramp experiments (shaded: melt ramping 0 to 8 m/yr) vs linear-response prediction from the step runs',
             x=0.01, ha='left', fontsize=11, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f'{ROOT}/docs/figures/larmip_lrt_test.png', dpi=160)
print('\nwrote docs/figures/larmip_vs_larmip2.png, docs/figures/larmip_lrt_test.png')
