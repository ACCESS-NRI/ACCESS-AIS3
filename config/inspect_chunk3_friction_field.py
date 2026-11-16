"""Inspect chunk 3's saved FrictionC field (models/AIS3_ho_friction_inv.nc, the checkpoint
every failed chunk-4/cf501-resweep attempt has resumed from) for pathological values that
could explain why the iteration-2 nonlinear stress-balance solve consistently hangs from this
exact starting point, regardless of cf501 (ruled out: failed identically at cf501=1e-6) or
dfmin_frac (ruled out: failed identically with a conservative 0.3).

Checks: basic stats, NaN/Inf, near-bound clustering (Schoof bounds are [0.05, 250**2=62500]
per friction_law_info in ais_0.1.py), extreme-gradient hotspots (large C jumps between
neighbouring vertices -- a classic source of Newton-solve non-convergence), and cross-
references extreme locations against thickness/velocity to see if they cluster somewhere
physically informative (grounding line, fast-flow trunks, thin ice).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'

print('-- Loading chunk 3 checkpoint (AIS3_ho_friction_inv.nc)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')

C = np.asarray(md.results.StressbalanceSolution.FrictionC).ravel()
on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
Cb = C[on_base]  # friction is only physically meaningful at the base

print(f'\n=== FrictionC basic stats (base-layer vertices, n={Cb.size}) ===')
print(f'   min={np.nanmin(Cb):.6g}  max={np.nanmax(Cb):.6g}  mean={np.nanmean(Cb):.6g}  '
      f'median={np.nanmedian(Cb):.6g}  std={np.nanstd(Cb):.6g}')
print(f'   NaN count: {np.isnan(Cb).sum()}')
print(f'   Inf count: {np.isinf(Cb).sum()}')

BOUND_MIN, BOUND_MAX = 0.05, 250.0 ** 2
print(f'\n=== Bound clustering (Schoof bounds [{BOUND_MIN}, {BOUND_MAX}]) ===')
near_min = np.abs(Cb - BOUND_MIN) < 1e-3
near_max = np.abs(Cb - BOUND_MAX) < 1.0
print(f'   at/near min bound ({BOUND_MIN}): {near_min.sum()} ({100*near_min.mean():.2f}%)')
print(f'   at/near max bound ({BOUND_MAX}): {near_max.sum()} ({100*near_max.mean():.2f}%)')
below_min = Cb < BOUND_MIN
above_max = Cb > BOUND_MAX
print(f'   BELOW min bound (should be impossible): {below_min.sum()}')
print(f'   ABOVE max bound (should be impossible): {above_max.sum()}')

print(f'\n=== Percentile distribution ===')
for p in [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]:
    print(f'   p{p:5.1f}: {np.nanpercentile(Cb, p):.4g}')

print('\n-- Checking for extreme neighbour-to-neighbour gradients (2D base mesh, vectorized)...', flush=True)
# BUGFIX: this is an extruded 3D mesh -- md.mesh.elements are 3D wedge elements (6 nodes),
# and requiring ALL 6 to be on-base is impossible (top-layer nodes never are), which crashed
# with a zero-size array. Use the 2D base-layer connectivity (elements2d, set by extrude())
# instead, matching every other script this session that works with this mesh's base layer.
elements2d = np.asarray(md.mesh.elements2d).astype(int) - 1  # indices into the 2D (base) vertex numbering
x2d = np.asarray(md.mesh.x2d).ravel()
# Map 2D vertex index -> full 3D vertex index (on_base_idx[k] is the 3D index of 2D vertex k,
# since vertexonbase preserves the base-layer vertex ordering used by extrude()).
C_at_2d = Cb  # Cb is already C[on_base], i.e. indexed in the same 2D order as elements2d
base_elements = elements2d

c0, c1, c2 = C_at_2d[base_elements[:, 0]], C_at_2d[base_elements[:, 1]], C_at_2d[base_elements[:, 2]]
edges = [(c0, c1), (c1, c2), (c0, c2)]
ratios_all = []
for a, b in edges:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    valid = lo > 1e-6
    ratios_all.append(np.where(valid, hi / np.where(valid, lo, 1.0), 1.0))
ratios = np.concatenate(ratios_all)
max_jump = np.nanmax(ratios)
jump_count_gt10x = int((ratios > 10).sum())
total_edges = ratios.size
print(f'   {total_edges} base-element edges: max C ratio (neighbour jump) = {max_jump:.2f}x, '
      f'{jump_count_gt10x} edges with >10x jump ({100*jump_count_gt10x/max(total_edges,1):.3f}%)')

print('\n-- Checking whether extreme jumps sit at grounded/floating boundaries...', flush=True)
ol_2d = np.asarray(md.mask.ocean_levelset).ravel()[on_base]
floating_2d = ol_2d < 0
f0, f1, f2 = floating_2d[base_elements[:, 0]], floating_2d[base_elements[:, 1]], floating_2d[base_elements[:, 2]]
edge_floating_pairs = [(f0, f1), (f1, f2), (f0, f2)]
crosses_gl_all = []
for (fa, fb) in edge_floating_pairs:
    crosses_gl_all.append(fa != fb)  # one floating, one grounded
crosses_gl = np.concatenate(crosses_gl_all)
big_jump = ratios > 10
both = big_jump & crosses_gl
print(f'   Of {jump_count_gt10x} edges with >10x jump: {int(both.sum())} '
      f'({100*both.sum()/max(jump_count_gt10x,1):.1f}%) cross a grounded/floating boundary')
print(f'   Of {int(crosses_gl.sum())} grounded/floating-boundary edges total: '
      f'{int((big_jump & crosses_gl).sum())} ({100*(big_jump & crosses_gl).sum()/max(crosses_gl.sum(),1):.1f}%) have >10x jump')
# Also: big jumps NOT at a GL boundary -- these would be within-grounded (or within-floating) instability
within_grounded_big_jump = big_jump & ~crosses_gl
print(f'   Big jumps NOT at a GL boundary (i.e. within same grounded/floating state): '
      f'{int(within_grounded_big_jump.sum())} ({100*within_grounded_big_jump.sum()/max(jump_count_gt10x,1):.1f}% of big jumps)')

print('\n-- Cross-referencing extreme-C locations against thickness/velocity...', flush=True)
H = np.asarray(md.geometry.thickness).ravel()[on_base]
ol = np.asarray(md.mask.ocean_levelset).ravel()[on_base]
vel_obs = np.asarray(md.inversion.vel_obs).ravel()[on_base] if hasattr(md.inversion, 'vel_obs') else None
extreme = near_max | (Cb > np.nanpercentile(Cb, 99.9))
print(f'   {extreme.sum()} vertices in top 0.1%/near-max-bound: '
      f'mean thickness there = {np.nanmean(H[extreme]):.1f} m (vs. domain mean {np.nanmean(H):.1f} m)')
if vel_obs is not None:
    print(f'   mean obs velocity there = {np.nanmean(vel_obs[extreme]):.1f} m/yr '
          f'(vs. domain mean {np.nanmean(vel_obs):.1f} m/yr)')

print(f'\n-- Cross-referencing the near-MIN-bound cluster (the big one, {near_min.sum()} vertices)...', flush=True)
print(f'   mean thickness there = {np.nanmean(H[near_min]):.1f} m (vs. domain mean {np.nanmean(H):.1f} m)')
print(f'   fraction floating (ocean_levelset<0) there = {100*np.mean(ol[near_min] < 0):.1f}% '
      f'(vs. domain floating fraction {100*np.mean(ol < 0):.1f}%)')
if vel_obs is not None:
    print(f'   mean obs velocity there = {np.nanmean(vel_obs[near_min]):.1f} m/yr '
          f'(vs. domain mean {np.nanmean(vel_obs):.1f} m/yr)')
print(f'   min/max thickness in this cluster: {np.nanmin(H[near_min]):.1f} / {np.nanmax(H[near_min]):.1f} m')

print(f'\n-- Investigating the p95=p99=5000 plateau...', flush=True)
near_5000 = np.abs(Cb - 5000) < 5
print(f'   {near_5000.sum()} vertices within 5 of exactly 5000 ({100*near_5000.mean():.2f}%)')

print('\nDONE')
