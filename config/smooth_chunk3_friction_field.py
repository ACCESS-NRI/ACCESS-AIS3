"""Targeted fix for the repeated iteration-2 nonlinear-solve stall (4 consecutive chunk-4
failures + all 5 cf501 probes, all ruled out on dfmin_frac/cf501 as the cause -- see
docs/post_inversion_pipeline_worklog.md SS3): chunk 3's saved FrictionC field has extreme
neighbour-to-neighbour jumps (up to 169,470x), and inspect_chunk3_friction_field.py showed
83.9% of grounded/floating-boundary edges have >10x jumps (expected -- friction really does
drop near zero on floating ice) but ALSO 48,098 edges (26% of all big jumps) with >10x jumps
WITHIN the same regime (both grounded, or both floating) -- genuine roughness/noise, not a
physically-expected discontinuity. This script smooths ONLY those anomalous same-regime
vertices (median of self + same-regime neighbours), leaving the physically-expected
grounded/floating boundary discontinuity and the rest of the (well-converged) domain
untouched.

Backs up the original checkpoint before overwriting, since `ho_friction_inv`'s resume logic
reads a fixed path (models/AIS3_ho_friction_inv.nc) -- there's no way to point it elsewhere
without editing that step itself, so the smoothed field is written back in place.

IMPORTANT: modifies md.results.StressbalanceSolution.FrictionC, not md.friction.C -- per
ho_friction_inv's own documented resume bugfix, that results field (not md.friction.C) is
what the resume logic actually reads on the next load.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import shutil
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
ORIG = f'{MODEL_DIR}/AIS3_ho_friction_inv.nc'
BACKUP = f'{MODEL_DIR}/AIS3_ho_friction_inv_chunk3_backup_presmoothing.nc'
JUMP_RATIO_THRESHOLD = 10.0

if not os.path.exists(BACKUP):
    print(f'-- Backing up {ORIG} -> {BACKUP} before any modification...', flush=True)
    shutil.copy2(ORIG, BACKUP)
else:
    print(f'-- Backup already exists at {BACKUP}, not overwriting it.', flush=True)

print('-- Loading chunk 3 checkpoint...', flush=True)
md = pyissm.model.io.load_model(ORIG)

C = np.asarray(md.results.StressbalanceSolution.FrictionC).ravel().copy()
on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
on_base_idx = np.where(on_base)[0]
nv2d = on_base_idx.size
Cb = C[on_base_idx]
ol = np.asarray(md.mask.ocean_levelset).ravel()[on_base_idx]
floating = ol < 0

elements2d = np.asarray(md.mesh.elements2d).astype(int) - 1  # 0-indexed into the 2D vertex numbering

print('-- Building vertex adjacency from base-layer elements...', flush=True)
# Adjacency lists, restricted to same-regime neighbours only (grounded-grounded or floating-floating).
neighbours = [[] for _ in range(nv2d)]
for tri in elements2d:
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        i, j = tri[a], tri[b]
        if floating[i] == floating[j]:
            neighbours[i].append(j)
            neighbours[j].append(i)

print('-- Identifying anomalous same-regime edges (>{:.0f}x jump)...'.format(JUMP_RATIO_THRESHOLD), flush=True)
c0 = Cb[elements2d[:, 0]]; c1 = Cb[elements2d[:, 1]]; c2 = Cb[elements2d[:, 2]]
f0 = floating[elements2d[:, 0]]; f1 = floating[elements2d[:, 1]]; f2 = floating[elements2d[:, 2]]
flagged = np.zeros(nv2d, dtype=bool)
for (ia, ib), (ca, cb), (fa, fb) in [
    ((elements2d[:, 0], elements2d[:, 1]), (c0, c1), (f0, f1)),
    ((elements2d[:, 1], elements2d[:, 2]), (c1, c2), (f1, f2)),
    ((elements2d[:, 0], elements2d[:, 2]), (c0, c2), (f0, f2)),
]:
    lo = np.minimum(ca, cb)
    hi = np.maximum(ca, cb)
    same_regime = fa == fb
    valid = (lo > 1e-6) & same_regime
    ratio = np.where(valid, hi / np.where(valid, lo, 1.0), 0.0)
    big = ratio > JUMP_RATIO_THRESHOLD
    flagged[ia[big]] = True
    flagged[ib[big]] = True

n_flagged = int(flagged.sum())
print(f'   {n_flagged} vertices ({100*n_flagged/nv2d:.2f}%) touch an anomalous same-regime edge -- these will be smoothed')

print('-- Applying median-of-self-and-same-regime-neighbours smoothing to flagged vertices...', flush=True)
Cb_new = Cb.copy()
n_changed = 0
for i in np.where(flagged)[0]:
    nb = neighbours[i]
    if not nb:
        continue
    vals = np.array([Cb[i]] + [Cb[j] for j in nb])
    new_val = np.median(vals)
    if new_val != Cb[i]:
        n_changed += 1
    Cb_new[i] = new_val

print(f'   {n_changed} vertices actually changed value')
print(f'   Cb range before: [{Cb.min():.4g}, {Cb.max():.4g}]  after: [{Cb_new.min():.4g}, {Cb_new.max():.4g}]')

C_new = C.copy()
C_new[on_base_idx] = Cb_new

print('-- Writing smoothed field back into md.results.StressbalanceSolution.FrictionC '
      '(the field ho_friction_inv\'s resume logic actually reads) and md.friction.C...', flush=True)
md.results.StressbalanceSolution.FrictionC = C_new
md.friction.C = C_new.copy()

print(f'-- Saving smoothed checkpoint to {ORIG} (overwriting, original preserved at {BACKUP})...', flush=True)
pyissm.model.io.save_model(md, ORIG)
print('DONE')
