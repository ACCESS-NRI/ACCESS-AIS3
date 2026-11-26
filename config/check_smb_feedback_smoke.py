"""Check the AIS3_PROJ_SMBFB wiring on a ctrl smoke run (docs/projection_experiments.md, v2 stage B).

In a ctrl run with feedback, smbref = SMB0 and b = the GCM's fixed historical dacabfdz mean. ISSM evaluates
SMB at the start of each time step, from the surface the step starts with, so the SmbMassBalance written at
output k must equal SMB0 + b (s_{k-1} - href), where s_{k-1} is the previous output's surface (href, i.e. zero
feedback, for the first). This needs output every time step, as smoke runs have (output_frequency 1).

usage: python check_smb_feedback_smoke.py <smoke history npz> [calib tag, e.g. _hm4000] [GCM]
"""
import sys
from pathlib import Path
import numpy as np

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in (_here, *_here.parents) if (p / 'ais3_site.py').exists()][:1]
import ais3_site as site

hist = sys.argv[1]
calib = sys.argv[2] if len(sys.argv) > 2 else ''
gcm = sys.argv[3] if len(sys.argv) > 3 else 'CESM2-WACCM'
model_dir = f'{site.ROOT}/models'

h = np.load(hist)
smb0 = np.load(f'{model_dir}/AIS3_proj_ssa_baseline{calib}.npz')['smb0']
href = np.load(f'{model_dir}/AIS3_proj_ssa_href{calib}.npy')
b = np.nan_to_num(np.load(f'{model_dir}/AIS3_proj_ssa_dsmbdzclim_{gcm}_dEBM2-8000m_1995-2014.npy'), nan=0.0)
assert 'surface' in h.files, 'history has no surface: not a feedback run?'
assert h['times'].size >= 2, 'need at least two outputs (one with a nonzero feedback term)'
ok_all = True
s_prev = href
for i, t in enumerate(h['times']):
    got = h['smb'][i].astype(float) - smb0
    pred = b * (s_prev - href)
    m = (h['thickness'][i] > 1.0 + 1e-3) & np.isfinite(got)
    resid = np.max(np.abs(got[m] - pred[m]))
    scale = np.max(np.abs(pred[m]))
    slope = np.sum(got[m] * pred[m]) / np.sum(pred[m] ** 2) if scale > 0 else float('nan')
    ok = resid <= 1e-6 + 1e-3 * scale  # float32 history: ~1e-7 m/yr absolute
    ok_all &= ok
    print(f"t={t:.2f}: {m.sum()} ice vertices; feedback term max |b(s_prev-href)| {scale:.3e} m/yr; fit slope "
          f"{slope:.6f}; max |SMB - SMB0 - b(s_prev-href)| {resid:.3e} -> {'OK' if ok else 'MISMATCH'}")
    s_prev = h['surface'][i].astype(float)
print('PASS' if ok_all else 'FAIL')
sys.exit(0 if ok_all else 1)
