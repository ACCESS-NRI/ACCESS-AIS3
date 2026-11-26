"""Sea-level table for the projection runs, each minus its own ctrl (docs/projection_experiments.md, worklog 6.35).

Stitches every chunk history (models/AIS3_proj_ssa_<exp>_<t0>-<t1>_history.npz) of a run and its ctrl, and prints
the change since 2019 in ISSM IceVolumeAboveFloatation as mm SLE (+ = rise; rho_ice 917, rho_sw 1028, ocean
3.625e14 m^2, as the projection driver prints it). Years past a run's last history are left blank. Light: reads
only the scalar series, so it runs on a login node.

usage: python proj_v2_sle_table.py [years, default 2050,2100,2150,2200,2250,2300]
"""
import glob
import os
import re
import sys
import numpy as np

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
YEARS = [int(y) for y in (sys.argv[1] if len(sys.argv) > 1 else '2050,2100,2150,2200,2250,2300').split(',')]
K = 917.0 / (1028.0 * 3.625e14) * 1000.0
V2 = '_coupledN_p3_hm4000'
ROWS = [  # (label, run, its ctrl)
    ('Oct ssp585 CESM2', 'ssp585_cesm2_ref_ref', 'ctrl'),
    ('Oct ssp585 CESM2 ocean-only', 'ssp585_cesm2_ref_oceanonly', 'ctrl'),
    ('Oct ssp585 CESM2 collapse', 'ssp585_cesm2_ref_ref_collapse', 'ctrl'),
    ('Oct ssp585 MRI', 'ssp585_mri_ref_ref', 'ctrl'),
    ('Oct ssp126 CESM2', 'ssp126_cesm2_ref_ref', 'ctrl'),
    ('Oct ssp126 MRI', 'ssp126_mri_ref_ref', 'ctrl'),
    ('v2 ssp585 CESM2', 'ssp585_cesm2_ref_ref' + V2, 'ctrl' + V2),
    ('v2 ssp585 CESM2 collapse', 'ssp585_cesm2_ref_ref_collapse' + V2, 'ctrl' + V2),
    ('v2 ssp585 MRI', 'ssp585_mri_ref_ref' + V2, 'ctrl' + V2),
    ('v2 ssp585 MRI ocean-only', 'ssp585_mri_ref_oceanonly' + V2, 'ctrl' + V2),
    ('v2 ssp585 MRI SMB-only', 'ssp585_mri_ref_smbonly' + V2, 'ctrl' + V2),
    ('v2 ssp126 CESM2', 'ssp126_cesm2_ref_ref' + V2, 'ctrl' + V2),
    ('v2 ssp126 MRI', 'ssp126_mri_ref_ref' + V2, 'ctrl' + V2),
    ('v2 ssp534-over CESM2', 'ssp534-over_cesm2_ref_ref' + V2, 'ctrl' + V2),
    ('v2 ssp534-over MRI', 'ssp534-over_mri_ref_ref' + V2, 'ctrl' + V2),
    ('v2+fb ssp585 CESM2', 'ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000', 'ctrl_coupledN_p3_smbfb-cesm2_hm4000'),
    ('v2+fb ssp585 MRI', 'ssp585_mri_ref_ref_coupledN_p3_smbfb_hm4000', 'ctrl_coupledN_p3_smbfb-mri_hm4000'),
    ('v2 driftcorr ssp585 CESM2', 'ssp585_cesm2_ref_ref_coupledN_p3_driftcorr_hm4000', 'ctrl_coupledN_p3_driftcorr_hm4000'),
    ('v2 hm8000 ssp585 CESM2', 'ssp585_cesm2_ref_ref_coupledN_p3_hm8000', 'ctrl_coupledN_p3_hm8000'),
    ('v2 SEM1 ssp585 CESM2', 'ssp585_cesm2_ref_ref_coupledN_p3_hm4000sem1', 'ctrl_coupledN_p3_hm4000sem1'),
    ('v2 SEM1+fb ssp585 CESM2', 'ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000sem1',
     'ctrl_coupledN_p3_smbfb-cesm2_hm4000sem1'),
]
CTRLS = ['ctrl', 'ctrl' + V2, 'ctrl_coupledN_p3_smbfb-cesm2_hm4000', 'ctrl_coupledN_p3_smbfb-mri_hm4000',
         'ctrl_coupledN_p3_driftcorr_hm4000', 'ctrl_coupledN_p3_hm8000', 'ctrl_coupledN_p3_hm4000sem1',
         'ctrl_coupledN_p3_smbfb-cesm2_hm4000sem1']
_cache = {}


def series(exp):
    """(times, VAF) over all chunks of exp; None if it has no history yet."""
    if exp in _cache:
        return _cache[exp]
    pat = re.compile(rf'^AIS3_proj_ssa_{re.escape(exp)}_(\d{{4}})-(\d{{4}})_history\.npz$')
    files = sorted((int(m.group(1)), f) for f in glob.glob(f'{M}/AIS3_proj_ssa_{glob.escape(exp)}_*_history.npz')
                   if (m := pat.match(os.path.basename(f))))
    if not files:
        _cache[exp] = None
        return None
    t, v = [], []
    for _, f in files:
        h = np.load(f)
        t.append(np.asarray(h['times'], float))
        v.append(np.asarray(h['scalar_IceVolumeAboveFloatation'], float))
    t, v = np.concatenate(t), np.concatenate(v)
    o = np.argsort(t, kind='stable')
    _cache[exp] = (t[o], v[o])
    return _cache[exp]


def sle_at(exp, y):
    s = series(exp)
    if s is None:
        return None
    t, v = s
    if t[-1] < y - 0.01:
        return None
    i = int(np.argmin(np.abs(t - y)))
    return -(v[i] - v[0]) * K


if __name__ == '__main__':
    hdr = f"{'mm SLE, run minus its ctrl':30s}" + ''.join(f'{y:>9d}' for y in YEARS)
    print(hdr)
    for label, e, c in ROWS:
        cells = []
        for y in YEARS:
            a, b = sle_at(e, y), sle_at(c, y)
            cells.append(f'{a - b:+9.1f}' if a is not None and b is not None else ' ' * 9)
        print(f'{label:30s}' + ''.join(cells))
    print('\nctrl drift (mm SLE since 2019)')
    for c in CTRLS:
        print(f'{c:42s}' + ''.join((f'{x:+9.1f}' if (x := sle_at(c, y)) is not None else ' ' * 9) for y in YEARS))
