"""Export the LARMIP sea-level time series (SSA track) for submission.

One column per experiment, 201 rows: model year 0 (2019.0, the common initial state) to 200
(2219.0), sea-level contribution from the change in volume above flotation, in metres (positive =
sea-level rise), NOT control-subtracted (LARMIP takes the differences). Experiments: ctrl and
larmip_r1..r5_step (core); larmip_r1..r5_ramp too if present.

VAF uses the ISMIP convention (vertex-area weighted, grounded vertices only, height above
flotation clipped at zero), computed from the saved annual thickness and grounded mask as in
check_projection_ssa.py. ISSM's own IceVolumeAboveFloatation scalar is printed as a cross-check;
it counts every not-fully-floating element whole and does not clip, so the two differ in the
grounding zone (CTRL 2020-2050: -16.75 vs -15.73 mm, ~6%).

Usage: python export_larmip.py [--ramp]        -> docs/larmip/AIS3_ssa_larmip_sle.csv
       python export_larmip.py [--ramp] --variant coupledN_p3
                                               -> docs/larmip/AIS3_ssa_larmip_sle_coupledN_p3.csv
       (experiments <exp>_<variant>, e.g. ctrl_coupledN_p3, larmip_r1_step_coupledN_p3)
"""
import os
import re
import sys
import glob
import numpy as np
import netCDF4

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
OUT = '/g/data/au88/jh7060/ACCESS-AIS3/docs/larmip/AIS3_ssa_larmip_sle.csv'
START, YEARS = 2019, 200
A_OCEAN = 3.625e14   # m^2, ISMIP6/7 ocean area
RHO_OCEAN = 1028.0   # kg m^-3, as used for the SLE conversion elsewhere in this project

exps = ['ctrl'] + [f'larmip_r{r}_step' for r in range(1, 6)]
if '--ramp' in sys.argv:
    exps += [f'larmip_r{r}_ramp' for r in range(1, 6)]
VARIANT = sys.argv[sys.argv.index('--variant') + 1] if '--variant' in sys.argv else ''
if VARIANT:
    exps = [f'{e}_{VARIANT}' for e in exps]
    OUT = OUT.replace('.csv', f'_{VARIANT}.csv')

nc = netCDF4.Dataset(f'{MODEL_DIR}/AIS3_proj_ssa_init.nc')
x = np.asarray(nc['mesh']['x'][:], dtype=float).ravel()
y = np.asarray(nc['mesh']['y'][:], dtype=float).ravel()
elx = np.asarray(nc['mesh']['elements'][:]).astype(int) - 1
bed = np.asarray(nc['geometry']['bed'][:], dtype=float).ravel()
H0 = np.asarray(nc['geometry']['thickness'][:], dtype=float).ravel()
ol0 = np.asarray(nc['mask']['ocean_levelset'][:], dtype=float).ravel()
rho_ice = float(nc['materials'].getncattr('rho_ice'))
rho_water = float(nc['materials'].getncattr('rho_water'))
nc.close()
tx, ty = x[elx], y[elx]
earea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
area = np.zeros(x.size)
for c in range(3):
    np.add.at(area, elx[:, c], earea / 3.0)


def vaf(H, ol):
    haf = H + np.minimum(bed, 0.0) * rho_water / rho_ice
    return np.sum(np.where(ol > 0, np.maximum(haf, 0.0), 0.0) * area)


vaf0 = vaf(H0, ol0)
cols, notes = {}, []
for exp in exps:
    series = {0: 0.0}
    issm = {}
    for f in sorted(glob.glob(f'{MODEL_DIR}/AIS3_proj_ssa_{exp}_*_history.npz')):
        if not re.fullmatch(rf'AIS3_proj_ssa_{re.escape(exp)}_\d+-\d+_history\.npz', os.path.basename(f)):
            continue
        d = dict(np.load(f))  # read each array once; indexing an NpzFile re-reads the whole array
        for k, t in enumerate(d['times']):
            yr = int(round(t)) - START
            if abs(t - round(t)) < 1e-6 and 1 <= yr <= YEARS:
                series[yr] = -(vaf(d['thickness'][k].astype(float), d['ocean_levelset'][k].astype(float)) - vaf0) \
                    * rho_ice / RHO_OCEAN / A_OCEAN
                issm[yr] = float(d['scalar_IceVolumeAboveFloatation'][k])
    missing = [yy for yy in range(YEARS + 1) if yy not in series]
    if missing:
        notes.append(f'{exp}: {len(missing)} of {YEARS + 1} years missing (first {missing[0]})')
    cols[exp] = [series.get(yy, np.nan) for yy in range(YEARS + 1)]
    if len(issm) > 1:  # VAF changes from both methods should agree closely
        y0, y1 = min(issm), max(issm)
        mine = (cols[exp][y1] - cols[exp][y0])
        theirs = -(issm[y1] - issm[y0]) * rho_ice / RHO_OCEAN / A_OCEAN
        print(f'{exp}: SLE year {y0}->{y1}: {mine * 1e3:+.2f} mm (this script) vs {theirs * 1e3:+.2f} mm (ISSM VAF)')

with open(OUT, 'w') as f:
    f.write('# ACCESS-AIS3 (SSA) LARMIP sea-level contribution, m (positive = sea-level rise), '
            'from volume above flotation; not control-subtracted. Year 0 = 2019.0 initial state. '
            'Regions (LARMIP3_sectors): r1 Weddell, r2 East Antarctica, r3 Ross, r4 Amundsen, r5 Peninsula.'
            + (' Friction variant coupledN_p3: Budd p=3 q=1 with effective pressure from flotation, '
               'drag-preserving rescale at 2019.' if VARIANT == 'coupledN_p3' else
               (f' Friction variant {VARIANT}.' if VARIANT else '')) + '\n')
    f.write('year,' + ','.join(cols) + '\n')
    for yy in range(YEARS + 1):
        f.write(f'{yy},' + ','.join('' if np.isnan(cols[e][yy]) else f'{cols[e][yy]:.6e}' for e in cols) + '\n')
print(f'Wrote {OUT}')
for n in notes:
    print('INCOMPLETE:', n)
