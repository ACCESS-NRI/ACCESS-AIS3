"""Summarise projection experiments (SSA, and HO base layer) from their per-chunk history files.

For each experiment: sea-level-equivalent from volume above flotation (whole domain and per
IMBIE2 basin), grounded area, total SMB/BMB, grounding-line and ice-front flux, and speed
change. With ctrl present, forced runs are also shown relative to ctrl (drift removed).

Usage: python check_projection_ssa.py [exp ...]      (default: every experiment with history)
Experiments are SSA names (ctrl, ssp585_cesm2_ref_ref, ...) or <track>/<name> for HO runs, e.g.
ho_relaxfix/ctrl (files AIS3_proj_ho_relaxfix_ctrl_*). Each track is measured from its own
initial state, and forced runs are compared with the CTRL of the same track.
Writes models/proj_ssa_summary.png and prints tables at the design's diagnostic years.
"""
import os
import sys
import glob
import re
import numpy as np
import xarray as xr
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
BASIN_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean/'
              'imbie2/basin_numbers_ismip8km_v2.nc')
DIAG_YEARS = [2020, 2050, 2100, 2200, 2300]
A_OCEAN = 3.625e14
RHO_OCEAN = 1028.0

# Mesh, bed and constants straight from the init file (netCDF groups, no full model load).
nc = netCDF4.Dataset(f'{MODEL_DIR}/AIS3_proj_ssa_init.nc')
x = np.asarray(nc['mesh']['x'][:], dtype=float).ravel()
y = np.asarray(nc['mesh']['y'][:], dtype=float).ravel()
elx = np.asarray(nc['mesh']['elements'][:]).astype(int) - 1
bed = np.asarray(nc['geometry']['bed'][:], dtype=float).ravel()  # HO base layer is this mesh (worklog 4w)
rho_ice = float(nc['materials'].getncattr('rho_ice'))
rho_water = float(nc['materials'].getncattr('rho_water'))
nc.close()
nv = x.size
tri_x, tri_y = x[elx], y[elx]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
area = np.zeros(nv)
for c in range(3):
    np.add.at(area, elx[:, c], elem_area / 3.0)
ds_b = xr.open_dataset(BASIN_FILE)
bxx, byy = np.meshgrid(ds_b['x'].values, ds_b['y'].values)
basin = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]),
                              ds_b['basinNumber'].values.ravel())(np.column_stack([x, y])).astype(int)
nb = int(basin.max()) + 1


def vaf_per_vertex(H, ol):
    haf = H + np.minimum(bed, 0.0) * rho_water / rho_ice
    return np.where(ol > 0, np.maximum(haf, 0.0), 0.0) * area


def sle_mm(dvaf_m3):
    return -dvaf_m3 * rho_ice / RHO_OCEAN / A_OCEAN * 1000.0


def split_exp(label):
    return label.split('/', 1) if '/' in label else ('ssa', label)


_start = {}


def track_start(track):
    """Base-layer thickness and ocean levelset of a track's projection initial state."""
    if track not in _start:
        n = netCDF4.Dataset(f'{MODEL_DIR}/AIS3_proj_{track}_init.nc')
        _start[track] = (np.asarray(n['geometry']['thickness'][:nv], dtype=float).ravel(),
                         np.asarray(n['mask']['ocean_levelset'][:nv], dtype=float).ravel())
        n.close()
    return _start[track]


def load_exp(label):
    track, exp = split_exp(label)
    H0, ol0 = track_start(track)
    files = []
    for f in glob.glob(f'{MODEL_DIR}/AIS3_proj_{track}_{exp}_*_history.npz'):
        m = re.fullmatch(rf'AIS3_proj_{re.escape(track)}_{re.escape(exp)}_(\d+)-(\d+)_history\.npz', os.path.basename(f))
        if m:
            files.append((int(m.group(1)), f))
    if not files:
        return None
    files.sort()
    vaf0 = vaf_per_vertex(H0, ol0)
    out = {'t': [2019.0], 'sle': [0.0], 'sle_basin': [np.zeros(nb)], 'garea': [np.sum(area[ol0 > 0])],
           'smb': [np.nan], 'bmb': [np.nan], 'glflux': [np.nan], 'ifflux': [np.nan], 'vel_mean': [np.nan]}
    for _, f in files:
        d = dict(np.load(f))  # read each array once; indexing an NpzFile re-reads the whole array
        for k in range(d['times'].size):
            H, ol = d['thickness'][k].astype(float), d['ocean_levelset'][k].astype(float)
            v = vaf_per_vertex(H, ol)
            out['t'].append(float(d['times'][k]))
            out['sle'].append(sle_mm(v.sum() - vaf0.sum()))
            out['sle_basin'].append(np.array([sle_mm(v[basin == b].sum() - vaf0[basin == b].sum()) for b in range(nb)]))
            out['garea'].append(np.sum(area[ol > 0]))
            out['smb'].append(float(d['scalar_TotalSmb'][k]))
            out['bmb'].append(float(d['scalar_TotalFloatingBmb'][k]))
            out['glflux'].append(float(d['scalar_GroundinglineMassFlux'][k]))
            out['ifflux'].append(float(d['scalar_IcefrontMassFlux'][k]) if 'scalar_IcefrontMassFlux' in d else np.nan)  # absent for HO
            g = ol > 0
            out['vel_mean'].append(np.sum(d['vel'][k][g] * area[g]) / area[g].sum())
    return {k: np.array(v) for k, v in out.items()}


if len(sys.argv) > 1:
    exps = sys.argv[1:]
else:
    exps = sorted({re.sub(r'_\d+-\d+_history\.npz$', '', os.path.basename(f))[len('AIS3_proj_ssa_'):]
                   for f in glob.glob(f'{MODEL_DIR}/AIS3_proj_ssa_*_history.npz') if '_smoke' not in f and re.search(r'_\d+-\d+_history\.npz$', f)})
    for f in glob.glob(f'{MODEL_DIR}/AIS3_proj_ho_relaxfix_*_history.npz'):
        if '_smoke' not in f and re.search(r'_\d+-\d+_history\.npz$', f):
            exps.append('ho_relaxfix/' + re.sub(r'_\d+-\d+_history\.npz$', '', os.path.basename(f))[len('AIS3_proj_ho_relaxfix_'):])
    exps = sorted(set(exps))
data = {e: load_exp(e) for e in exps}
data = {e: d for e, d in data.items() if d is not None}
print(f'Experiments: {list(data)}')


def at(d, key, yr):
    i = np.argmin(np.abs(d['t'] - yr))
    return d[key][i] if abs(d['t'][i] - yr) <= 0.6 else np.nan


print(f"\nSea-level contribution from VAF (mm, positive = sea-level rise); grounded area change (km^2)")
print(f"{'experiment':32s} " + ' '.join(f'{y:>9d}' for y in DIAG_YEARS) + f"  {'last t':>7} {'dGrArea':>10}")
for e, d in data.items():
    print(f"{e:32s} " + ' '.join(f'{at(d, "sle", y):9.2f}' for y in DIAG_YEARS)
          + f"  {d['t'][-1]:7.1f} {(d['garea'][-1] - d['garea'][0]) / 1e6:10.0f}")
for track in sorted({split_exp(e)[0] for e in data}):
    ctrl_label = 'ctrl' if track == 'ssa' else f'{track}/ctrl'
    if ctrl_label not in data:
        continue
    c = data[ctrl_label]
    rate = np.polyfit(c['t'][1:], c['sle'][1:], 1)[0] if c['t'].size > 3 else np.nan
    print(f"\n[{track}] CTRL drift: {rate:.3f} mm SLE/yr over {c['t'][1]:.0f}-{c['t'][-1]:.0f} (linear fit)")
    print(f'[{track}] per-basin CTRL SLE at last output (mm): '
          + ', '.join(f'b{b}:{c["sle_basin"][-1][b]:.2f}' for b in range(nb)))
    for e, d in data.items():
        if e == ctrl_label or split_exp(e)[0] != track:
            continue
        print(f"{e:32s} minus ctrl: " + ' '.join(
            f'{at(d, "sle", y) - at(c, "sle", y):9.2f}' for y in DIAG_YEARS))

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for e, d in data.items():
    ax[0, 0].plot(d['t'], d['sle'], label=e)
    ax[0, 1].plot(d['t'], (d['garea'] - d['garea'][0]) / 1e6, label=e)
    ax[1, 0].plot(d['t'][1:], d['glflux'][1:], label=e)
    ax[1, 1].plot(d['t'][1:], d['vel_mean'][1:], label=e)
ax[0, 0].set_ylabel('SLE from VAF (mm)')
ax[0, 1].set_ylabel('grounded area change (km$^2$)')
ax[1, 0].set_ylabel('GroundinglineMassFlux (ISSM units)')
ax[1, 1].set_ylabel('area-mean grounded speed (m/yr)')
for a in ax.ravel():
    a.grid(alpha=0.3)
    a.set_xlabel('year')
ax[0, 0].legend(fontsize=8)
fig.tight_layout()
png = f'{MODEL_DIR}/proj_ssa_summary.png'
fig.savefig(png, dpi=120)
print(f'\nSaved {png}')
