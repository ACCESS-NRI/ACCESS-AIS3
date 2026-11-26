"""Per-basin deltaT for the history-matched melt calibration (worklog 6.30-6.31).

melt_history_match.py gives gamma_0 ~ 4,000 as the reference (range ~1,000-8,000). This refits the 16
IMBIE2-basin deltaT values at gamma_0 = 1000, 4000, 8000 so that each basin's present-day melt (Zhou
climatology, ISSM ismip6 local formula in Python, checked against ISSM in ssa_melt_j3_refit.py) matches
the Paolo/Davison/Adusumilli basin total, using the exact vertex-basin totals (simultaneous bisection,
which converges because each basin total depends almost only on its own deltaT). Same mesh, geometry
and basin ids as AIS3_melt_final_ssa.nc. Writes models/melt_calibration_hm.json (refuses to overwrite);
members are named hm<gamma_0>, e.g. hm4000, for AIS3_SSA_CALIB in ais_0.1_SSA.py.
"""
import os
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import json
import numpy as np
import pandas as pd
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
OUT = f'{M}/melt_calibration_hm.json'
OC = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
ZHOU_TF = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
           'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
GAMMAS = (1000.0, 4000.0, 8000.0)
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')

md = pyissm.model.io.load_model(f'{M}/AIS3_melt_final_ssa.nc')
md.results.TransientSolution = []
nv = md.mesh.numberofvertices
mx, my = np.asarray(md.mesh.x, float).ravel(), np.asarray(md.mesh.y, float).ravel()
elx = np.asarray(md.mesh.elements).astype(int) - 1
base = np.asarray(md.geometry.base, float).ravel()
ol = np.asarray(md.mask.ocean_levelset, float).ravel()
il = np.asarray(md.mask.ice_levelset, float).ravel()
basin_el = np.asarray(md.basalforcings.basin_id, float).ravel().astype(int) - 1
nb = int(md.basalforcings.num_basins)
rho_ice = float(md.materials.rho_ice)
coef = ((1028.0 * float(md.materials.mixed_layer_capacity)) / (rho_ice * float(md.materials.latentheat))) ** 2
el_float = (ol[elx] < 0).all(axis=1) & (il[elx] < 0).any(axis=1)
n_conn = np.bincount(elx.ravel(), minlength=nv).astype(float)
tx, ty = mx[elx], my[elx]
earea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
varea = np.bincount(elx.ravel(), weights=np.repeat(earea / 3.0, 3), minlength=nv)
basin_v = np.round(np.bincount(elx.ravel(), weights=np.repeat(basin_el, 3), minlength=nv) / np.maximum(n_conn, 1)).astype(int)
floating = ol < 0
FLUX = rho_ice * varea / 1e12

ds = xr.open_dataset(ZHOU_TF)
gx, gy, gz = ds['x'].values.astype(float), ds['y'].values.astype(float), ds['z'].values
dep = -np.asarray(gz, float)
tfm = np.empty((gz.size, nv))
for k in range(gz.size):
    tfm[k] = RegularGridInterpolator((gy, gx), ds['tf'].values[k], method='linear', bounds_error=False, fill_value=np.nan)(np.column_stack([my, mx]))
tfm = np.nan_to_num(tfm, nan=0.0)   # as the ISSM input
d = -base
k = np.clip(np.searchsorted(dep, d, side='right') - 1, 0, dep.size - 2)
a2 = np.clip((d - dep[k]) / (dep[k + 1] - dep[k]), 0.0, 1.0)
tfd = (1 - a2) * tfm[k, np.arange(nv)] + a2 * tfm[k + 1, np.arange(nv)]
del tfm
TFe = tfd[elx]


def basin_totals(g0, dt):
    m = g0 * coef * np.maximum(TFe + dt[basin_el][:, None], 0.0) ** 2
    m[~el_float] = 0.0
    mv = np.bincount(elx.ravel(), weights=m.ravel(), minlength=nv) / np.maximum(n_conn, 1)
    return np.bincount(basin_v[floating], weights=(mv * FLUX)[floating], minlength=nb)


obs = pd.read_csv(f'{OC}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv')
target = obs['BMR (Gt/yr)'].values
out = {'description': __doc__.strip().splitlines()[0], 'target_gt_per_yr': target.tolist(), 'members': {}}
for g0 in GAMMAS:
    lo, hi = np.full(nb, -8.0), np.full(nb, 40.0)
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        tot = basin_totals(g0, mid)
        lo = np.where(tot < target, mid, lo)
        hi = np.where(tot >= target, mid, hi)
    dt = 0.5 * (lo + hi)
    tot = basin_totals(g0, dt)
    err = tot - target
    name = f'hm{g0:.0f}'
    out['members'][name] = {'gamma_0': g0, 'delta_t': dt.tolist(), 'basin_totals': tot.tolist()}
    print(f'{name}: deltaT {dt.min():6.3f} .. {dt.max():6.3f} (median {np.median(dt):.3f}; |deltaT|>2 in '
          f'{int((np.abs(dt) > 2).sum())} basins); total {tot.sum():.1f} vs {target.sum():.1f} Gt/yr; '
          f'max |basin error| {np.abs(err).max():.3f} Gt/yr')
    print('   deltaT by basin: ' + ' '.join(f'{v:.3f}' for v in dt))
json.dump(out, open(OUT, 'w'), indent=1)
print(f'saved {OUT}\nDONE')
