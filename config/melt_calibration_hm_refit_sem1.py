"""SEM1 melt calibration: per-basin deltaT for melt on partly floating elements (docs/projection_experiments.md,
v2 stage C; worklog 6.35).

With the patched ISSM (ismip6 melt computed on partly floating elements) and
groundingline.melt_interpolation = SubelementMelt1, ISSM's basal mass balance on a partly floating element is
(1 - phi) fmb + phi gmb at each Gauss point (MasstransportAnalysis.cpp), phi = grounded fraction. So each element
contributes  mean(vertex melt) * area * floating fraction.

This refits the 16 IMBIE2-basin deltaT at gamma_0 = 4000 so that, with SEM1, each basin's present-day melt equals
what the same element integral gives WITHOUT SEM1 at the existing hm4000 deltaT (fully floating elements only).
The SEM1 run then has the same present-day melt per basin as the no-SEM1 run, moved towards the grounding line, so
the stage-C 2x2 isolates where the melt goes, not how much. Sanity check printed: the no-SEM1 element integral at
hm4000 vs the basin totals stored in melt_calibration_hm.json (vertex-averaged form used there).

Same mesh, geometry, TF (Zhou), basin ids and melt formula as melt_calibration_hm_refit.py. Writes
models/melt_calibration_hm_sem1.json (refuses to overwrite) with member hm4000sem1, for AIS3_SSA_CALIB=_hm4000sem1.
"""
import os
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import json
import numpy as np
import xarray as xr
import pyissm
from scipy.interpolate import RegularGridInterpolator

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
OUT = f'{M}/melt_calibration_hm_sem1.json'
ZHOU_TF = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
           'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
BASE_MEMBER = 'hm4000'
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')

hm = json.load(open(f'{M}/melt_calibration_hm.json'))
mem0 = hm['members'][BASE_MEMBER]
g0, dt0 = float(mem0['gamma_0']), np.asarray(mem0['delta_t'], float)

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
tx, ty = mx[elx], my[elx]
earea = 0.5 * np.abs((tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (tx[:, 2] - tx[:, 0]) * (ty[:, 1] - ty[:, 0]))
has_ice = (il[elx] < 0).any(axis=1)

# Floating fraction of each element from the linear ocean levelset (negative = floating)
L = ol[elx]
nneg = (L < 0).sum(axis=1)
ffrac = np.where(nneg == 3, 1.0, 0.0)
with np.errstate(divide='ignore', invalid='ignore'):
    one = nneg == 1   # the one floating vertex's corner triangle
    i_ = np.argmin(L, axis=1)
    a = L[np.arange(L.shape[0]), i_]
    b, c = L[np.arange(L.shape[0]), (i_ + 1) % 3], L[np.arange(L.shape[0]), (i_ + 2) % 3]
    ffrac = np.where(one, a * a / ((a - b) * (a - c)), ffrac)
    two = nneg == 2   # one grounded vertex: floating = 1 - its corner triangle
    j_ = np.argmax(L, axis=1)
    p = L[np.arange(L.shape[0]), j_]
    q, r = L[np.arange(L.shape[0]), (j_ + 1) % 3], L[np.arange(L.shape[0]), (j_ + 2) % 3]
    ffrac = np.where(two, 1.0 - p * p / ((p - q) * (p - r)), ffrac)
ffrac = np.clip(np.nan_to_num(ffrac, nan=0.0), 0.0, 1.0)
full = (nneg == 3) & has_ice
part = (nneg > 0) & (nneg < 3) & has_ice

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


def basin_totals(dt, sem1):
    """Gt/yr per basin: mean vertex melt * element area * (floating fraction with SEM1, else fully floating only)."""
    m = (g0 * coef * np.maximum(TFe + dt[basin_el][:, None], 0.0) ** 2).mean(axis=1)
    w = np.where(full, 1.0, 0.0) + (np.where(part, ffrac, 0.0) if sem1 else 0.0)
    return np.bincount(basin_el, weights=m * earea * w * rho_ice / 1e12, minlength=nb)


tot0 = basin_totals(dt0, False)                 # target: no-SEM1 present-day melt at hm4000
tot0_sem1 = basin_totals(dt0, True)
stored = np.asarray(mem0['basin_totals'], float)
print(f'{int(part.sum())} partly floating ice elements, {int(full.sum())} fully floating; partly floating area '
      f'{np.sum(earea[part] * ffrac[part]) / 1e6:.0f} km^2 of floating ice')
print(f'sanity: no-SEM1 element integral at {BASE_MEMBER} {tot0.sum():.1f} Gt/yr vs stored vertex-form total '
      f'{stored.sum():.1f}; per basin max |diff| {np.abs(tot0 - stored).max():.2f} Gt/yr')
print(f'SEM1 at {BASE_MEMBER} deltaT (no refit): {tot0_sem1.sum():.1f} Gt/yr (+{100 * (tot0_sem1.sum() / tot0.sum() - 1):.1f}%); '
      'per basin %: ' + ' '.join(f'{100 * (s / t - 1):+.1f}' for s, t in zip(tot0_sem1, tot0)))

lo, hi = dt0 - 10.0, dt0 + 2.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    tot = basin_totals(mid, True)
    lo = np.where(tot < tot0, mid, lo)
    hi = np.where(tot >= tot0, mid, hi)
dt = 0.5 * (lo + hi)
tot = basin_totals(dt, True)
print(f'hm4000sem1: deltaT {dt.min():.3f} .. {dt.max():.3f} (shift from hm4000 {np.min(dt - dt0):+.3f} .. '
      f'{np.max(dt - dt0):+.3f} degC); total {tot.sum():.1f} vs target {tot0.sum():.1f} Gt/yr; max |basin error| '
      f'{np.abs(tot - tot0).max():.4f} Gt/yr')
print('   deltaT by basin: ' + ' '.join(f'{v:.3f}' for v in dt))
out = {'description': __doc__.strip().splitlines()[0], 'base_member': BASE_MEMBER,
       'target_gt_per_yr_element_integral_no_sem1': tot0.tolist(),
       'members': {'hm4000sem1': {'gamma_0': g0, 'delta_t': dt.tolist(), 'basin_totals_sem1': tot.tolist(),
                                  'melt_interpolation': 'SubelementMelt1', 'requires': 'ISSM with the ismip6 '
                                  'IsAllFloating test removed (marker file SEM1_PATCH in AIS3_ISSM_DIR)'}}}
json.dump(out, open(OUT, 'w'), indent=1)
print(f'saved {OUT}\nDONE')
