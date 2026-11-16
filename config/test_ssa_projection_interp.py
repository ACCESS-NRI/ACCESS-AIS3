"""Smoke test for projection_ssp_ssa's new time-varying SMB/TF interpolation plumbing
(_interp_weights/_interp_apply in ais_0.1_SSA.py), WITHOUT running the actual (expensive)
transient solve. Restricted to a small 3-year window (2019-2021) to keep this cheap --
validates: real ISMIP7 file loading, unit conversion correctness, interpolation weight
build + apply timing on the REAL SSA mesh scale, and sane output magnitudes.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import time
import numpy as np
import xarray as xr
import pyissm
from scipy.spatial import Delaunay
import glob
from pathlib import Path

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
FORCING_ROOT = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS'
START_YEAR = 2019
END_YEAR = 2021  # small window for a cheap smoke test -- real scaffold uses 2019-2100
SMB_PARAM = 'SDBN1-8000m'
GCM, SCENARIO = 'CESM2-WACCM', 'ssp585'


def _interp_weights(src_x, src_y, tgt_x, tgt_y):
    src_xy = np.column_stack([np.asarray(src_x).ravel(), np.asarray(src_y).ravel()])
    tgt_xy = np.column_stack([np.asarray(tgt_x).ravel(), np.asarray(tgt_y).ravel()])
    tri = Delaunay(src_xy)
    simplex = tri.find_simplex(tgt_xy)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = tgt_xy - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    return vertices, wts, simplex


def _interp_apply(values, vtx, wts, simplex, fill_value=np.nan):
    values = np.asarray(values).ravel()
    out = np.einsum('nj,nj->n', values[vtx], wts)
    out[simplex == -1] = fill_value
    return out


def _load_decadal_tf(gcm, scenario, start_year, end_year):
    tf_dir = f'{FORCING_ROOT}/{gcm}/{scenario}/ocean/tf/v3'
    files = sorted(glob.glob(f'{tf_dir}/tf_AIS_{gcm}_{scenario}_ocean_v3_*.nc'))
    wanted = []
    for f in files:
        y0, y1 = (int(v) for v in Path(f).stem.split('_')[-1].split('-'))
        if y1 >= start_year and y0 <= end_year:
            wanted.append(f)
    if not wanted:
        raise FileNotFoundError(f"no tf chunks found for {gcm}/{scenario} in {tf_dir}")
    return xr.open_mfdataset(sorted(wanted), combine='by_coords')['tf']


def _load_annual_smb(gcm, scenario, smb_param, start_year, end_year):
    smb_dir = f'{FORCING_ROOT}/{gcm}/{scenario}/{smb_param}/acabf/v2'
    files = []
    for yr in range(start_year, end_year + 1):
        matches = glob.glob(f'{smb_dir}/acabf_AIS_{gcm}_{scenario}_{smb_param}_v2_{yr}.nc')
        if matches:
            files.append(matches[0])
    if not files:
        raise FileNotFoundError(f"no acabf files found for {gcm}/{scenario}/{smb_param} in {smb_dir}")
    return xr.open_mfdataset(sorted(files), combine='by_coords')['acabf']


print('-- Loading SSA mesh (from AIS3_historical_1995_2019_SSA.nc)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv = md.mesh.numberofvertices
print(f'   nv={nv}')

print(f'-- Loading {GCM}/{SCENARIO} forcing files ({START_YEAR}-{END_YEAR})...', flush=True)
tf_da = _load_decadal_tf(GCM, SCENARIO, START_YEAR, END_YEAR)
smb_da = _load_annual_smb(GCM, SCENARIO, SMB_PARAM, START_YEAR, END_YEAR)

proj_years = np.arange(START_YEAR, END_YEAR + 1)

print('-- SMB: building interpolation weights + applying per year...', flush=True)
SECONDS_PER_YEAR = 365.0 * 24 * 3600
smb_xx, smb_yy = np.meshgrid(smb_da['x'].values, smb_da['y'].values)
t0 = time.time()
smb_vtx, smb_wts, smb_simplex = _interp_weights(smb_xx, smb_yy, md.mesh.x, md.mesh.y)
t1 = time.time()
print(f'   weight build: {t1 - t0:.2f}s')

mb_arr = np.empty((nv + 1, proj_years.size))
for i, yr in enumerate(proj_years):
    smb_yr = smb_da.sel(time=smb_da['time.year'] == yr)
    assert smb_yr.sizes['time'] == 12, f'expected 12 months for {yr}, got {smb_yr.sizes["time"]}'
    t2 = time.time()
    smb_yr_myr = (smb_yr.mean('time') * SECONDS_PER_YEAR / md.materials.rho_ice).to_numpy()
    mb_arr[:nv, i] = _interp_apply(smb_yr_myr, smb_vtx, smb_wts, smb_simplex)
    t3 = time.time()
    print(f'   {yr}: mesh-mean SMB = {np.nanmean(mb_arr[:nv, i]):.3f} m ice eq/yr '
          f'(NaN={np.isnan(mb_arr[:nv, i]).sum()}, apply took {t3 - t2:.3f}s)')
mb_arr[nv, :] = proj_years.astype(float)
print(f'   SMB sanity: min={np.nanmin(mb_arr[:nv]):.3f} max={np.nanmax(mb_arr[:nv]):.3f} '
      f'(compare RACMO historical mesh-mean ~0.5 m/yr from the RACMO loop\'s own printed range)')

print('\n-- TF: building interpolation weights + applying per depth/year...', flush=True)
tf_z = tf_da['z'].values
tf_xx, tf_yy = np.meshgrid(tf_da['x'].values, tf_da['y'].values)
t0 = time.time()
tf_vtx, tf_wts, tf_simplex = _interp_weights(tf_xx, tf_yy, md.mesh.x, md.mesh.y)
t1 = time.time()
print(f'   weight build: {t1 - t0:.2f}s')

tf_years = tf_da['time.year'].values
year_mask = (tf_years >= START_YEAR) & (tf_years <= END_YEAR)
tf_years_used = tf_years[year_mask]
print(f'   years selected from decadal chunk: {tf_years_used}')
tf_da_used = tf_da.isel(time=np.where(year_mask)[0])

t_apply_total = 0.0
n_applied = 0
tf_list = []
for k, zk in enumerate(tf_z):
    tf_depth_vals = tf_da_used.isel(z=k).to_numpy()
    col_arr = np.empty((nv + 1, tf_years_used.size))
    for j in range(tf_years_used.size):
        t2 = time.time()
        col_arr[:nv, j] = _interp_apply(tf_depth_vals[j], tf_vtx, tf_wts, tf_simplex)
        t_apply_total += time.time() - t2
        n_applied += 1
    col_arr[nv, :] = tf_years_used.astype(float)
    tf_list.append(col_arr)
    if k in (0, len(tf_z) // 2, len(tf_z) - 1):
        print(f'      depth z={zk:.0f}m: tf range = [{np.nanmin(col_arr[:nv]):.3f}, '
              f'{np.nanmax(col_arr[:nv]):.3f}] degC, NaN={np.isnan(col_arr[:nv]).sum()}')

print(f'\n   {n_applied} depth/year interpolations applied, total {t_apply_total:.2f}s '
      f'({1000 * t_apply_total / n_applied:.1f}ms/call average) -- projected full-scale '
      f'cost (30 depths x 81 years = 2430 calls): '
      f'{2430 * t_apply_total / n_applied / 60:.1f} minutes')

md.basalforcings.tf = tf_list
md.basalforcings.tf_depths = tf_z.copy()
print('\nDONE -- interpolation plumbing validated, no full transient solve run.')
