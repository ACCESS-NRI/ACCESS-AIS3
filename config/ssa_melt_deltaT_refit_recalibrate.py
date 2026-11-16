"""SSA-mesh version of melt_deltaT_refit_recalibrate.py -- recompute J1 (basin-aggregated)
AND J2 (BFRN-weighted bin fit, not computed by the original melt_gamma_tuning_ssa) using the
per-basin-refit deltaT_basin from ssa_melt_deltaT_basin_refit.py, then pick the best gamma_0.

J2 has to be added here that wasn't in the original SSA melt calibration (which was J1-only,
deliberately simplified): once deltaT_basin is refit per basin, J1 becomes ~0 for every
converged candidate by construction (same finding as HO) -- it stops discriminating, and J2
(the within-basin/BFRN-bin spatial pattern, which a spatially-uniform-per-basin deltaT_basin
cannot correct) becomes the only real signal left to pick a winner.

No Monte Carlo here (SSA never had J3 either) -- just report J1+J2 per candidate and pick the
minimum-J2 winner directly (matching HO's own finding that the two agree once J1 saturates).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import json
import numpy as np
import pandas as pd
import xarray as xr
import pyissm
from scipy.interpolate import NearestNDInterpolator

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
STATE_FILE = f'{MODEL_DIR}/deltaT_refit_state_ssa.json'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
BFRN_FILE = f'{ISMIP7_OCEAN}/bfrns/BFRN_ismip8km_v2.nc'
MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'

cluster = pyissm.model.classes.cluster.gadi()
cluster.codepath = os.environ['ISSM_DIR'] + '/bin'
cluster.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
cluster.storage = 'gdata/au88+gdata/vk83+gdata/av17'
cluster.moduleuse = ['/g/data/vk83/modules/']
cluster.moduleload = ['access-issm_ad/2026.05.0']
cluster.np = 48
cluster.memory = 190
cluster.time = 60 * 48
cluster.login = 'jh7060'
cluster.project = 'au88'

print('-- Loading mesh/basin/BFRN geometry (from AIS3_SSA_relaxed.nc)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_relaxed.nc')
md.results.TransientSolution = []
mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()
elx = np.asarray(md.mesh.elements).astype(int) - 1
tri_x, tri_y = mesh_x[elx], mesh_y[elx]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
vertex_area = np.zeros(md.mesh.numberofvertices)
for c in range(3):
    np.add.at(vertex_area, elx[:, c], elem_area / 3.0)

ds_basin = xr.open_dataset(BASIN_FILE)
basin_grid = ds_basin['basinNumber'].values
bx, by = ds_basin['x'].values, ds_basin['y'].values
num_basins = int(np.nanmax(basin_grid)) + 1
bxx, byy = np.meshgrid(bx, by)
basin_lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
ecx = mesh_x[elx[:, :3]].mean(axis=1)
ecy = mesh_y[elx[:, :3]].mean(axis=1)
basin_id_elements = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)
counts = np.zeros(md.mesh.numberofvertices)
basin_id_vtx = np.zeros(md.mesh.numberofvertices)
for c in range(3):
    np.add.at(basin_id_vtx, elx[:, c], basin_id_elements - 1)
    np.add.at(counts, elx[:, c], 1)
basin_id_vtx = np.round(basin_id_vtx / np.maximum(counts, 1)).astype(int)

ol = np.asarray(md.mask.ocean_levelset).ravel()
floating = ol < 0
rho_ice = float(md.materials.rho_ice)

ds_bfrn = xr.open_dataset(BFRN_FILE)
bfrn_bins_grid = ds_bfrn['BFRN_bins'].values
bfrn_medians = ds_bfrn['BFRN_medians'].values
bfrn_median = float(ds_bfrn['BFRN_median'].values)
bx_bfrn, by_bfrn = ds_bfrn['x'].values, ds_bfrn['y'].values
bxx_b, byy_b = np.meshgrid(bx_bfrn, by_bfrn)
bin_lookup = NearestNDInterpolator(np.column_stack([bxx_b.ravel(), byy_b.ravel()]), bfrn_bins_grid.ravel())
vertex_bin = bin_lookup(np.column_stack([mesh_x, mesh_y]))
n_bfrn_bins = int(np.nanmax(bfrn_bins_grid[np.isfinite(bfrn_bins_grid)])) + 1

obs_df = pd.read_csv(MELT_OBS_CSV)
bmb_obs_mean = obs_df['BMR (Gt/yr)'].values
weights_bfrn = bfrn_medians[np.clip(np.arange(n_bfrn_bins), 0, len(bfrn_medians) - 1)] / bfrn_median


def basin_bin_area_matrix():
    M = np.zeros((num_basins, n_bfrn_bins))
    for b in range(num_basins):
        mb = (basin_id_vtx == b) & floating
        for k in range(n_bfrn_bins):
            mk = mb & (vertex_bin == k)
            M[b, k] = vertex_area[mk].sum()
    return M


area_bb = basin_bin_area_matrix()
basin_area_tot = area_bb.sum(axis=1, keepdims=True)
basin_area_tot[basin_area_tot == 0] = 1.0
bin_obs_from_basin = (bmb_obs_mean[:, None] * area_bb / basin_area_tot).sum(axis=0)


def basin_aggregated_bmb(melt_myr):
    bmb = np.zeros(num_basins)
    mass_flux = melt_myr * rho_ice * vertex_area / 1e12
    for b in range(num_basins):
        mask = (basin_id_vtx == b) & floating
        bmb[b] = np.nansum(mass_flux[mask])
    return bmb


def bin_aggregated_bmb(melt_myr):
    bmb_bin = np.zeros(n_bfrn_bins)
    mass_flux = melt_myr * rho_ice * vertex_area / 1e12
    for k in range(n_bfrn_bins):
        mask = (vertex_bin == k) & floating
        bmb_bin[k] = np.nansum(mass_flux[mask])
    return bmb_bin


print(f'-- Loading refit state ({STATE_FILE})...', flush=True)
with open(STATE_FILE) as f:
    state = json.load(f)

print(f'\n{"i":>2} {"gamma_0":>10} {"round":>6} {"J1(refit)":>10} {"J2(refit)":>10}')
members = []
for i_str, cand in state['candidates'].items():
    i = int(i_str)
    g0 = cand['gamma_0']
    final_round = len(cand['history']) - 1
    name = f'AIS3_ssa_deltaT_refit_g{i}_r{final_round}'
    try:
        mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_SSA_relaxed.nc')
        mdi.results.TransientSolution = []
        mdi.miscellaneous.name = name
        mdi.cluster = cluster
        mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
    except Exception as e:
        print(f'   g{i} (gamma_0={g0:.1f}): FAILED to load {name} ({e})')
        continue
    melt = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
    bmb = basin_aggregated_bmb(melt)
    bmb_bin = bin_aggregated_bmb(melt)
    J1 = np.nanmean(np.abs(bmb - bmb_obs_mean))
    J2 = np.nanmean(weights_bfrn * np.abs(bmb_bin - bin_obs_from_basin))
    print(f'{i:2d} {g0:10.1f} {final_round:6d} {J1:10.4f} {J2:10.3f}')
    members.append({'i': i, 'gamma_0': g0, 'J1': J1, 'J2': J2, 'converged': cand['converged']})

if members:
    best = min(members, key=lambda m: m['J2'])
    print(f'\nBest by J2 (refit): gamma_0={best["gamma_0"]:.1f} (i={best["i"]}, J2={best["J2"]:.3f}, '
          f'J1={best["J1"]:.4f})')
print('DONE')
