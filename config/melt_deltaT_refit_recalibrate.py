"""Recompute J1/J2 (and report on J3) using the per-basin-refit deltaT_basin from
melt_deltaT_basin_refit.py (8/9 candidates fully converged -- see
models/deltaT_refit_state.json), and rerun the Monte Carlo calibration.

IMPORTANT CONCEPTUAL POINT: with deltaT_basin refit per-basin to match the Paolo/Adusumilli
target exactly, J1 (basin-aggregated MAE) is now ~0 for every converged candidate BY
CONSTRUCTION -- it's no longer a discriminator between gamma_0 candidates the way it was
when deltaT_basin was held fixed. The real information now lives in J2 (does the WITHIN-basin
spatial pattern -- which deltaT_basin, being spatially uniform per basin, cannot correct --
still match the BFRN-bin-distributed target?) and J3 (warming sensitivity). J1 is kept in the
objective for completeness/protocol-shape but expect it to carry little weight in practice.

J3 is NOT recomputed here -- doing so honestly needs 2 more forward solves per candidate
(Mathiot cold/warm) under that candidate's OWN refit deltaT_basin (18 more hugemem solves),
which hasn't been run. This script reports J1(refit)/J2(refit) and, separately, flags J3 as
stale (still the fixed-deltaT value) rather than silently mixing a refit metric with a
stale one in one number without saying so.
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
STATE_FILE = f'{MODEL_DIR}/deltaT_refit_state.json'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
BFRN_FILE = f'{ISMIP7_OCEAN}/bfrns/BFRN_ismip8km_v2.nc'
MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'

# Stale (fixed-deltaT) J3 values, carried over from melt_ismip7_calibration.py's corrected
# (post-units-bugfix) run -- see docs/post_inversion_pipeline_worklog.md SS4a/4b.
STALE_J3 = {
    100.0: 489.562, 300.0: 466.502, 1000.0: 385.794, 3000.0: 260.697, 5537.7: 360.562,
    8306.6: 584.885, 11075.45: 864.318, 15000.0: 1293.889, 25000.0: 2404.799,
}


def cluster_config():
    cl = pyissm.model.classes.cluster.gadi()
    cl.codepath = os.environ['ISSM_DIR'] + '/bin'
    cl.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
    cl.storage = 'gdata/au88+gdata/vk83+gdata/av17'
    cl.moduleuse = ['/g/data/vk83/modules/']
    cl.moduleload = ['access-issm_ad/2026.05.0']
    cl.login = 'jh7060'
    cl.project = 'au88'
    cl.queue = 'hugemem'
    cl.np = 96
    cl.memory = 1450 * 2
    cl.time = 60 * 24
    return cl


print('-- Loading mesh/basin/BFRN geometry (from AIS3_ho_friction_inv.nc)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
x2d = np.asarray(md.mesh.x2d).ravel()
y2d = np.asarray(md.mesh.y2d).ravel()
elx2d = np.asarray(md.mesh.elements2d).astype(int) - 1
tri_x, tri_y = x2d[elx2d], y2d[elx2d]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
vertex_area_2d = np.zeros(md.mesh.numberofvertices2d)
for c in range(3):
    np.add.at(vertex_area_2d, elx2d[:, c], elem_area / 3.0)

ds_basin = xr.open_dataset(BASIN_FILE)
basin_grid = ds_basin['basinNumber'].values
bx, by = ds_basin['x'].values, ds_basin['y'].values
num_basins = int(np.nanmax(basin_grid)) + 1
elx = np.asarray(md.mesh.elements2d).astype(int) - 1
vx2d, vy2d = x2d, y2d
ecx = vx2d[elx[:, :3]].mean(axis=1)
ecy = vy2d[elx[:, :3]].mean(axis=1)
bxx, byy = np.meshgrid(bx, by)
basin_lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
basin_id_2d_elements = (basin_lookup(np.column_stack([ecx, ecy])) + 1).astype(float)
counts = np.zeros(md.mesh.numberofvertices2d)
basin_id_2d_vtx = np.zeros(md.mesh.numberofvertices2d)
for c in range(3):
    np.add.at(basin_id_2d_vtx, elx2d[:, c], basin_id_2d_elements - 1)
    np.add.at(counts, elx2d[:, c], 1)
basin_id_2d_vtx = np.round(basin_id_2d_vtx / np.maximum(counts, 1)).astype(int)

ol = np.asarray(md.mask.ocean_levelset).ravel()
floating_2d = ol[on_base] < 0
rho_ice = float(md.materials.rho_ice)

ds_bfrn = xr.open_dataset(BFRN_FILE)
bfrn_bins_grid = ds_bfrn['BFRN_bins'].values
bfrn_medians = ds_bfrn['BFRN_medians'].values
bfrn_median = float(ds_bfrn['BFRN_median'].values)
bx_bfrn, by_bfrn = ds_bfrn['x'].values, ds_bfrn['y'].values
bxx_b, byy_b = np.meshgrid(bx_bfrn, by_bfrn)
bin_lookup = NearestNDInterpolator(np.column_stack([bxx_b.ravel(), byy_b.ravel()]), bfrn_bins_grid.ravel())
vertex_bin_2d = bin_lookup(np.column_stack([x2d, y2d]))
n_bfrn_bins = int(np.nanmax(bfrn_bins_grid[np.isfinite(bfrn_bins_grid)])) + 1

obs_df = pd.read_csv(MELT_OBS_CSV)
bmb_obs_mean = obs_df['BMR (Gt/yr)'].values

weights_bfrn = bfrn_medians[np.clip(np.arange(n_bfrn_bins), 0, len(bfrn_medians) - 1)] / bfrn_median


def basin_bin_area_matrix():
    M = np.zeros((num_basins, n_bfrn_bins))
    for b in range(num_basins):
        mb = (basin_id_2d_vtx == b) & floating_2d
        for k in range(n_bfrn_bins):
            mk = mb & (vertex_bin_2d == k)
            M[b, k] = vertex_area_2d[mk].sum()
    return M


area_bb = basin_bin_area_matrix()
basin_area_tot = area_bb.sum(axis=1, keepdims=True)
basin_area_tot[basin_area_tot == 0] = 1.0
bin_obs_from_basin = (bmb_obs_mean[:, None] * area_bb / basin_area_tot).sum(axis=0)


def basin_aggregated_bmb(melt_myr_2d):
    bmb = np.zeros(num_basins)
    mass_flux = melt_myr_2d * rho_ice * vertex_area_2d / 1e12
    for b in range(num_basins):
        mask = (basin_id_2d_vtx == b) & floating_2d
        bmb[b] = np.nansum(mass_flux[mask])
    return bmb


def bin_aggregated_bmb(melt_myr_2d):
    bmb_bin = np.zeros(n_bfrn_bins)
    mass_flux = melt_myr_2d * rho_ice * vertex_area_2d / 1e12
    for k in range(n_bfrn_bins):
        mask = (vertex_bin_2d == k) & floating_2d
        bmb_bin[k] = np.nansum(mass_flux[mask])
    return bmb_bin


print(f'-- Loading refit state ({STATE_FILE})...', flush=True)
with open(STATE_FILE) as f:
    state = json.load(f)

cl = cluster_config()
print(f'\n{"i":>2} {"gamma_0":>10} {"round":>6} {"J1(refit)":>10} {"J2(refit)":>10} {"J3(stale)":>10}')
members = []
for i_str, cand in state['candidates'].items():
    i = int(i_str)
    g0 = cand['gamma_0']
    final_round = len(cand['history']) - 1
    name = f'AIS3_deltaT_refit_g{i}_r{final_round}'
    try:
        mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
        mdi.miscellaneous.name = name
        mdi.cluster = cl
        mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only=True, runtime_name=False)
    except Exception as e:
        print(f'   g{i} (gamma_0={g0:.1f}): FAILED to load {name} ({e})')
        continue
    melt = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
    melt_base = melt[on_base]
    bmb = basin_aggregated_bmb(melt_base)
    bmb_bin = bin_aggregated_bmb(melt_base)
    J1 = np.nanmean(np.abs(bmb - bmb_obs_mean))
    J2 = np.nanmean(weights_bfrn * np.abs(bmb_bin - bin_obs_from_basin))
    J3 = STALE_J3.get(g0, np.nan)
    print(f'{i:2d} {g0:10.1f} {final_round:6d} {J1:10.4f} {J2:10.3f} {J3:10.3f}')
    members.append({'i': i, 'gamma_0': g0, 'J1': J1, 'J2': J2, 'J3': J3, 'converged': cand['converged']})

medJ1 = np.median([m['J1'] for m in members])
medJ2 = np.median([m['J2'] for m in members])
medJ3 = np.median([m['J3'] for m in members])
print(f'\nMedians: J1={medJ1:.4f}  J2={medJ2:.3f}  J3={medJ3:.3f}')
print('NOTE: J1 is near-zero for every converged candidate by construction (deltaT_basin was '
      'fit to make it so) -- expect it to contribute little real discriminating signal now. '
      'J3 is STALE (fixed-deltaT value, not recomputed under refit deltaT_basin).')

print('\n-- Monte Carlo sampling objective weights (10000 samples, J1/J2/J3(stale))...', flush=True)
rng = np.random.default_rng(42)
n_samples = 10000
raw = rng.uniform(size=(n_samples, 3))
w = raw / raw.sum(axis=1, keepdims=True)
J1s = np.array([m['J1'] for m in members])
J2s = np.array([m['J2'] for m in members])
J3s = np.array([m['J3'] for m in members])
gammas = np.array([m['gamma_0'] for m in members])
I = (w[:, 0:1] * J1s[None, :] / max(medJ1, 1e-6)
     + w[:, 1:2] * J2s[None, :] / medJ2
     + w[:, 2:3] * J3s[None, :] / medJ3)
winners = gammas[np.argmin(I, axis=1)]
p5, p50, p95 = np.percentile(winners, [5, 50, 95])
print(f'\n=== Recalibrated result (refit deltaT_basin, J1/J2 refit + J3 stale) ===')
print(f'   5th percentile gamma_0:  {p5:.1f}')
print(f'   50th percentile gamma_0: {p50:.1f}')
print(f'   95th percentile gamma_0: {p95:.1f}')

print('\n-- Same Monte Carlo using ONLY J1+J2 (both refit, no stale J3 mixed in), for comparison...', flush=True)
I_j1j2 = (w[:, 0:1] * J1s[None, :] / max(medJ1, 1e-6) + w[:, 1:2] * J2s[None, :] / medJ2) / (w[:, 0:1] + w[:, 1:2])
winners_j1j2 = gammas[np.argmin(I_j1j2, axis=1)]
p5b, p50b, p95b = np.percentile(winners_j1j2, [5, 50, 95])
print(f'   5th/50th/95th percentile gamma_0 (J1+J2 only): {p5b:.1f} / {p50b:.1f} / {p95b:.1f}')
print('DONE')
