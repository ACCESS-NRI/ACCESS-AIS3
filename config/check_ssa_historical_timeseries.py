"""Build a genuine year-by-year dH/dt time series comparison for the SSA historical run,
rather than the single spatial-snapshot RMSE the production script reports (which compares
a 24-year SIMULATED MEAN rate against only dhdt_cpom's LAST (2019) time slice, not an
actual multi-year trajectory match). AIS3_historical_1995_2019_SSA.nc is a single-shot
(not chunked) run, so its full 240-timestep TransientSolution is genuinely available --
unlike the HO track's chunked checkpoints, which only carry the final state of each chunk
forward.

Domain-mean (grounded-ice) dH/dt per year, both simulated and observed, plotted together.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'

print('-- Loading AIS3_historical_1995_2019_SSA.nc (full timestep history)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv = md.mesh.numberofvertices
ts = md.results.TransientSolution

# BUGFIX (found on this exact run): this checkpoint's TransientSolution comes back as the
# same "collapsed bare object holding the full history flattened" quirk found in
# check_ho_relaxed.py / check_historical_chunk1.py -- but unlike those cases (which only
# needed the FINAL timestep), this script needs EVERY timestep's Thickness. Handle both
# shapes uniformly by reshaping Thickness to (n_steps, nv) either way. Simulation times are
# derived directly from the known timestepping config (0.1yr steps, 1995->2019) rather than
# trusted from the object's own possibly-ambiguous per-step time field.
start_t = float(md.timestepping.start_time)
final_t = float(md.timestepping.final_time)
dt = float(md.timestepping.time_step)
n_steps_expected = round((final_t - start_t) / dt)

if isinstance(ts, list):
    print(f'   {len(ts)} timesteps saved (list)')
    H_stack = np.stack([np.asarray(t.Thickness).ravel() for t in ts])
else:
    raw = np.asarray(ts.Thickness).ravel()
    n_steps = raw.size // nv
    print(f'   collapsed object holds {n_steps} timesteps flattened (size={raw.size}, nv={nv})')
    H_stack = raw.reshape(n_steps, nv)

assert H_stack.shape[0] == n_steps_expected, (
    f'expected {n_steps_expected} steps from timestepping config, got {H_stack.shape[0]}')
sim_times = start_t + dt * np.arange(1, H_stack.shape[0] + 1)  # step j ends at t=start+dt*(j+1)
print(f'   sim_times range: {sim_times.min():.2f} to {sim_times.max():.2f} '
      f'({H_stack.shape[0]} steps @ {dt}yr)')

il = np.asarray(md.mask.ice_levelset).ravel()
ol = np.asarray(md.mask.ocean_levelset).ravel()
gr = (il < 0) & (ol > 0)  # static current-final grounded mask -- same simplification the
# production RMSE diagnostic already uses (grounding state does evolve over 24 years, but
# using one fixed mask keeps this a clean, direct extension of the existing methodology
# rather than a new one).
print(f'   {gr.sum()} of {nv} vertices classified grounded (for domain-mean averaging)')

# Simulated grounded-mean thickness at each saved timestep.
H_all = H_stack[:, gr].mean(axis=1)

# Build one grounded-mean dH/dt value per calendar year 1996-2019 (need a full year of lead
# time from 1995 to get a first difference) as (H(year) - H(year-1)) / 1yr, using the saved
# timestep nearest each integer year boundary.
years_avail = np.arange(1996, 2020)
sim_dhdt_by_year = []
for yr in years_avail:
    idx_now = np.argmin(np.abs(sim_times - yr))
    idx_prev = np.argmin(np.abs(sim_times - (yr - 1)))
    dt_actual = sim_times[idx_now] - sim_times[idx_prev]
    sim_dhdt_by_year.append((H_all[idx_now] - H_all[idx_prev]) / dt_actual)
sim_dhdt_by_year = np.array(sim_dhdt_by_year)

print('\n-- Loading MIPKIT dhdt_cpom (ALL 27 years, not just the last one)...', flush=True)
mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
obs_years_all = mipkit['cpom_dhdt_time'].dt.year.values
mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()

obs_dhdt_by_year = []
for yr in years_avail:
    match = np.where(obs_years_all == yr)[0]
    if match.size == 0:
        obs_dhdt_by_year.append(np.nan)
        continue
    ti = int(match[0])
    obs_grid = mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=ti)
    obs_mesh = pyissm.data.interp.xr_to_mesh(
        obs_grid, 'dhdt_cpom', mesh_x, mesh_y, x_var='x1km', y_var='y1km')
    obs_dhdt_by_year.append(np.nanmean(obs_mesh[gr]))
    print(f'   {yr}: obs grounded-mean dH/dt = {obs_dhdt_by_year[-1]:.4f} m/yr')
obs_dhdt_by_year = np.array(obs_dhdt_by_year)

print('\n=== Year-by-year comparison (grounded-mean, m/yr) ===')
print(f'{"year":>6} {"sim":>10} {"obs":>10} {"diff":>10}')
for yr, s, o in zip(years_avail, sim_dhdt_by_year, obs_dhdt_by_year):
    diff = s - o if np.isfinite(o) else np.nan
    print(f'{yr:6d} {s:10.4f} {o:10.4f} {diff:10.4f}')

valid = np.isfinite(obs_dhdt_by_year)
rmse_ts = np.sqrt(np.nanmean((sim_dhdt_by_year[valid] - obs_dhdt_by_year[valid]) ** 2))
corr = np.corrcoef(sim_dhdt_by_year[valid], obs_dhdt_by_year[valid])[0, 1]
print(f'\n=== Time-series RMSE (grounded-mean, year-by-year) = {rmse_ts:.4f} m/yr ===')
print(f'=== Correlation (sim vs obs, year-by-year) = {corr:.3f} ===')
print('(compare against the production script\'s single-snapshot spatial RMSE=1.402 m/yr,')
print(' which is a DIFFERENT quantity: per-vertex spatial RMSE of a 24yr sim-mean rate vs')
print(' only dhdt_cpom\'s 2019 slice -- not a temporal comparison at all.)')

print('\n-- Plotting time series...', flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years_avail, sim_dhdt_by_year, 'o-', label='Simulated (SSA)', color='C0')
ax.plot(years_avail[valid], obs_dhdt_by_year[valid], 's-', label='Observed (CPOM)', color='C1')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Grounded-mean dH/dt (m/yr)')
ax.set_title(f'SSA historical run: simulated vs observed dH/dt time series\n'
             f'time-series RMSE={rmse_ts:.4f} m/yr, correlation={corr:.3f}')
ax.legend()
plt.tight_layout()
out_png = f'{MODEL_DIR}/ssa_historical_dhdt_timeseries.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
print('DONE')
