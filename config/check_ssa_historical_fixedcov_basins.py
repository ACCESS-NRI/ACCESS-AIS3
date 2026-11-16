"""Fixed-coverage, per-basin follow-up to check_ssa_historical_timeseries_areaweighted.py.

Two limitations of the area-weighted whole-domain comparison are addressed here:
 1. Coverage changed over the record (CPOM coverage grows ~2011), so year-to-year means were
    over different areas. Here every year uses the SAME vertices: grounded AND dhdt_cpom finite
    in ALL years 1996-2019.
 2. A near-zero whole-domain mean can hide compensating regional errors. Here the same
    comparison is done per IMBIE2 basin (16 basins, same basin grid the melt calibration uses),
    plus a spatial-pattern test: area-weighted correlation between the simulated and observed
    per-vertex 1996-2019 linear TREND maps.
All means are area-weighted (vertex control-volume area).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyissm
from scipy.interpolate import NearestNDInterpolator

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
ISMIP7_OCEAN = '/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/parameterisations/ocean'
BASIN_FILE = f'{ISMIP7_OCEAN}/imbie2/basin_numbers_ismip8km_v2.nc'
MELT_OBS_CSV = f'{ISMIP7_OCEAN}/meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv'

print('-- Loading AIS3_historical_1995_2019_SSA.nc...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv = md.mesh.numberofvertices
ts = md.results.TransientSolution
start_t = float(md.timestepping.start_time)
final_t = float(md.timestepping.final_time)
dt = float(md.timestepping.time_step)
if isinstance(ts, list):
    H_stack = np.stack([np.asarray(t.Thickness).ravel() for t in ts])
else:
    raw = np.asarray(ts.Thickness).ravel()
    H_stack = raw.reshape(raw.size // nv, nv)
assert H_stack.shape[0] == round((final_t - start_t) / dt)
sim_times = start_t + dt * np.arange(1, H_stack.shape[0] + 1)

mesh_x = np.asarray(md.mesh.x).ravel()
mesh_y = np.asarray(md.mesh.y).ravel()
elx = np.asarray(md.mesh.elements).astype(int) - 1
tri_x, tri_y = mesh_x[elx], mesh_y[elx]
elem_area = 0.5 * np.abs((tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0])
                          - (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0]))
vertex_area = np.zeros(nv)
for c in range(3):
    np.add.at(vertex_area, elx[:, c], elem_area / 3.0)

il = np.asarray(md.mask.ice_levelset).ravel()
ol = np.asarray(md.mask.ocean_levelset).ravel()
gr = (il < 0) & (ol > 0)
rho_ice = float(md.materials.rho_ice)

# IMBIE2 basin per vertex, same construction as the melt-calibration scripts (element-centre
# lookup, averaged onto vertices).
ds_basin = xr.open_dataset(BASIN_FILE)
basin_grid = ds_basin['basinNumber'].values
bx, by = ds_basin['x'].values, ds_basin['y'].values
bxx, byy = np.meshgrid(bx, by)
lookup = NearestNDInterpolator(np.column_stack([bxx.ravel(), byy.ravel()]), basin_grid.ravel())
ecx = mesh_x[elx[:, :3]].mean(axis=1)
ecy = mesh_y[elx[:, :3]].mean(axis=1)
basin_el = lookup(np.column_stack([ecx, ecy])).astype(float)
acc = np.zeros(nv); cnt = np.zeros(nv)
for c in range(3):
    np.add.at(acc, elx[:, c], basin_el)
    np.add.at(cnt, elx[:, c], 1)
basin_v = np.round(acc / np.maximum(cnt, 1)).astype(int)
num_basins = int(np.nanmax(basin_grid)) + 1
try:
    names = pd.read_csv(MELT_OBS_CSV).iloc[:, 0].astype(str).tolist()
except Exception:
    names = [str(b) for b in range(num_basins)]
if len(names) != num_basins:
    names = [str(b) for b in range(num_basins)]

years = np.arange(1996, 2020)
mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
obs_years_all = mipkit['cpom_dhdt_time'].dt.year.values

print('-- Building per-vertex sim and obs dH/dt for every year...', flush=True)
sim_stack = np.empty((years.size, nv), dtype=np.float32)
obs_stack = np.empty((years.size, nv), dtype=np.float32)
for k, yr in enumerate(years):
    i_now = np.argmin(np.abs(sim_times - yr))
    i_prev = np.argmin(np.abs(sim_times - (yr - 1)))
    sim_stack[k] = (H_stack[i_now] - H_stack[i_prev]) / (sim_times[i_now] - sim_times[i_prev])
    ti = int(np.where(obs_years_all == yr)[0][0])
    obs_grid = mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=ti)
    obs_stack[k] = pyissm.data.interp.xr_to_mesh(
        obs_grid, 'dhdt_cpom', mesh_x, mesh_y, x_var='x1km', y_var='y1km')

fixed = gr & np.all(np.isfinite(obs_stack), axis=0) & np.all(np.isfinite(sim_stack), axis=0)
print(f'   fixed-coverage vertices (grounded, obs finite in ALL {years.size} years): {fixed.sum()} '
      f'= {vertex_area[fixed].sum() / 1e12:.3f} Mkm^2 '
      f'({100 * vertex_area[fixed].sum() / vertex_area[gr].sum():.1f}% of grounded area)')


def wmean(arr2d, mask):
    w = vertex_area[mask]
    return (arr2d[:, mask] * w).sum(axis=1) / w.sum()


def summarize(sim_ts, obs_ts):
    rmse = np.sqrt(np.mean((sim_ts - obs_ts) ** 2))
    corr = np.corrcoef(sim_ts, obs_ts)[0, 1] if sim_ts.std() > 0 and obs_ts.std() > 0 else np.nan
    return rmse, corr, np.mean(sim_ts - obs_ts)


sim_all = wmean(sim_stack, fixed)
obs_all = wmean(obs_stack, fixed)
rm, co, bi = summarize(sim_all, obs_all)
area_all = vertex_area[fixed].sum()
print('\n=== Whole domain, FIXED coverage, area-weighted ===')
print(f'{"year":>6} {"sim":>9} {"obs":>9}')
for y, s, o in zip(years, sim_all, obs_all):
    print(f'{y:6d} {s:9.4f} {o:9.4f}')
print(f'RMSE={rm:.4f} m/yr  bias={bi:+.4f} m/yr  corr={co:.3f}')
print(f'Mean: sim={sim_all.mean():+.4f} obs={obs_all.mean():+.4f} m/yr  '
      f'(= {sim_all.mean() * area_all * rho_ice / 1e12:+.1f} vs '
      f'{obs_all.mean() * area_all * rho_ice / 1e12:+.1f} Gt/yr over the fixed area)')

# Per-vertex linear trend (m/yr per yr) over 1996-2019, then area-weighted spatial correlation
yc = years - years.mean()
sim_trend = (yc[:, None] * (sim_stack - sim_stack.mean(0))).sum(0) / (yc ** 2).sum()
obs_trend = (yc[:, None] * (obs_stack - obs_stack.mean(0))).sum(0) / (yc ** 2).sum()
sim_mean_v = sim_stack.mean(0)
obs_mean_v = obs_stack.mean(0)


def wcorr(a, b, mask):
    w = vertex_area[mask]; a = a[mask]; b = b[mask]
    ma = (w * a).sum() / w.sum(); mb = (w * b).sum() / w.sum()
    cov = (w * (a - ma) * (b - mb)).sum() / w.sum()
    return cov / np.sqrt((w * (a - ma) ** 2).sum() / w.sum() * (w * (b - mb) ** 2).sum() / w.sum())


print('\n=== Spatial-pattern test (fixed coverage, area-weighted correlation across vertices) ===')
print(f'   1996-2019 MEAN dH/dt map:  corr(sim, obs) = {wcorr(sim_mean_v, obs_mean_v, fixed):.3f}')
print(f'   1996-2019 TREND map:       corr(sim, obs) = {wcorr(sim_trend, obs_trend, fixed):.3f}')
print('   (a model with no regional skill would sit near 0; a whole-domain mean near 0 can hide')
print('    compensating regional errors, which these correlations would expose)')

print('\n=== Per basin, FIXED coverage, area-weighted ===')
print(f'{"b":>2} {"name":<22} {"area_Mkm2":>9} {"sim_mean":>9} {"obs_mean":>9} {"bias":>8} '
      f'{"RMSE":>8} {"corr":>7} {"sim_Gt/yr":>10} {"obs_Gt/yr":>10}')
basin_rows = []
for b in range(num_basins):
    m = fixed & (basin_v == b)
    if m.sum() < 50:
        continue
    s = wmean(sim_stack, m); o = wmean(obs_stack, m)
    rmse_b, corr_b, bias_b = summarize(s, o)
    a = vertex_area[m].sum()
    sg = s.mean() * a * rho_ice / 1e12
    og = o.mean() * a * rho_ice / 1e12
    basin_rows.append((b, names[b], a / 1e12, s, o, s.mean(), o.mean(), bias_b, rmse_b, corr_b, sg, og))
    print(f'{b:2d} {names[b][:22]:<22} {a / 1e12:9.3f} {s.mean():+9.4f} {o.mean():+9.4f} {bias_b:+8.4f} '
          f'{rmse_b:8.4f} {corr_b:7.2f} {sg:+10.1f} {og:+10.1f}')
sg_tot = sum(r[10] for r in basin_rows); og_tot = sum(r[11] for r in basin_rows)
print(f'   sum over basins: sim={sg_tot:+.1f} Gt/yr  obs={og_tot:+.1f} Gt/yr')
signs = [np.sign(r[5]) == np.sign(r[6]) for r in basin_rows]
print(f'   basins where mean sim and obs dH/dt have the SAME sign: {sum(signs)} of {len(signs)}')

print('\n-- Plotting...', flush=True)
n = len(basin_rows)
ncol = 4
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), sharex=True)
axes = np.atleast_1d(axes).ravel()
for ax, r in zip(axes, basin_rows):
    ax.plot(years, r[3], 'o-', ms=3, color='C0', label='sim')
    ax.plot(years, r[4], 's-', ms=3, color='C1', label='obs')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'{r[0]} {r[1][:18]}  corr={r[9]:.2f}', fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes[n:]:
    ax.axis('off')
axes[0].legend(fontsize=7)
fig.suptitle('SSA historical: fixed-coverage, area-weighted dH/dt per IMBIE2 basin (m/yr)', fontsize=10)
plt.tight_layout()
out_png = f'{MODEL_DIR}/ssa_historical_dhdt_basins_fixedcov.png'
plt.savefig(out_png, dpi=140)
print(f'Saved {out_png}')
print('DONE')
