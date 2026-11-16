"""Area-weighted version of check_ssa_historical_timeseries.py.

The first version used an UNWEIGHTED vertex mean, which overweights the finely-resolved
margin/fast-flow regions of the SSA mesh, and averaged sim over all grounded vertices but obs
over only the vertices where dhdt_cpom is finite (mismatched coverage). Here both sim and obs
use the SAME vertex set each year (grounded AND obs finite) and are weighted by each
vertex's control-volume area (element area / 3 summed over adjacent elements), so the result
is a true area-mean. Also reports the unweighted common-mask result for direct comparison, and
converts the area-mean to an integrated grounded mass rate (Gt/yr) as an informational scale.
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

print('-- Loading AIS3_historical_1995_2019_SSA.nc...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_historical_1995_2019_SSA.nc')
nv = md.mesh.numberofvertices
ts = md.results.TransientSolution
start_t = float(md.timestepping.start_time)
final_t = float(md.timestepping.final_time)
dt = float(md.timestepping.time_step)
n_steps_expected = round((final_t - start_t) / dt)

if isinstance(ts, list):
    H_stack = np.stack([np.asarray(t.Thickness).ravel() for t in ts])
else:
    raw = np.asarray(ts.Thickness).ravel()
    H_stack = raw.reshape(raw.size // nv, nv)
assert H_stack.shape[0] == n_steps_expected
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
print(f'   total mesh area = {vertex_area.sum() / 1e12:.3f} million km^2')

il = np.asarray(md.mask.ice_levelset).ravel()
ol = np.asarray(md.mask.ocean_levelset).ravel()
gr = (il < 0) & (ol > 0)
print(f'   grounded vertices: {gr.sum()} of {nv} ({100 * gr.sum() / nv:.1f}% of vertices) '
      f'but {100 * vertex_area[gr].sum() / vertex_area.sum():.1f}% of mesh area '
      f'(grounded area = {vertex_area[gr].sum() / 1e12:.3f} million km^2)')
rho_ice = float(md.materials.rho_ice)

years_avail = np.arange(1996, 2020)
mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
obs_years_all = mipkit['cpom_dhdt_time'].dt.year.values

rows = []
for yr in years_avail:
    idx_now = np.argmin(np.abs(sim_times - yr))
    idx_prev = np.argmin(np.abs(sim_times - (yr - 1)))
    dt_actual = sim_times[idx_now] - sim_times[idx_prev]
    sim_v = (H_stack[idx_now] - H_stack[idx_prev]) / dt_actual  # per-vertex dH/dt

    ti = int(np.where(obs_years_all == yr)[0][0])
    obs_grid = mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=ti)
    obs_v = pyissm.data.interp.xr_to_mesh(
        obs_grid, 'dhdt_cpom', mesh_x, mesh_y, x_var='x1km', y_var='y1km')

    common = gr & np.isfinite(obs_v) & np.isfinite(sim_v)
    w = vertex_area[common]
    sim_w = np.sum(w * sim_v[common]) / w.sum()
    obs_w = np.sum(w * obs_v[common]) / w.sum()
    sim_u = sim_v[common].mean()
    obs_u = obs_v[common].mean()
    area_m2 = w.sum()
    # area-mean dH/dt (m/yr) * area (m^2) * rho_ice (kg/m^3) / 1e12 -> Gt/yr
    sim_gt = sim_w * area_m2 * rho_ice / 1e12
    obs_gt = obs_w * area_m2 * rho_ice / 1e12
    rows.append((yr, sim_w, obs_w, sim_u, obs_u, sim_gt, obs_gt, common.sum(), area_m2 / 1e12))

rows = np.array(rows)
yrs, sim_w, obs_w, sim_u, obs_u, sim_gt, obs_gt, ncommon, area_mkm2 = rows.T

print('\n=== Area-weighted, common-mask (grounded & obs finite) ===')
print(f'{"year":>6} {"sim_w":>9} {"obs_w":>9} {"diff":>9} | {"sim_unw":>9} {"obs_unw":>9} | '
      f'{"sim_Gt/yr":>10} {"obs_Gt/yr":>10} | {"n_vtx":>8} {"area_Mkm2":>10}')
for r in rows:
    print(f'{int(r[0]):6d} {r[1]:9.4f} {r[2]:9.4f} {r[1]-r[2]:9.4f} | {r[3]:9.4f} {r[4]:9.4f} | '
          f'{r[5]:10.1f} {r[6]:10.1f} | {int(r[7]):8d} {r[8]:10.3f}')

rmse_w = np.sqrt(np.mean((sim_w - obs_w) ** 2))
corr_w = np.corrcoef(sim_w, obs_w)[0, 1]
rmse_u = np.sqrt(np.mean((sim_u - obs_u) ** 2))
bias_w = np.mean(sim_w - obs_w)
print(f'\nArea-weighted:  RMSE={rmse_w:.4f} m/yr, mean bias (sim-obs)={bias_w:.4f} m/yr, corr={corr_w:.3f}')
print(f'Unweighted (same common mask): RMSE={rmse_u:.4f} m/yr')
print(f'Mean 1996-2019 area-weighted: sim={sim_w.mean():.4f} m/yr, obs={obs_w.mean():.4f} m/yr')
print(f'Mean 1996-2019 integrated over grounded common area: sim={sim_gt.mean():.1f} Gt/yr, '
      f'obs={obs_gt.mean():.1f} Gt/yr')
print('NOTE: obs dhdt_cpom is NOT firn-corrected (per its own attrs), so obs Gt/yr here is an')
print('apparent-thickness-change mass rate, not directly comparable to IMBIE mass-balance totals.')

print('\n-- Plotting...', flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.plot(yrs, sim_w, 'o-', label='Simulated (SSA)', color='C0')
ax.plot(yrs, obs_w, 's-', label='Observed (CPOM)', color='C1')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Year'); ax.set_ylabel('Area-weighted grounded-mean dH/dt (m/yr)')
ax.set_title(f'Area-weighted: RMSE={rmse_w:.3f} m/yr, bias={bias_w:+.3f}, corr={corr_w:.2f}')
ax.legend()
ax = axes[1]
ax.plot(yrs, sim_gt, 'o-', label='Simulated (SSA)', color='C0')
ax.plot(yrs, obs_gt, 's-', label='Observed (CPOM, non-firn-corrected)', color='C1')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Year'); ax.set_ylabel('Integrated grounded mass rate (Gt/yr)')
ax.set_title('Same, integrated over the common grounded area')
ax.legend()
plt.tight_layout()
out_png = f'{MODEL_DIR}/ssa_historical_dhdt_timeseries_areaweighted.png'
plt.savefig(out_png, dpi=150)
print(f'Saved {out_png}')
print('DONE')
