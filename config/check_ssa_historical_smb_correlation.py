"""Is the SSA historical run's interannual dH/dt signal just an SMB response?

Correlates the area-weighted, grounded-mean simulated dH/dt (per year, 1996-2019) with the
area-weighted grounded-mean RACMO SMB the run was actually forced with (read back from the
saved model's own md.smb.mass_balance, so no separate RACMO load), at lags 0 and 1 year, in
levels and first differences. Also reports the dynamic residual (dH/dt - SMB, i.e. -div(flux)
and everything not surface mass balance) and correlates observed dH/dt with SMB for context.
Tests one candidate explanation for the sim/obs disagreement (sim variability being
SMB-driven while altimetry variability is dominated by other things).
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
w = vertex_area[gr]

mb = np.asarray(md.smb.mass_balance)
print(f'   md.smb.mass_balance shape = {mb.shape}')
assert mb.shape[0] == nv + 1, 'expected (nv+1, n_years) timeseries with time in the last row'
smb_years = mb[-1, :].astype(int)
smb_field = mb[:nv, :]
smb_area_mean = (smb_field[gr, :] * w[:, None]).sum(axis=0) / w.sum()
smb_by_year = dict(zip(smb_years, smb_area_mean))
print('   RACMO SMB area-mean over grounded (m ice eq/yr): '
      + ', '.join(f'{y}:{v:.3f}' for y, v in zip(smb_years, smb_area_mean)))

years = np.arange(1996, 2020)
sim = np.empty(years.size)
for k, yr in enumerate(years):
    i_now = np.argmin(np.abs(sim_times - yr))
    i_prev = np.argmin(np.abs(sim_times - (yr - 1)))
    d = (H_stack[i_now, gr] - H_stack[i_prev, gr]) / (sim_times[i_now] - sim_times[i_prev])
    sim[k] = (d * w).sum() / w.sum()

mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
obs_years_all = mipkit['cpom_dhdt_time'].dt.year.values
obs = np.empty(years.size)
for k, yr in enumerate(years):
    ti = int(np.where(obs_years_all == yr)[0][0])
    o = pyissm.data.interp.xr_to_mesh(mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=ti),
                                       'dhdt_cpom', mesh_x, mesh_y, x_var='x1km', y_var='y1km')
    m = gr & np.isfinite(o)
    obs[k] = (o[m] * vertex_area[m]).sum() / vertex_area[m].sum()

smb_lag0 = np.array([smb_by_year[y] for y in years])          # SMB of the same calendar year
smb_lag1 = np.array([smb_by_year[y - 1] for y in years])      # SMB of the preceding year
smb_avg = 0.5 * (smb_lag0 + smb_lag1)                         # the year's H(yr)-H(yr-1) spans both


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def dcorr(a, b):
    return np.corrcoef(np.diff(a), np.diff(b))[0, 1]


print('\n=== Year-by-year (area-weighted, grounded; m/yr) ===')
print(f'{"year":>6} {"sim dH/dt":>10} {"SMB(yr)":>9} {"SMB(yr-1)":>10} {"resid=dH-SMBavg":>16} {"obs dH/dt":>10}')
resid = sim - smb_avg
for y, s, a, b, r, o in zip(years, sim, smb_lag0, smb_lag1, resid, obs):
    print(f'{y:6d} {s:10.4f} {a:9.4f} {b:10.4f} {r:16.4f} {o:10.4f}')

print('\n=== Correlations (n=24) ===')
print(f'sim dH/dt vs SMB(same yr):   levels {corr(sim, smb_lag0):+.3f}   first-diff {dcorr(sim, smb_lag0):+.3f}')
print(f'sim dH/dt vs SMB(prev yr):   levels {corr(sim, smb_lag1):+.3f}   first-diff {dcorr(sim, smb_lag1):+.3f}')
print(f'sim dH/dt vs SMB(avg of both): levels {corr(sim, smb_avg):+.3f}   first-diff {dcorr(sim, smb_avg):+.3f}')
print(f'obs dH/dt vs SMB(avg):       levels {corr(obs, smb_avg):+.3f}   first-diff {dcorr(obs, smb_avg):+.3f}')
print(f'sim vs obs dH/dt:            levels {corr(sim, obs):+.3f}   first-diff {dcorr(sim, obs):+.3f}')
print(f'\nStd dev (m/yr): sim dH/dt={sim.std():.4f}  SMB(avg)={smb_avg.std():.4f}  '
      f'resid={resid.std():.4f}  obs={obs.std():.4f}')
print(f'Mean (m/yr):    sim dH/dt={sim.mean():+.4f}  SMB(avg)={smb_avg.mean():+.4f}  '
      f'resid(-div flux etc.)={resid.mean():+.4f}  obs={obs.mean():+.4f}')
yc = years - years.mean()


def slope(a):
    return (yc * (a - a.mean())).sum() / (yc ** 2).sum()


print(f'\nLinear trend 1996-2019 (m/yr per yr): sim={slope(sim):+.5f}  SMB(avg)={slope(smb_avg):+.5f}  '
      f'resid={slope(resid):+.5f}  obs={slope(obs):+.5f}')
print('Reading guide: if sim tracks SMB (high corr, resid roughly constant) its variability is a')
print('surface-forcing response; a rising resid trend means ice dynamics/discharge, not SMB, drives')
print('the sim trend (e.g. start-up adjustment). Compare against the obs trend/variance above.')

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(years, sim, 'o-', label='sim dH/dt', color='C0')
ax[0].plot(years, smb_avg, '^-', label='RACMO SMB (avg of yr, yr-1)', color='C2')
ax[0].plot(years, obs, 's-', label='obs dH/dt (CPOM)', color='C1')
ax[0].axhline(0, color='gray', lw=0.5)
ax[0].set_title('Area-weighted grounded means (m/yr)')
ax[0].legend(fontsize=8)
ax[1].plot(years, resid, 'o-', color='C3')
ax[1].axhline(0, color='gray', lw=0.5)
ax[1].set_title('Sim dynamic residual = dH/dt - SMB (m/yr)')
for a in ax:
    a.set_xlabel('Year')
plt.tight_layout()
out_png = f'{MODEL_DIR}/ssa_historical_smb_correlation.png'
plt.savefig(out_png, dpi=140)
print(f'Saved {out_png}')
print('DONE')
