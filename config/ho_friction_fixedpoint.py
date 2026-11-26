"""Forward-only friction refinement for the pin-free HO track in the formerly pinned regions (worklog 6.38).

The pin-free HO historical run (6.28, 6.35) uses the SSA track's Budd p=q=1 field. On the 1995 HO state the
formerly pinned grounded columns start ~30% too slow (speed/obs median 0.69, 6.27), and Ross and
Filchner-Ronne then thicken too much over 1995-2019 (6.35). This nudges C there towards the observed speed
with HO stress-balance solves only (no adjoint):
  Budd p=1 drag tau = C^2 N u, so at roughly fixed driving stress u ~ 1/C^2, and
  C_new = C_old * sqrt(u_model / u_obs)
on formerly pinned grounded vertices with u_obs > 10 m/yr; the ratio is smoothed over neighbouring vertices
(two passes of element averaging), limited to [0.67, 1.5] per iteration, and C kept in [0.05, 900] (the
inversion's bounds). Elsewhere C is unchanged. Iteration 0 speeds come from check_ho_unpin_1995.py's B95
solve (identical state and friction, models/ho_unpin_1995_test.npz).
State: models/AIS3_ho_relaxed.nc loaded as historical_dhdt_tuning does (tf fix, sync_mesh_z, results
dropped), friction swapped to the SSA Budd field with all pins released, basal melt off.
Writes models/ho_friction_fixedpoint.npz (C and surface speed per iteration; refuses to overwrite).
Run dirs execution/AIS3_ho_fricfp_it<k>.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import time
import numpy as np
import netCDF4
import pyissm

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
# AIS3_FP_CONTINUE=1: continue from the last iteration in models/ho_friction_fixedpoint.npz for NIT more
# iterations, writing models/ho_friction_fixedpoint_cont.npz (worklog 6.42).
CONT = os.environ.get('AIS3_FP_CONTINUE', '') == '1'
PREV = f'{M}/ho_friction_fixedpoint.npz'
OUT = f'{M}/ho_friction_fixedpoint_cont.npz' if CONT else PREV
NIT = 4 if CONT else 3
if CONT:
    _p = np.load(PREV)
    K0 = max(int(k[1:]) for k in _p.files if k.startswith('C') and k[1:].isdigit())
else:
    K0 = 0
CLIP = (0.67, 1.5)
CBOUNDS = (0.05, 900.0)
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')
for k in range(K0 + 1, K0 + NIT + 1):
    if os.path.exists(f'{EXE}/AIS3_ho_fricfp_it{k}'):
        raise SystemExit(f'{EXE}/AIS3_ho_fricfp_it{k} exists; move it aside first')


def sync_mesh_z(md):  # copied from ais_0.1.py (keep in sync)
    nv = md.mesh.numberofvertices
    nvb = int(md.mesh.numberofvertices2d)
    nl = nv // nvb
    z = np.asarray(md.mesh.z, dtype=float).ravel().reshape(nl, nvb)
    span = z[-1] - z[0]
    safe = np.abs(span) > 1e-9
    sigma = np.where(safe[None, :], (z - z[0]) / np.where(safe, span, 1.0)[None, :], np.linspace(0.0, 1.0, nl)[:, None])
    base = np.asarray(md.geometry.base, dtype=float).ravel()[:nvb]
    thick = np.asarray(md.geometry.thickness, dtype=float).ravel()[:nvb]
    md.mesh.z = (base[None, :] + sigma * thick[None, :]).ravel()


md = pyissm.model.io.load_model(f'{M}/AIS3_ho_relaxed.nc')
md.basalforcings.tf = [np.asarray(t, dtype=float) for t in md.basalforcings.tf]
sync_mesh_z(md)
md.results.TransientSolution = []
md.results.StressbalanceSolution = []
md.results.ThermalSolution = []
nv = md.mesh.numberofvertices
nvb = int(md.mesh.numberofvertices2d)
nl = nv // nvb
ne = md.mesh.numberofelements
il = np.asarray(md.mask.ice_levelset, float).ravel()[:nvb]
ol = np.asarray(md.mask.ocean_levelset, float).ravel()[:nvb]
gr = (il < 0) & (ol > 0)

# SSA Budd friction, no pins (as use_ssa_budd_friction_no_pins in ais_0.1.py)
s = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
sf = s['friction']
assert sf.getncattr('classtype') == 'friction.default' and int(sf.getncattr('coupling')) == 3
C = np.asarray(sf.variables['coefficient'][:], float).ravel().copy()
N2 = np.asarray(sf.variables['effective_pressure'][:], float).ravel()
lim = float(sf.getncattr('effective_pressure_limit'))
el2 = np.asarray(s['mesh'].variables['elements'][:]).astype(int) - 1
if el2.shape[0] == 3:
    el2 = el2.T
s.close()
md.friction = pyissm.model.classes.friction.default()
md.friction.p = np.ones(ne)
md.friction.q = np.ones(ne)
md.friction.coupling = 3
md.friction.linearize = 0
md.friction.effective_pressure = np.tile(N2, nl)
md.friction.effective_pressure_limit = lim
md.stressbalance.spcvx = np.full(nv, np.nan)
md.stressbalance.spcvy = np.full(nv, np.nan)
md.stressbalance.spcvz = np.full(nv, np.nan)
md.basalforcings = pyissm.model.classes.basalforcings.default(md.basalforcings)
md.basalforcings.groundedice_melting_rate = np.zeros(nv)
md.basalforcings.floatingice_melting_rate = np.zeros(nv)
md.inversion.iscontrol = 0
md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
md.stressbalance.requested_outputs = ['default']
md.settings.solver_residue_threshold = 1e-3
md.verbose.solution = 1
c = pyissm.model.classes.cluster.gadi()
c.codepath = os.environ['ISSM_DIR'] + '/bin'
c.executionpath = EXE
c.storage = 'gdata/au88+gdata/vk83+gdata/av17'
c.moduleuse = ['/g/data/vk83/modules/']
c.moduleload = ['access-issm_ad/2026.05.0']
c.login = 'jh7060'
c.project = 'au88'
c.queue, c.np, c.memory, c.time = 'hugemem', 96, 1450 * 2, 6 * 60
md.cluster = c
md.settings.waitonlock = 30 * 60

z0 = np.load(f'{M}/ho_unpin_1995_test.npz')
vobs, u = z0['vobs'], z0['surf_B95'].copy()
pin = z0['pin']
region = pin & gr & (vobs > 10)
print(f'region: {int(region.sum())} formerly pinned grounded vertices with obs > 10 m/yr', flush=True)
n_conn = np.bincount(el2.ravel(), minlength=nvb).astype(float)


def smooth(f, mask, passes=2):
    g = f.copy()
    for _ in range(passes):
        e = g[el2].mean(axis=1)
        v = np.bincount(el2.ravel(), weights=np.repeat(e, 3), minlength=nvb) / np.maximum(n_conn, 1)
        g = np.where(mask, v, g)
    return g


def report(tag, u):
    m = region
    r = u[m] / vobs[m]
    o = ~pin & gr & (vobs > 10)
    ro = u[o] / vobs[o]
    print(f'{tag}: region speed/obs median {np.median(r):.3f} (p10 {np.percentile(r, 10):.2f}, p90 {np.percentile(r, 90):.2f}), '
          f'RMSE {np.sqrt(np.mean((u[m] - vobs[m]) ** 2)):.1f} m/yr; never-pinned grounded speed/obs median {np.median(ro):.3f}, '
          f'RMSE {np.sqrt(np.mean((u[o] - vobs[o]) ** 2)):.1f} m/yr; max ice speed {np.nanmax(u[il < 0]):.0f}', flush=True)


if CONT:
    C, u = _p[f'C{K0}'].copy(), _p[f'u{K0}'].copy()
    report(f'it{K0} (from {PREV})', u)
    hist = {f'C{K0}': C.copy(), f'u{K0}': u.copy()}
else:
    report('it0 (B95)', u)
    hist = {'C0': C.copy(), 'u0': u.copy()}
for k in range(K0 + 1, K0 + NIT + 1):
    ratio = np.ones(nvb)
    ratio[region] = np.sqrt(np.maximum(u[region], 1.0) / vobs[region])
    ratio = np.clip(smooth(ratio, region), *CLIP)
    C = np.where(region, np.clip(C * ratio, *CBOUNDS), C)
    print(f'it{k}: C factor on region median {np.median(ratio[region]):.3f} (p10 {np.percentile(ratio[region], 10):.3f}, '
          f'p90 {np.percentile(ratio[region], 90):.3f})', flush=True)
    md.friction.coefficient = np.tile(C, nl)
    name = f'AIS3_ho_fricfp_it{k}'
    md.miscellaneous.name = name
    pyissm.model.execute.solve(md, 'Stressbalance', batch=True, load_only=False, runtime_name=False)
    dd = f'{EXE}/{name}/{name}'
    t0 = time.time()
    while not os.path.exists(dd + '.lock'):
        log = open(dd + '.outlog', errors='replace').read() if os.path.exists(dd + '.outlog') else ''
        if 'Exit Status:' in log:
            raise SystemExit(f'{name} ended without a lock file')
        if time.time() - t0 > 30 * 3600:
            raise SystemExit(f'{name} timed out')
        time.sleep(120)
    md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
    u = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().reshape(nl, nvb)[-1].copy()
    md.results.StressbalanceSolution = []
    report(f'it{k}', u)
    hist[f'C{k}'], hist[f'u{k}'] = C.copy(), u.copy()
    np.savez(OUT, vobs=vobs, pin=pin, region=region, **hist)
print(f'saved {OUT}\nDONE')
