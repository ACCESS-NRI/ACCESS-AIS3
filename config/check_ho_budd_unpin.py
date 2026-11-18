"""Can HO run pin-free with the SSA track's friction (worklog 6.21 follow-up)? check_ho_unpin.py showed
that releasing the ~176k pinned columns on the HO 2019 state brings back extreme speeds (up to 7e5 m/yr)
with the HO Schoof field. The SSA track runs with no pins at all on its Budd p=q=1 field. This swaps
that field into the HO 2019 state and releases every pin.

Stress-balance-only solves on models/AIS3_proj_ho_relaxfix_init.nc (HO geometry, rheology and
temperature unchanged; only the friction and the pins change):
  B2   Budd p=q=1 exactly as in models/AIS3_proj_ssa_init.nc (coefficient and Ehrenfeucht N, coupling 3,
       limit 0.07), copied up every column; all pins released (on-ice and ice_levelset >= 0).
  B2f  as B2, but HO-grounded ice with coefficient 0 (spots floating in the SSA inversion, or grounded
       only in the HO geometry) gets the coefficient of the nearest HO-grounded vertex with C > 0.
Basal melt off, as in check_ho_unpin.py. Compared against observations, against U0 (the production HO
state with pins, models/ho_unpin_test.npz) and against the SSA 2019 speed (first record of
models/AIS3_proj_ssa_ctrl_2019-2050_history.npz). Run dirs execution/AIS3_ho_budd_unpin_B2*.
Read-only for model files; writes models/ho_budd_unpin_test.npz (refuses to overwrite).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import time
import numpy as np
import netCDF4
from scipy.spatial import cKDTree
import pyissm

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
OUT = f'{M}/ho_budd_unpin_test.npz'
TAGS = ('B2', 'B2f')
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')
for tag in TAGS:
    if os.path.exists(f'{EXE}/AIS3_ho_budd_unpin_{tag}'):
        raise SystemExit(f'{EXE}/AIS3_ho_budd_unpin_{tag} exists; move it aside first')

md = pyissm.model.io.load_model(f'{M}/AIS3_proj_ho_relaxfix_init.nc')
md.results.TransientSolution = []
nv = md.mesh.numberofvertices
nvb = int(md.mesh.numberofvertices2d)
nl = nv // nvb
ne = md.mesh.numberofelements
il = np.asarray(md.mask.ice_levelset, float).ravel()[:nvb]
ol = np.asarray(md.mask.ocean_levelset, float).ravel()[:nvb]
vobs = np.asarray(md.inversion.vel_obs, float).ravel()[:nvb]
spx0 = np.asarray(md.stressbalance.spcvx, float).ravel()
pin = np.isfinite(spx0[:nvb])
ice = il < 0
gr = ice & (ol > 0)
fl = ice & (ol < 0)
print(f'HO state: {nvb} x {nl} layers; {int(pin.sum())} pinned columns ({int((pin & gr).sum())} grounded ice); '
      f'friction {type(md.friction).__name__}', flush=True)

# SSA Budd field on the same 2D mesh
s = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
assert np.allclose(np.asarray(s['mesh'].variables['x'][:]), np.asarray(md.mesh.x)[:nvb])
assert np.allclose(np.asarray(s['mesh'].variables['y'][:]), np.asarray(md.mesh.y)[:nvb])
sf = s['friction']
assert sf.getncattr('classtype') == 'friction.default' and int(sf.getncattr('coupling')) == 3
C2 = np.asarray(sf.variables['coefficient'][:], float).ravel()
N2 = np.asarray(sf.variables['effective_pressure'][:], float).ravel()
assert np.all(np.asarray(sf.variables['p'][:]) == 1) and np.all(np.asarray(sf.variables['q'][:]) == 1)
Nlim = float(sf.getncattr('effective_pressure_limit'))
ssa_gr = (np.asarray(s['mask'].variables['ocean_levelset'][:]) > 0) & (np.asarray(s['mask'].variables['ice_levelset'][:]) < 0)
s.close()
hs = np.load(f'{M}/AIS3_proj_ssa_ctrl_2019-2050_history.npz')
v_ssa = np.asarray(hs['vel'][0], float)
print(f'SSA speed from history t={hs["times"][0]}', flush=True)
u0 = np.load(f'{M}/ho_unpin_test.npz')['surf_U0']

zero = gr & (C2 <= 0)
src = gr & (C2 > 0)
C2f = C2.copy()
_, k = cKDTree(np.c_[md.mesh.x[:nvb][src], md.mesh.y[:nvb][src]]).query(np.c_[md.mesh.x[:nvb][zero], md.mesh.y[:nvb][zero]])
C2f[zero] = C2[src][k]
print(f'HO-grounded ice with C = 0: {int(zero.sum())} vertices ({int((zero & ssa_gr).sum())} also grounded in SSA, '
      f'{int((zero & ~ssa_gr).sum())} HO-only); B2f fills them, filled C median {np.median(C2f[zero]):.0f}', flush=True)

md.friction = pyissm.model.classes.friction.default()
md.friction.p = np.ones(ne)
md.friction.q = np.ones(ne)
md.friction.coupling = 3
md.friction.linearize = 0
md.friction.effective_pressure = np.tile(N2, nl)
md.friction.effective_pressure_limit = Nlim
md.stressbalance.spcvx = np.full(nv, np.nan)  # no pins at all (none are on the domain boundary, 6.21)
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
md.settings.waitonlock = 30 * 60  # > 0 so ISSM writes the lock file; batch=True skips pyissm's own wait

coef = {'B2': C2, 'B2f': C2f}
for tag in TAGS:
    md.friction.coefficient = np.tile(coef[tag], nl)
    md.miscellaneous.name = f'AIS3_ho_budd_unpin_{tag}'
    print(f'-- submitting {tag}', flush=True)
    pyissm.model.execute.solve(md, 'Stressbalance', batch=True, load_only=False, runtime_name=False)

pending, done, t0 = set(TAGS), {}, time.time()
while pending and time.time() - t0 < 34 * 3600:
    time.sleep(120)
    for tag in sorted(pending):
        dd = f'{EXE}/AIS3_ho_budd_unpin_{tag}/AIS3_ho_budd_unpin_{tag}'
        log = open(dd + '.outlog', errors='replace').read() if os.path.exists(dd + '.outlog') else ''
        if os.path.exists(dd + '.lock') or 'Exit Status:' in log:
            pending.discard(tag)
            done[tag] = os.path.exists(dd + '.lock')
            print(f'{tag}: lock {done[tag]}; NaN {"residue is NaN" in log}; residue-too-high '
                  f'{log.count("residue too high")}; nonlinear cap {log.count("exceeded")}; '
                  f'{(log.split("Exit Status:")[1].splitlines()[0].strip() if "Exit Status:" in log else "")}', flush=True)
print('pending (timed out):', sorted(pending), flush=True)

surf, base = {}, {}
for tag, ok in done.items():
    if not ok:
        continue
    md.miscellaneous.name = f'AIS3_ho_budd_unpin_{tag}'
    md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
    v = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().reshape(nl, nvb)
    surf[tag], base[tag] = v[-1].copy(), v[0].copy()
    md.results.StressbalanceSolution = []


def ratio_line(nm, a, b, m):
    m = m & np.isfinite(a) & (b > 0)
    if m.sum() == 0:
        return
    r = a[m] / b[m]
    print(f'   {nm:34s} n {int(m.sum()):7d}: median {np.median(r):.2f}, p05 {np.percentile(r, 5):.2f}, '
          f'p95 {np.percentile(r, 95):.2f}, frac > 2x {np.mean(r > 2):.3f}, > 10x {np.mean(r > 10):.4f}')


obs = vobs > 0
for tag, vs in surf.items():
    print(f'\n== {tag}: ice surface speed max {np.nanmax(vs[ice]):.0f} m/yr; > 10 km/yr {int((vs[ice] > 1e4).sum())}, '
          f'> 30 km/yr {int((vs[ice] > 3e4).sum())}; NaN {int(np.isnan(vs[ice]).sum())}')
    print('  speed / obs:')
    for nm, m in (('formerly pinned, grounded', pin & gr & obs), ('formerly pinned, floating', pin & fl & obs),
                  ('never pinned, grounded', ~pin & gr & obs), ('never pinned, floating', ~pin & fl & obs),
                  ('C = 0 grounded spots', zero & obs)):
        ratio_line(nm, vs, vobs, m)
    print('  speed / SSA 2019 speed (grounded, never pinned, by SSA speed):')
    for lo, hi in ((1, 10), (10, 50), (50, 200), (200, 1e9)):
        ratio_line(f'SSA {lo}-{hi:.0f} m/yr', vs, v_ssa, ~pin & gr & (v_ssa >= lo) & (v_ssa < hi))
    ratio_line('formerly pinned, grounded', vs, v_ssa, pin & gr & (v_ssa > 1))
    print('  speed / U0 (production HO, pinned):')
    ratio_line('never pinned, grounded', vs, u0, ~pin & gr & (u0 > 1))
    for nm, a, b in (('HO-Budd', vs, vobs), ('SSA', v_ssa, vobs), ('U0 (HO prod)', u0, vobs)):
        for rn, m in (('never pinned', ~pin & gr & obs), ('formerly pinned', pin & gr & obs)):
            mm = m & np.isfinite(a)
            print(f'   grounded RMSE vs obs, {nm:12s} {rn:15s}: {np.sqrt(np.mean((a[mm] - b[mm]) ** 2)):8.1f} m/yr')
    ratio_line('basal / surface, grounded > 10 m/yr', base[tag], vs, gr & (vs > 10))
    top = np.argsort(np.where(ice, vs, -1))[::-1][:8]
    print('   fastest ice vertices: ' + '; '.join(
        f'({md.mesh.x[k] / 1e3:.0f},{md.mesh.y[k] / 1e3:.0f}) km {vs[k]:.0f} obs {vobs[k]:.0f} ssa {v_ssa[k]:.0f} '
        f'C {C2[k]:.0f}{" pinned" if pin[k] else ""}{" grounded" if gr[k] else ""}' for k in top))
np.savez(OUT, vobs=vobs, pin=pin, zero=zero, v_ssa=v_ssa, surf_U0=u0,
         **{f'surf_{k}': v for k, v in surf.items()}, **{f'base_{k}': v for k, v in base.items()})
print(f'\nsaved {OUT}')
print('DONE')
