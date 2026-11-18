"""Does HO run pin-free BEFORE the pinned-column spikes form (worklog 6.23)? On the 2019 HO state
releasing the pins gives extreme speeds with both the Schoof field (6.21) and the SSA Budd field
(6.23), at columns that thickened by up to ~3 km during the pinned 1995-2019 historical run. This
repeats the test on models/AIS3_ho_relaxed.nc, the state the historical run starts from (inverted
geometry plus the 1-year relaxation), after the same load fixes as historical_dhdt_tuning
(basalforcings.tf to arrays, sync_mesh_z, drop the relaxation's TransientSolution).

Stress-balance-only solves, all pins released (none are on the domain boundary, 6.21):
  S95  HO Schoof friction as is (the production HO field).
  B95  Budd p=q=1 exactly as in models/AIS3_proj_ssa_init.nc (coefficient and Ehrenfeucht N,
       coupling 3, limit 0.07), copied up every column.
Basal melt off, as in check_ho_unpin.py. Compared against observations and against the SSA 2019
speed (first record of models/AIS3_proj_ssa_ctrl_2019-2050_history.npz). Run dirs
execution/AIS3_ho_unpin1995_*. Read-only for model files; writes models/ho_unpin_1995_test.npz
(refuses to overwrite).
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
OUT = f'{M}/ho_unpin_1995_test.npz'
TAGS = ('S95', 'B95')
NAME = 'AIS3_ho_unpin1995_{}'
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')
for tag in TAGS:
    if os.path.exists(f'{EXE}/' + NAME.format(tag)):
        raise SystemExit(f'{EXE}/' + NAME.format(tag) + ' exists; move it aside first')


# copied from ais_0.1.py (keep in sync)
def sync_mesh_z(md):
    """Recompute the extruded mesh's vertex z from the current geometry (base + sigma*H).

    Keeps each column's relative layer spacing (sigma from the existing z), so it works even if
    md.mesh.z is stale -- e.g. after geometry was edited without touching z, as the thickness
    restoration (restore_floored_thickness.py) and the historical chunk resume did.
    """
    nv = md.mesh.numberofvertices
    nvb = int(md.mesh.numberofvertices2d)
    n_layers = nv // nvb
    assert n_layers * nvb == nv
    assert np.asarray(md.mesh.vertexonbase).astype(bool).ravel()[:nvb].all()
    assert np.asarray(md.mesh.vertexonsurface).astype(bool).ravel()[-nvb:].all()
    z = np.asarray(md.mesh.z, dtype=float).ravel().reshape(n_layers, nvb)
    span = z[-1] - z[0]
    safe = np.abs(span) > 1e-9
    sigma = np.where(safe[None, :], (z - z[0]) / np.where(safe, span, 1.0)[None, :],
                     np.linspace(0.0, 1.0, n_layers)[:, None])
    base = np.asarray(md.geometry.base, dtype=float).ravel()[:nvb]
    thick = np.asarray(md.geometry.thickness, dtype=float).ravel()[:nvb]
    d_base = np.abs(z[0] - base)
    d_surf = np.abs(z[-1] - (base + thick))
    print(f"   mesh.z vs geometry before sync: base max {d_base.max():.3f} m ({int((d_base > 0.01).sum())} "
          f"columns off by >1 cm), surface max {d_surf.max():.3f} m ({int((d_surf > 0.01).sum())} columns)")
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
vobs = np.asarray(md.inversion.vel_obs, float).ravel()[:nvb]
spx0 = np.asarray(md.stressbalance.spcvx, float).ravel()
pin = np.isfinite(spx0[:nvb])
ice = il < 0
gr = ice & (ol > 0)
fl = ice & (ol < 0)
Ccol = np.asarray(md.friction.C, float).ravel().reshape(nl, nvb)
# Layers 1-14 hold a different (pre-inversion) C than the base on ~1.46M columns, in the 2019 state
# too; only the base layer enters the friction law, so Ccol[0] is the field.
H95 = np.asarray(md.geometry.thickness, float).ravel()[:nvb]
S95 = np.asarray(md.geometry.surface, float).ravel()[:nvb]
print(f'HO 1995 state: {nvb} x {nl} layers; {int(pin.sum())} pinned columns ({int((pin & gr).sum())} grounded ice); '
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
S_ssa = np.asarray(s['geometry'].variables['surface'][:], float).ravel()
H_ssa = np.asarray(s['geometry'].variables['thickness'][:], float).ravel()
s.close()
hs = np.load(f'{M}/AIS3_proj_ssa_ctrl_2019-2050_history.npz')
v_ssa = np.asarray(hs['vel'][0], float)
print(f'SSA speed from history t={hs["times"][0]}', flush=True)
dS = S95 - S_ssa
print('HO 1995 surface minus SSA surface, grounded ice: ' + ', '.join(
    f'|dS| > {t} m {int((gr & (np.abs(dS) > t)).sum())}' for t in (100, 300, 1000)) +
    f'; max {np.abs(dS[gr]).max():.0f} m', flush=True)

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

for tag in TAGS:
    if tag == 'B95':
        md.friction = pyissm.model.classes.friction.default()
        md.friction.coefficient = np.tile(C2, nl)
        md.friction.p = np.ones(ne)
        md.friction.q = np.ones(ne)
        md.friction.coupling = 3
        md.friction.linearize = 0
        md.friction.effective_pressure = np.tile(N2, nl)
        md.friction.effective_pressure_limit = Nlim
    md.miscellaneous.name = NAME.format(tag)
    print(f'-- submitting {tag} (friction {type(md.friction).__name__})', flush=True)
    pyissm.model.execute.solve(md, 'Stressbalance', batch=True, load_only=False, runtime_name=False)

pending, done, t0 = set(TAGS), {}, time.time()
while pending and time.time() - t0 < 34 * 3600:
    time.sleep(120)
    for tag in sorted(pending):
        dd = f'{EXE}/{NAME.format(tag)}/{NAME.format(tag)}'
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
    md.miscellaneous.name = NAME.format(tag)
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
thin = H95 <= 150
for tag, vs in surf.items():
    print(f'\n== {tag}: ice surface speed max {np.nanmax(vs[ice]):.0f} m/yr; > 10 km/yr {int((vs[ice] > 1e4).sum())}, '
          f'> 30 km/yr {int((vs[ice] > 3e4).sum())}; NaN {int(np.isnan(vs[ice]).sum())}')
    bad = gr & (vs > 3 * np.maximum(v_ssa, 1)) & (vs > 1000)
    print(f'   grounded > 3x SSA and > 1 km/yr: {int(bad.sum())} (pinned {int((bad & pin).sum())}, '
          f'H <= 150 m {int((bad & thin).sum())}, |dS| > 100 m {int((bad & (np.abs(dS) > 100)).sum())})')
    print('  speed / obs:')
    for nm, m in (('formerly pinned, grounded', pin & gr & obs), ('formerly pinned, floating', pin & fl & obs),
                  ('never pinned, grounded', ~pin & gr & obs), ('never pinned, floating', ~pin & fl & obs),
                  ('grounded, H <= 150 m', thin & gr & obs)):
        ratio_line(nm, vs, vobs, m)
    print('  speed / SSA 2019 speed (grounded, never pinned, by SSA speed):')
    for lo, hi in ((1, 10), (10, 50), (50, 200), (200, 1e9)):
        ratio_line(f'SSA {lo}-{hi:.0f} m/yr', vs, v_ssa, ~pin & gr & (v_ssa >= lo) & (v_ssa < hi))
    ratio_line('formerly pinned, grounded', vs, v_ssa, pin & gr & (v_ssa > 1))
    for rn, m in (('never pinned', ~pin & gr & obs), ('formerly pinned', pin & gr & obs)):
        for nm, a in ((tag, vs), ('SSA', v_ssa)):
            r = lambda mm: np.sqrt(np.mean((a[mm] - vobs[mm]) ** 2))
            print(f'   grounded RMSE vs obs, {nm:4s} {rn:15s}: all {r(m & np.isfinite(a)):8.1f}, '
                  f'excl. > 3x SSA {r(m & np.isfinite(a) & ~bad):7.1f} m/yr')
    ratio_line('basal / surface, grounded > 10 m/yr', base[tag], vs, gr & (vs > 10))
    top = np.argsort(np.where(ice, vs, -1))[::-1][:8]
    print('   fastest ice vertices: ' + '; '.join(
        f'({md.mesh.x[k] / 1e3:.0f},{md.mesh.y[k] / 1e3:.0f}) km {vs[k]:.0f} obs {vobs[k]:.0f} ssa {v_ssa[k]:.0f} '
        f'H {H95[k]:.0f} (SSA {H_ssa[k]:.0f}) Cschoof {Ccol[0, k]:.0f} Cbudd {C2[k]:.0f}'
        f'{" pinned" if pin[k] else ""}{" grounded" if gr[k] else ""}' for k in top))
np.savez(OUT, vobs=vobs, pin=pin, v_ssa=v_ssa, H95=H95, dS=dS,
         **{f'surf_{k}': v for k, v in surf.items()}, **{f'base_{k}': v for k, v in base.items()})
print(f'\nsaved {OUT}')
print('DONE')
