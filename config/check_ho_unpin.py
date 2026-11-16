"""Can the HO model run without its velocity pins (worklog 6.20)? The HO projection start carries
spcvx/spcvy = observed velocity (spcvz = 0) on ~176k columns. They were added after the Schoof
inversion to stop extreme forward speeds (thin ice, C at the lower bound, ice_levelset >= 0;
ais_0.1.py ~1596-1653). In transients they freeze the flow, and where the pinned field converges
thickness runs away (226 km at one Peninsula vertex by 2220).

Stress-balance-only solves on models/AIS3_proj_ho_relaxfix_init.nc (the 2019 HO projection state):
  U0 as is                                     (all pins; baseline)
  U1 release the on-ice pins                   (keep domain-boundary and ice_levelset >= 0 pins)
  U2 release on-ice and ice_levelset >= 0 pins (keep domain-boundary pins only)
Pin groups are classified per column on the base layer and applied to every layer. Basal melt off
(the default class, zero melt): ismip6 melt in a stress-balance-only HO solve SEGVs (worklog 5c),
and melt does not change the horizontal speed. All three are submitted at once (batch) and polled;
then speeds are compared with observations and with U0. Run dirs execution/AIS3_ho_unpin_U*.
Read-only for model files; prints a summary and writes models/ho_unpin_test.npz.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import time
import numpy as np
import pyissm

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution'

md = pyissm.model.io.load_model(f'{M}/AIS3_proj_ho_relaxfix_init.nc')
md.results.TransientSolution = []
nv = md.mesh.numberofvertices
nvb = int(md.mesh.numberofvertices2d)
nl = nv // nvb
il = np.asarray(md.mask.ice_levelset, float).ravel()[:nvb]
ol = np.asarray(md.mask.ocean_levelset, float).ravel()[:nvb]
onb = np.asarray(md.mesh.vertexonboundary).ravel().astype(bool)[:nvb]
vobs = np.asarray(md.inversion.vel_obs, float).ravel()[:nvb]
spx0 = np.asarray(md.stressbalance.spcvx, float).ravel().copy()
spy0 = np.asarray(md.stressbalance.spcvy, float).ravel().copy()
spz0 = np.asarray(md.stressbalance.spcvz, float).ravel().copy()
pin = np.isfinite(spx0[:nvb])
assert np.array_equal(np.isfinite(spx0).reshape(nl, nvb), np.tile(pin, (nl, 1))), 'pins are not whole columns'
ice = il < 0
grp_boundary = pin & ice & onb
grp_nonice = pin & ~ice
grp_onice = pin & ice & ~onb
print(f'pins: {int(pin.sum())} columns = boundary {int(grp_boundary.sum())} + nonice {int(grp_nonice.sum())} '
      f'+ onice {int(grp_onice.sum())} (grounded {int((grp_onice & (ol > 0)).sum())}); '
      f'friction {type(md.friction).__name__}', flush=True)

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

variants = {'U0': np.zeros(nvb, bool), 'U1': grp_onice, 'U2': grp_onice | grp_nonice}
for tag, release in variants.items():
    rel = np.tile(release, nl)
    md.stressbalance.spcvx = np.where(rel, np.nan, spx0)
    md.stressbalance.spcvy = np.where(rel, np.nan, spy0)
    md.stressbalance.spcvz = np.where(rel, np.nan, spz0)
    md.miscellaneous.name = f'AIS3_ho_unpin_{tag}'
    print(f'-- submitting {tag}: releasing {int(release.sum())} columns', flush=True)
    pyissm.model.execute.solve(md, 'Stressbalance', batch=True, load_only=False, runtime_name=False)

pending, done, t0 = set(variants), {}, time.time()
while pending and time.time() - t0 < 34 * 3600:
    time.sleep(120)
    for tag in sorted(pending):
        dd = f'{EXE}/AIS3_ho_unpin_{tag}/AIS3_ho_unpin_{tag}'
        log = open(dd + '.outlog', errors='replace').read() if os.path.exists(dd + '.outlog') else ''
        if os.path.exists(dd + '.lock') or 'Exit Status:' in log:
            pending.discard(tag)
            done[tag] = os.path.exists(dd + '.lock')
            print(f'{tag}: lock {done[tag]}; NaN {"residue is NaN" in log}; residue-too-high '
                  f'{log.count("residue too high")}; nonlinear cap {log.count("exceeded")}; '
                  f'{(log.split("Exit Status:")[1].splitlines()[0].strip() if "Exit Status:" in log else "")}', flush=True)
print('pending (timed out):', sorted(pending), flush=True)

speeds = {}
for tag, ok in done.items():
    if not ok:
        continue
    md.miscellaneous.name = f'AIS3_ho_unpin_{tag}'
    md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
    v = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().reshape(nl, nvb)
    speeds[tag] = v[-1].copy()
    md.results.StressbalanceSolution = []

gr = ice & (ol > 0)
fl = ice & (ol < 0)
obs = vobs > 0
for tag, vs in speeds.items():
    rel = variants[tag]
    big = ice & np.isfinite(vs)
    print(f'\n== {tag}: ice speed max {np.nanmax(vs[ice]):.0f} m/yr; > 10 km/yr {int((vs[ice] > 1e4).sum())}, '
          f'> 30 km/yr {int((vs[ice] > 3e4).sum())}; NaN {int(np.isnan(vs[ice]).sum())}')
    for nm, m in (('released, grounded', rel & gr & obs), ('released, floating', rel & fl & obs),
                  ('released, ice-free', rel & ~ice & obs), ('never pinned, grounded', ~pin & gr & obs)):
        if m.sum() == 0:
            continue
        r = vs[m] / vobs[m]
        print(f'   {nm:24s} n {int(m.sum()):7d}: speed/obs median {np.median(r):.2f}, p05 {np.percentile(r, 5):.2f}, '
              f'p95 {np.percentile(r, 95):.2f}, frac > 2x {np.mean(r > 2):.3f}, > 10x {np.mean(r > 10):.4f}')
    if tag != 'U0' and 'U0' in speeds:
        m = ~pin & big & (speeds['U0'] > 1)
        r = vs[m] / speeds['U0'][m]
        print(f'   vs U0 on never-pinned ice: median {np.median(r):.3f}, p01 {np.percentile(r, 1):.3f}, '
              f'p99 {np.percentile(r, 99):.3f}, frac |change| > 20% {np.mean(np.abs(r - 1) > 0.2):.4f}')
    top = np.argsort(np.where(ice, vs, -1))[::-1][:8]
    print('   fastest ice vertices: ' + '; '.join(
        f'({md.mesh.x[k] / 1e3:.0f},{md.mesh.y[k] / 1e3:.0f}) km {vs[k]:.0f} obs {vobs[k]:.0f} '
        f'{"pinned-" + ("onice" if grp_onice[k] else "nonice" if grp_nonice[k] else "bnd") if pin[k] else "free"}'
        for k in top))
np.savez(f'{M}/ho_unpin_test.npz', vobs=vobs, grp_onice=grp_onice, grp_nonice=grp_nonice,
         grp_boundary=grp_boundary, **{f'surf_{k}': v for k, v in speeds.items()})
print(f'\nsaved {M}/ho_unpin_test.npz')
print('DONE')
