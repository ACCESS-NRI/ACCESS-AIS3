"""How much do the ice shelves buttress the SSA initial state (melt-sensitivity question, worklog 6.24)?

LARMIP step runs thin the shelves almost completely, but the grounded ice barely speeds up (6.14).
Hypothesis: the inverted shelf B (a median 0.66x the temperature-based B of AIS3_param.nc, p10 0.37)
leaves the shelves little buttressing to lose. Test: instantaneous (stress-balance-only) response of
the grounded ice to removing ALL floating ice, on models/AIS3_proj_ssa_init.nc:
  S0   as is: Budd p=q=1, prescribed N (coupling 3), inverted shelf B          (production standard)
  T0   as S0, floating-ice B replaced by AIS3_param.nc's temperature-based B
  X1   as S0, all floating ice removed (calving front at the grounding line)  (B on shelves irrelevant)
  P0   coupledN_p3 friction (p=3, flotation N, C rescaled to the same initial drag; as
       projection_ssp_ssa), inverted shelf B                                    (LARMIP-3 main set)
  PT0  as P0 with the temperature-based shelf B
  P1   as P0, all floating ice removed
Response = X1 vs S0 / T0 and P1 vs P0 / PT0, per LARMIP-3 sector and distance from the grounding line:
flux-weighted speed ratio (sum v H A / sum v_ref H A) and median speed ratio on grounded ice.
Removal follows apply_collapse in ais_0.1_SSA.py: removed vertices ice-free (ice_levelset +1, H =
min_thickness, flotation base), plus every unanchored piece left (no fully grounded element with C > 0).
Basal melt off (irrelevant to a stress balance). Run dirs execution_SSA/AIS3_shelfrm_*.
Read-only for model files; writes models/shelf_removal_test.npz (refuses to overwrite).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import copy
import time
import numpy as np
import netCDF4
import xarray as xr
from scipy.interpolate import NearestNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import pyissm

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
OUT = f'{M}/shelf_removal_test.npz'
TAGS = ('S0', 'T0', 'X1', 'P0', 'PT0', 'P1')
NAME = 'AIS3_shelfrm_{}'
SECTORS = {1: 'Weddell', 2: 'East Antarctica', 3: 'Ross', 4: 'Amundsen', 5: 'Peninsula'}  # numbering as put to Anders (6.17)
BANDS = ((0, 5e3), (5e3, 20e3), (20e3, 50e3), (50e3, 150e3))
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists; move it aside first')
for tag in TAGS:
    if os.path.exists(f'{EXE}/' + NAME.format(tag)):
        raise SystemExit(f'{EXE}/' + NAME.format(tag) + ' exists; move it aside first')

md0 = pyissm.model.io.load_model(f'{M}/AIS3_proj_ssa_init.nc')
md0.results.TransientSolution = []
nv = md0.mesh.numberofvertices
x = np.asarray(md0.mesh.x, float).ravel()
y = np.asarray(md0.mesh.y, float).ravel()
el = np.asarray(md0.mesh.elements).astype(int) - 1
il = np.asarray(md0.mask.ice_levelset, float).ravel()
ol = np.asarray(md0.mask.ocean_levelset, float).ravel()
H = np.asarray(md0.geometry.thickness, float).ravel()
vobs = np.asarray(md0.inversion.vel_obs, float).ravel()
ice = il < 0
gr = ice & (ol > 0)
fl = ice & (ol < 0)
assert int(md0.friction.coupling) == 3 and np.all(np.asarray(md0.friction.p) == 1)
C0 = np.asarray(md0.friction.coefficient, float).ravel()
tri = 0.5 * np.abs((x[el[:, 1]] - x[el[:, 0]]) * (y[el[:, 2]] - y[el[:, 0]]) - (x[el[:, 2]] - x[el[:, 0]]) * (y[el[:, 1]] - y[el[:, 0]]))
area = np.zeros(nv)
np.add.at(area, el.ravel(), np.repeat(tri / 3, 3))

ds = xr.open_dataset('assets/larmip3/LARMIP3_sectors-1km.nc')
var = next(v for v in ds.data_vars if ds[v].ndim == 2)
mx, my = np.meshgrid(ds['x'].values, ds['y'].values)
mv = ds[var].values.astype(float)
ok = np.isfinite(mv)
sector = np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mv[ok])(x, y)).astype(int)

glv = np.zeros(nv, bool)
glv[el[np.any(gr[el], axis=1) & np.any(fl[el], axis=1)].ravel()] = True
glv &= gr
dist = cKDTree(np.column_stack([x[glv], y[glv]])).query(np.column_stack([x, y]))[0]

# temperature-based shelf B
p = netCDF4.Dataset(f'{M}/AIS3_param.nc')
Bparam = np.asarray(p['materials'].variables['rheology_B'][:], float).ravel()
p.close()
assert Bparam.size == nv
B0 = np.asarray(md0.materials.rheology_B, float).ravel()
Btemp = np.where(fl, Bparam, B0)
print(f'floating B inverted/temperature median {np.median(B0[fl] / Bparam[fl]):.2f}', flush=True)


def detached(icem, oln):
    """Ice vertices outside every piece holding a fully grounded element with C > 0 at all three
    vertices (pieces = fully-ice elements joined across shared edges); as _detached in ais_0.1_SSA.py."""
    ie = np.where(np.all(icem[el], axis=1))[0]
    e = el[ie]
    ed = np.sort(np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]]), axis=1)
    key = ed[:, 0].astype(np.int64) * nv + ed[:, 1]
    owner = np.tile(np.arange(ie.size), 3)
    order = np.argsort(key, kind='stable')
    same = key[order][1:] == key[order][:-1]
    a_, b_ = owner[order][:-1][same], owner[order][1:][same]
    _, lab = connected_components(coo_matrix((np.ones(a_.size), (a_, b_)), shape=(ie.size, ie.size)), directed=False)
    anchored = np.zeros(lab.max() + 1, bool)
    np.logical_or.at(anchored, lab, np.all((oln[e] > 0) & (C0[e] > 0), axis=1))
    held = np.zeros(nv, bool)
    held[e[anchored[lab]].ravel()] = True
    return icem & ~held


def remove_shelves(md):
    berg = detached(ice & ~fl, ol)
    gone = fl | berg
    il_ = il.copy()
    il_[gone] = 1.0
    md.mask.ice_levelset = il_
    H_ = H.copy()
    H_[gone] = float(md.masstransport.min_thickness)
    bed = np.asarray(md.geometry.bed, float).ravel()
    base = np.asarray(md.geometry.base, float).ravel().copy()
    hfl = -H_[gone] * float(md.materials.rho_ice) / float(md.materials.rho_water)
    floats = bed[gone] < hfl
    base[gone] = np.where(floats, hfl, bed[gone])
    oln = ol.copy()
    oln[gone] = np.where(floats, -1.0, np.where(ol[gone] > 0, ol[gone], 1.0))
    md.mask.ocean_levelset = oln
    md.geometry.thickness, md.geometry.base, md.geometry.surface = H_, base, base + H_
    print(f'   removed {int(fl.sum())} floating + {int(berg.sum())} unanchored vertices', flush=True)
    return gone


def coupledN_p3(md):
    """As projection_ssp_ssa (friction_mode coupledN_p3) at the initial state."""
    g_, ri_, rw_ = float(md.constants.g), float(md.materials.rho_ice), float(md.materials.rho_water)
    lim_ = float(md.friction.effective_pressure_limit)
    base_ = np.asarray(md0.geometry.base, float).ravel()
    floor_ = lim_ * ri_ * g_ * H
    N_old = np.maximum(np.maximum(np.asarray(md.friction.effective_pressure, float).ravel(), 0.0), floor_)
    N_new = np.maximum(ri_ * g_ * H - rw_ * g_ * np.maximum(-base_, 0.0), floor_)
    u0_ = np.maximum(np.asarray(md.initialization.vel, float).ravel(), 1.0) / float(md.constants.yts)
    ratio = np.sqrt(N_old * u0_ ** (2.0 / 3.0) / np.maximum(N_new, 1e-6) ** (1.0 / 3.0))
    md.friction.p = 3.0 * np.ones_like(np.asarray(md.friction.p, float))
    md.friction.q = np.ones_like(np.asarray(md.friction.q, float))
    md.friction.coefficient = C0 * ratio
    md.friction.coupling = 2


md0.basalforcings = pyissm.model.classes.basalforcings.default(md0.basalforcings)
md0.basalforcings.groundedice_melting_rate = np.zeros(nv)
md0.basalforcings.floatingice_melting_rate = np.zeros(nv)
md0.inversion.iscontrol = 0
md0.transient = pyissm.model.classes.transient.deactivate_all(md0.transient)
md0.stressbalance.requested_outputs = ['default']
md0.settings.solver_residue_threshold = 1e-3
md0.verbose.solution = 1
c = pyissm.model.classes.cluster.gadi()
c.codepath = os.environ['ISSM_DIR'] + '/bin'
c.executionpath = EXE
c.storage = 'gdata/au88+gdata/vk83+gdata/av17'
c.moduleuse = ['/g/data/vk83/modules/']
c.moduleload = ['access-issm_ad/2026.05.0']
c.login = 'jh7060'
c.project = 'au88'
c.queue, c.np, c.memory, c.time = 'normal', 48, 190, 60
md0.cluster = c
md0.settings.waitonlock = 30  # > 0 so ISSM writes the lock file; batch=True skips pyissm's own wait

removed = None
for tag in TAGS:
    md = copy.deepcopy(md0)
    if tag.startswith('P'):
        coupledN_p3(md)
    if 'T' in tag:
        md.materials.rheology_B = Btemp.copy()
    if tag.endswith('1'):
        removed = remove_shelves(md)
    md.miscellaneous.name = NAME.format(tag)
    print(f'-- submitting {tag}', flush=True)
    pyissm.model.execute.solve(md, 'Stressbalance', batch=True, load_only=False, runtime_name=False)
    del md

pending, done, t0 = set(TAGS), {}, time.time()
while pending and time.time() - t0 < 10 * 3600:
    time.sleep(60)
    for tag in sorted(pending):
        dd = f'{EXE}/{NAME.format(tag)}/{NAME.format(tag)}'
        log = open(dd + '.outlog', errors='replace').read() if os.path.exists(dd + '.outlog') else ''
        if os.path.exists(dd + '.lock') or 'Exit Status:' in log:
            pending.discard(tag)
            done[tag] = os.path.exists(dd + '.lock')
            print(f'{tag}: lock {done[tag]}; NaN {"residue is NaN" in log}; '
                  f'{(log.split("Exit Status:")[1].splitlines()[0].strip() if "Exit Status:" in log else "")}', flush=True)
print('pending (timed out):', sorted(pending), flush=True)

v = {}
for tag, okk in done.items():
    if not okk:
        continue
    md0.miscellaneous.name = NAME.format(tag)
    md0 = pyissm.model.execute.solve(md0, 'Stressbalance', load_only=True, runtime_name=False)
    v[tag] = np.asarray(md0.results.StressbalanceSolution.Vel, float).ravel().copy()
    md0.results.StressbalanceSolution = []

keep = gr & ~removed if removed is not None else gr
obs = vobs > 0
for tag in [t for t in ('S0', 'T0', 'P0', 'PT0') if t in v]:
    m = gr & obs
    print(f'{tag}: grounded RMSE vs obs {np.sqrt(np.mean((v[tag][m] - vobs[m]) ** 2)):.1f} m/yr; '
          f'floating {np.sqrt(np.mean((v[tag][fl & obs] - vobs[fl & obs]) ** 2)):.1f}')


def table(a, b, title):
    if a not in v or b not in v:
        return
    print(f'\n== {title}: {a} / {b}, grounded ice kept in both; flux-weighted ratio (median ratio)')
    print(f'   {"sector":16s}' + ''.join(f'{lo / 1e3:>5.0f}-{hi / 1e3:<4.0f}km     ' for lo, hi in BANDS))
    for s, nm in list(SECTORS.items()) + [(0, 'all')]:
        row = []
        for lo, hi in BANDS:
            m = keep & (dist >= lo) & (dist < hi) & (v[b] > 1) & (sector == s if s else True)
            if m.sum() < 20:
                row.append(f'{"--":>15s}')
                continue
            fw = np.sum(v[a][m] * H[m] * area[m]) / np.sum(v[b][m] * H[m] * area[m])
            row.append(f'{fw:6.2f} ({np.median(v[a][m] / v[b][m]):4.2f})  ')
        print(f'   {nm:16s}' + ''.join(row))


table('X1', 'S0', 'shelves removed vs inverted shelf B (Budd p=1, prescribed N)')
table('X1', 'T0', 'shelves removed vs temperature shelf B (Budd p=1, prescribed N)')
table('T0', 'S0', 'temperature vs inverted shelf B, shelves present (Budd p=1)')
table('P1', 'P0', 'shelves removed vs inverted shelf B (coupledN_p3)')
table('P1', 'PT0', 'shelves removed vs temperature shelf B (coupledN_p3)')
np.savez(OUT, vobs=vobs, sector=sector, dist=dist, removed=removed, H=H, area=area,
         **{f'vel_{k}': a for k, a in v.items()})
print(f'\nsaved {OUT}')
print('DONE')
