"""Does more plastic sliding let Amundsen's grounded ice feel the loss of its shelves (worklog 6.42)?

Shelf removal (all floating ice, front at the grounding line) barely speeds up Amundsen's grounded ice with the
production linear Budd law and prescribed N (trunks +6-8%, SSA and HO alike, 6.24 / 6.41; p=3 with flotation N:
+17%). This repeats check_shelf_removal.py's test on models/AIS3_proj_ssa_init.nc with two more plastic laws,
both with N from flotation (coupling 2) and the same initial basal drag as production where possible:
  P5_0 / P5_1  Budd p=5, q=1: C_new = C_old sqrt(N_old u0^(1-1/p) / N_new^(1/p))
  SC_0 / SC_1  Schoof law (ISSM friction.schoof, MISMIP+ form; not regcoulomb), m = 1/3, Cmax = 0.4:
               C^2 = r Cmax N / (u0^m (1 - r^(1/m))^m), r = tau0 / (Cmax N), r capped at 0.95
_0 = shelves as is, _1 = all floating ice removed (+ unanchored pieces). Responses _1/_0 per LARMIP-3 sector and
distance band, next to the production (S0/X1) and p=3 (P0/P1) results in models/shelf_removal_test.npz.
Run dirs execution_SSA/AIS3_shelfrm_{P5,SC}_{0,1}. Writes models/shelf_removal_plastic_test.npz (refuses to overwrite).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import copy
import time
import numpy as np
import xarray as xr
import pyissm
from scipy.interpolate import NearestNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
OUT = f'{M}/shelf_removal_plastic_test.npz'
TAGS = ('P5_0', 'P5_1', 'SC_0', 'SC_1')
NAME = 'AIS3_shelfrm_{}'
CMAX, MEXP, RCAP = 0.4, 1.0 / 3.0, 0.95
SECTORS = {1: 'Weddell', 2: 'East Antarctica', 3: 'Ross', 4: 'Amundsen', 5: 'Peninsula'}
BANDS = ((0, 5e3), (5e3, 20e3), (20e3, 50e3))
if os.path.exists(OUT) or any(os.path.exists(f'{EXE}/' + NAME.format(t)) for t in TAGS):
    raise SystemExit('output or a run dir exists; move it aside first')

md0 = pyissm.model.io.load_model(f'{M}/AIS3_proj_ssa_init.nc')
md0.results.TransientSolution = []
nv = md0.mesh.numberofvertices
x, y = np.asarray(md0.mesh.x, float).ravel(), np.asarray(md0.mesh.y, float).ravel()
el = np.asarray(md0.mesh.elements).astype(int) - 1
il = np.asarray(md0.mask.ice_levelset, float).ravel()
ol = np.asarray(md0.mask.ocean_levelset, float).ravel()
H = np.asarray(md0.geometry.thickness, float).ravel()
base0 = np.asarray(md0.geometry.base, float).ravel()
ice, gr, fl = il < 0, (il < 0) & (ol > 0), (il < 0) & (ol < 0)
assert int(md0.friction.coupling) == 3 and np.all(np.asarray(md0.friction.p) == 1)
C0 = np.asarray(md0.friction.coefficient, float).ravel()
g_, ri, rw = float(md0.constants.g), float(md0.materials.rho_ice), float(md0.materials.rho_water)
yts = float(md0.constants.yts)
lim = float(md0.friction.effective_pressure_limit)
floor_ = lim * ri * g_ * H
N_old = np.maximum(np.maximum(np.asarray(md0.friction.effective_pressure, float).ravel(), 0.0), floor_)
N_new = np.maximum(ri * g_ * H - rw * g_ * np.maximum(-base0, 0.0), floor_)
u0 = np.maximum(np.asarray(md0.initialization.vel, float).ravel(), 1.0) / yts      # m/s
tau0 = C0 ** 2 * N_old * u0                                                         # production drag (Pa)


def friction_p5(md):
    p = 5.0
    md.friction.coefficient = C0 * np.sqrt(N_old * u0 ** (1 - 1 / p) / np.maximum(N_new, 1e-6) ** (1 / p))
    md.friction.p = p * np.ones_like(np.asarray(md.friction.p, float))
    md.friction.q = np.ones_like(np.asarray(md.friction.q, float))
    md.friction.coupling = 2


def friction_schoof(md):
    r = np.clip(tau0 / (CMAX * np.maximum(N_new, 1e-6)), 0.0, RCAP)
    C2 = r * CMAX * N_new / (u0 ** MEXP * (1.0 - r ** (1 / MEXP)) ** MEXP)
    C = np.sqrt(np.maximum(C2, 0.0))
    C = np.where(gr, np.maximum(C, 1e-3), 1e-3)
    capped = gr & (tau0 / (CMAX * np.maximum(N_new, 1e-6)) > RCAP)
    print(f'   Schoof: grounded vertices at the {RCAP} cap (drag not preserved): {int(capped.sum())} '
          f'({capped.sum() / gr.sum():.1%})', flush=True)
    neff = md.friction.effective_pressure
    md.friction = pyissm.model.classes.friction.schoof()
    md.friction.C = C
    md.friction.Cmax = np.full(nv, CMAX)
    md.friction.m = np.full(md.mesh.numberofelements, MEXP)
    md.friction.coupling = 2
    md.friction.effective_pressure = neff
    md.friction.effective_pressure_limit = lim
    return capped


def detached(icem, oln, Cpos):
    ie = np.where(np.all(icem[el], axis=1))[0]
    e = el[ie]
    ed = np.sort(np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]]), axis=1)
    key = ed[:, 0].astype(np.int64) * nv + ed[:, 1]
    owner = np.tile(np.arange(ie.size), 3)
    order = np.argsort(key, kind='stable')
    same = key[order][1:] == key[order][:-1]
    _, lab = connected_components(coo_matrix((np.ones(int(same.sum())), (owner[order][:-1][same], owner[order][1:][same])),
                                             shape=(ie.size, ie.size)), directed=False)
    anchored = np.zeros(lab.max() + 1, bool)
    np.logical_or.at(anchored, lab, np.all((oln[e] > 0) & Cpos[e], axis=1))
    held = np.zeros(nv, bool)
    held[e[anchored[lab]].ravel()] = True
    return icem & ~held


gone = fl | detached(ice & ~fl, ol, C0 > 0)


def remove_shelves(md):
    il_ = il.copy(); il_[gone] = 1.0
    H_ = H.copy(); H_[gone] = float(md.masstransport.min_thickness)
    bed = np.asarray(md.geometry.bed, float).ravel()
    hfl = -H_[gone] * ri / rw
    floats = bed[gone] < hfl
    b_ = base0.copy(); b_[gone] = np.where(floats, hfl, bed[gone])
    ol_ = ol.copy(); ol_[gone] = np.where(floats, -1.0, np.where(ol[gone] > 0, ol[gone], 1.0))
    md.mask.ice_levelset, md.mask.ocean_levelset = il_, ol_
    md.geometry.thickness, md.geometry.base, md.geometry.surface = H_, b_, b_ + H_


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
md0.settings.waitonlock = 30
for tag in TAGS:
    md = copy.deepcopy(md0)
    if tag.startswith('P5'):
        friction_p5(md)
    else:
        friction_schoof(md)
    if tag.endswith('1'):
        remove_shelves(md)
    md.miscellaneous.name = NAME.format(tag)
    print(f'-- submitting {tag}', flush=True)
    pyissm.model.execute.solve(md, 'Stressbalance', batch=True, load_only=False, runtime_name=False)
    del md

pending, done, t0 = set(TAGS), {}, time.time()
while pending and time.time() - t0 < 6 * 3600:
    time.sleep(60)
    for tag in sorted(pending):
        dd = f'{EXE}/{NAME.format(tag)}/{NAME.format(tag)}'
        log = open(dd + '.outlog', errors='replace').read() if os.path.exists(dd + '.outlog') else ''
        if os.path.exists(dd + '.lock') or 'Exit Status:' in log:
            pending.discard(tag)
            done[tag] = os.path.exists(dd + '.lock')
            print(f'{tag}: lock {done[tag]}', flush=True)
v = {}
for tag, ok_ in done.items():
    if ok_:
        md0.miscellaneous.name = NAME.format(tag)
        md0 = pyissm.model.execute.solve(md0, 'Stressbalance', load_only=True, runtime_name=False)
        v[tag] = np.asarray(md0.results.StressbalanceSolution.Vel, float).ravel().copy()
        md0.results.StressbalanceSolution = []

ds = xr.open_dataset('assets/larmip3/LARMIP3_sectors-1km.nc')
var = next(k for k in ds.data_vars if ds[k].ndim == 2)
mx, my = np.meshgrid(ds['x'].values, ds['y'].values)
mv = ds[var].values.astype(float)
ok = np.isfinite(mv)
sector = np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mv[ok])(x, y)).astype(int)
glv = np.zeros(nv, bool)
glv[el[np.any(gr[el], axis=1) & np.any(fl[el], axis=1)].ravel()] = True
glv &= gr
dist = cKDTree(np.column_stack([x[glv], y[glv]])).query(np.column_stack([x, y]))[0]
old = np.load(f'{M}/shelf_removal_test.npz')
pairs = {'Budd p=1 prescribed N (production)': (old['vel_S0'], old['vel_X1']),
         'Budd p=3 flotation N': (old['vel_P0'], old['vel_P1'])}
if 'P5_0' in v and 'P5_1' in v:
    pairs['Budd p=5 flotation N'] = (v['P5_0'], v['P5_1'])
if 'SC_0' in v and 'SC_1' in v:
    pairs[f'Schoof Cmax {CMAX} flotation N'] = (v['SC_0'], v['SC_1'])
keep = gr & ~gone & ~old['removed']
vobs = np.asarray(md0.inversion.vel_obs, float).ravel()
for nm, (a, b) in pairs.items():
    m0 = gr & (vobs > 0)
    print(f'\n== {nm}: baseline grounded RMSE vs obs {np.sqrt(np.mean((a[m0] - vobs[m0]) ** 2)):.1f} m/yr')
    for sk, sn in SECTORS.items():
        row = []
        for lo, hi in BANDS:
            m = keep & (sector == sk) & (dist >= lo) & (dist < hi) & (a > 100)
            if m.sum() < 20:
                continue
            r = b[m] / a[m]
            fw = np.sum(np.minimum(r, 10) * a[m] * H[m]) / np.sum(a[m] * H[m])
            row.append(f'{lo / 1e3:.0f}-{hi / 1e3:.0f} km {np.median(r):.2f} ({fw:.2f})')
        mt = keep & (sector == sk) & (dist < 20e3) & (vobs > 1000)
        if mt.sum() >= 20:
            row.append(f'trunks {np.median(b[mt] / a[mt]):.2f}')
        print(f'   {sn:16s} ' + ';  '.join(row))
np.savez(OUT, sector=sector, dist=dist, gone=gone, **{f'vel_{k}': a for k, a in v.items()})
print(f'\nsaved {OUT}\nDONE')
