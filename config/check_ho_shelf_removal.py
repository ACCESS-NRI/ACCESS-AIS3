"""Does HO's grounded ice respond to losing its shelves more than SSA's (worklog 6.39-6.40)?

In the SSA setup, Amundsen grounded ice barely responds to shelf loss (removing every shelf: trunks +8%, 6.24;
observed front retreat: +0.4%, 6.39). Suspects: the SSA approximation, or how the inverted drag and prescribed N
hold the grounding zone. This repeats the shelf-removal test (check_shelf_removal.py, X1 vs S0) on the pin-free
HO 1995 state:
  F0  HO, SSA Budd friction refined in the formerly pinned regions (ho_friction_fixedpoint.py, last iteration),
      no pins: its surface speed is the fixed-point run's last solve (no new solve)
  F1  the same with all floating ice removed (front at the grounding line; unanchored pieces removed as
      apply_collapse does), one 96-cpu hugemem stress-balance solve
Response F1/F0 per LARMIP-3 sector and distance from the grounding line, next to the SSA response X1/S0 from
models/shelf_removal_test.npz on the same 2D mesh. Basal melt off. Run dir execution/AIS3_ho_shelfrm_F1.
Read-only for model files; writes models/ho_shelf_removal_test.npz (refuses to overwrite).
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import time
import numpy as np
import netCDF4
import xarray as xr
import pyissm
from scipy.interpolate import NearestNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
OUT = f'{M}/ho_shelf_removal_test.npz'
NAME = 'AIS3_ho_shelfrm_F1'
SECTORS = {1: 'Weddell', 2: 'East Antarctica', 3: 'Ross', 4: 'Amundsen', 5: 'Peninsula'}
BANDS = ((0, 5e3), (5e3, 20e3), (20e3, 50e3))
if os.path.exists(OUT) or os.path.exists(f'{EXE}/{NAME}'):
    raise SystemExit('output or run dir exists; move it aside first')
fp = np.load(f'{M}/ho_friction_fixedpoint.npz')
kmax = max(int(k[1:]) for k in fp.files if k.startswith('C') and k[1:].isdigit())
C, u0 = fp[f'C{kmax}'], fp[f'u{kmax}']
print(f'friction: fixed-point iteration {kmax}', flush=True)


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
x = np.asarray(md.mesh.x, float).ravel()[:nvb]
y = np.asarray(md.mesh.y, float).ravel()[:nvb]
il = np.asarray(md.mask.ice_levelset, float).ravel()[:nvb]
ol = np.asarray(md.mask.ocean_levelset, float).ravel()[:nvb]
H = np.asarray(md.geometry.thickness, float).ravel()[:nvb]
bed = np.asarray(md.geometry.bed, float).ravel()[:nvb]
base0 = np.asarray(md.geometry.base, float).ravel()[:nvb]
ice = il < 0
gr = ice & (ol > 0)
fl = ice & (ol < 0)

s = netCDF4.Dataset(f'{M}/AIS3_proj_ssa_init.nc')
sf = s['friction']
N2 = np.asarray(sf.variables['effective_pressure'][:], float).ravel()
lim = float(sf.getncattr('effective_pressure_limit'))
el2 = np.asarray(s['mesh'].variables['elements'][:]).astype(int) - 1
if el2.shape[0] == 3:
    el2 = el2.T
s.close()
md.friction = pyissm.model.classes.friction.default()
md.friction.coefficient = np.tile(C, nl)
md.friction.p = np.ones(ne)
md.friction.q = np.ones(ne)
md.friction.coupling = 3
md.friction.linearize = 0
md.friction.effective_pressure = np.tile(N2, nl)
md.friction.effective_pressure_limit = lim

# remove all floating ice (+ unanchored pieces), column-wise
ie = np.where(np.all((ice & ~fl)[el2], axis=1))[0]
e = el2[ie]
ed = np.sort(np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]]), axis=1)
key = ed[:, 0].astype(np.int64) * nvb + ed[:, 1]
owner = np.tile(np.arange(ie.size), 3)
order = np.argsort(key, kind='stable')
same = key[order][1:] == key[order][:-1]
_, lab = connected_components(coo_matrix((np.ones(int(same.sum())), (owner[order][:-1][same], owner[order][1:][same])),
                                         shape=(ie.size, ie.size)), directed=False)
anchored = np.zeros(lab.max() + 1, bool)
np.logical_or.at(anchored, lab, np.all((ol[e] > 0) & (C[e] > 0), axis=1))
held = np.zeros(nvb, bool)
held[e[anchored[lab]].ravel()] = True
gone = fl | ((ice & ~fl) & ~held)
il_ = il.copy(); il_[gone] = 1.0
H_ = H.copy(); H_[gone] = float(md.masstransport.min_thickness)
hfl = -H_[gone] * float(md.materials.rho_ice) / float(md.materials.rho_water)
floats = bed[gone] < hfl
b_ = base0.copy(); b_[gone] = np.where(floats, hfl, bed[gone])
ol_ = ol.copy(); ol_[gone] = np.where(floats, -1.0, np.where(ol[gone] > 0, ol[gone], 1.0))
md.mask.ice_levelset = np.tile(il_, nl)
md.mask.ocean_levelset = np.tile(ol_, nl)
md.geometry.thickness = np.tile(H_, nl)
md.geometry.base = np.tile(b_, nl)
md.geometry.surface = np.tile(b_ + H_, nl)
sync_mesh_z(md)
print(f'removed {int(gone.sum())} columns ({int(fl.sum())} floating, {int((gone & ~fl).sum())} unanchored)', flush=True)

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
md.miscellaneous.name = NAME
pyissm.model.execute.solve(md, 'Stressbalance', batch=True, load_only=False, runtime_name=False)
dd = f'{EXE}/{NAME}/{NAME}'
t0 = time.time()
while not os.path.exists(dd + '.lock'):
    log = open(dd + '.outlog', errors='replace').read() if os.path.exists(dd + '.outlog') else ''
    if 'Exit Status:' in log:
        raise SystemExit(f'{NAME} ended without a lock file')
    if time.time() - t0 > 30 * 3600:
        raise SystemExit(f'{NAME} timed out')
    time.sleep(120)
md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=True, runtime_name=False)
u1 = np.asarray(md.results.StressbalanceSolution.Vel, float).ravel().reshape(nl, nvb)[-1].copy()

# ---- compare with SSA X1/S0 on the same 2D mesh --------------------------------------------------
ds = xr.open_dataset('assets/larmip3/LARMIP3_sectors-1km.nc')
var = next(v for v in ds.data_vars if ds[v].ndim == 2)
mx, my = np.meshgrid(ds['x'].values, ds['y'].values)
mv = ds[var].values.astype(float)
ok = np.isfinite(mv)
sector = np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mv[ok])(x, y)).astype(int)
glv = np.zeros(nvb, bool)
glv[el2[np.any(gr[el2], axis=1) & np.any(fl[el2], axis=1)].ravel()] = True
glv &= gr
dist = cKDTree(np.column_stack([x[glv], y[glv]])).query(np.column_stack([x, y]))[0]
ssa = np.load(f'{M}/shelf_removal_test.npz')
s0, s1, srem = ssa['vel_S0'], ssa['vel_X1'], ssa['removed']
keep = gr & ~gone & ~srem
vobs = fp['vobs']
print('\n== immediate speed-up from removing every shelf, grounded ice kept, baseline speed > 100 m/yr:')
print('   median ratio (flux-weighted, capped at 10): HO F1/F0  |  SSA X1/S0 on the same vertices')
for sk, nm in SECTORS.items():
    row = []
    for lo, hi in BANDS:
        m = keep & (sector == sk) & (dist >= lo) & (dist < hi) & (u0 > 100) & (s0 > 100)
        if m.sum() < 20:
            row.append('--'); continue
        rh, rs = u1[m] / u0[m], s1[m] / s0[m]
        fh = np.sum(np.minimum(rh, 10) * u0[m] * H[m]) / np.sum(u0[m] * H[m])
        fs = np.sum(np.minimum(rs, 10) * s0[m] * H[m]) / np.sum(s0[m] * H[m])
        row.append(f'{lo / 1e3:.0f}-{hi / 1e3:.0f} km HO {np.median(rh):.2f} ({fh:.2f}) | SSA {np.median(rs):.2f} ({fs:.2f})')
    mt = keep & (sector == sk) & (dist < 20e3) & (vobs > 1000)
    trunks = f'trunks HO {np.median(u1[mt] / u0[mt]):.2f} | SSA {np.median(s1[mt] / s0[mt]):.2f} (n {int(mt.sum())})' if mt.sum() >= 20 else ''
    print(f'  {nm:16s} ' + ';  '.join(row) + ('; ' + trunks if trunks else ''))
print(f'HO F0 grounded speed/obs median {np.median(u0[gr & (vobs > 10)] / vobs[gr & (vobs > 10)]):.3f}; max F1 ice speed {np.nanmax(u1[~gone & ice]):.0f} m/yr')
np.savez(OUT, u0=u0, u1=u1, gone=gone, sector=sector, dist=dist, vobs=vobs)
print(f'saved {OUT}\nDONE')
