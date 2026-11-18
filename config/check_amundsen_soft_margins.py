"""Do softer shear margins make Amundsen's trunks sensitive to losing their shelves (worklog 6.24-6.25)?

6.24: Amundsen fast trunks carry ~1/3 of their driving stress on the bed, yet removing every shelf
speeds them up only 8%; grounded B is the temperature-based B, so margins cannot soften. 6.25: a
joint C + B inversion cannot test this (ISSM's grounded-ice B gradient does not move B). Instead,
prescribe softer margins and re-invert friction only:
  J0   reference: friction-only inversion with B0 (run already done by check_amundsen_joint_inv.py)
  M70  B = 0.7 B0 in the margins (enhancement factor ~2.9), friction re-inverted as J0
  M50  B = 0.5 B0 in the margins (enhancement factor 8), friction re-inverted as J0
Margins: grounded ice (obs > 20 m/yr) where the flow-frame shear strain rate of the OBSERVED
velocity exceeds EPS_MARGIN and the along-flow strain rate (independent of any model fit); 3.9% of
the sector's grounded area. Plain effective strain rate was not used: Amundsen trunks stretch
fast along flow (median 0.07/yr), so it would soften the trunks too. Then for each of J0, M70, M50: F0 shelves as is, F1 all floating
ice removed (as check_shelf_removal.py). Same Amundsen subdomain, BCs, costs and settings as
check_amundsen_joint_inv.py J0. Results are read from the .outbin files directly.
Run dirs execution_SSA/AIS3_amusoft_*. Writes models/amundsen_soft_margins_test.npz (refuses to
overwrite). Read-only for other model files.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import copy
import time
import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import pyissm
from pyissm.model import execute as ex

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
OUT = f'{M}/amundsen_soft_margins_test.npz'
NAME = 'AIS3_amusoft_{}'
J0DIR = f'{EXE}/AIS3_amujoint_J0/AIS3_amujoint_J0'
SOFT = {'M70': 0.7, 'M50': 0.5}
CASES = ('J0',) + tuple(SOFT)
FWD = tuple(f'{c}F{k}' for c in CASES for k in (0, 1))
EPS_MARGIN = 0.01   # 1/yr, flow-frame shear strain rate of the observed velocity
SECTOR = 4
for p_ in [OUT] + [f'{EXE}/' + NAME.format(t) for t in tuple(SOFT) + FWD]:
    if os.path.exists(p_):
        raise SystemExit(f'{p_} exists; move it aside first')
assert os.path.exists(J0DIR + '.lock'), 'J0 run (check_amundsen_joint_inv.py) not found'


def outbin(tag_path):
    r = ex.parse_results_from_disk(pyissm.model.Model(), tag_path + '.outbin')
    return (r.steps[0] if hasattr(r, 'steps') else r).__dict__


# ---- Amundsen subdomain, exactly as check_amundsen_joint_inv.py --------------------------------
md = pyissm.model.io.load_model(f'{M}/AIS3_proj_ssa_init.nc')
md.results.TransientSolution = []
x, y = np.asarray(md.mesh.x, float).ravel(), np.asarray(md.mesh.y, float).ravel()
el = np.asarray(md.mesh.elements).astype(int) - 1
ds = xr.open_dataset('assets/larmip3/LARMIP3_sectors-1km.nc')
var = next(v for v in ds.data_vars if ds[v].ndim == 2)
mx, my = np.meshgrid(ds['x'].values, ds['y'].values)
mv = ds[var].values.astype(float)
ok = np.isfinite(mv)
sector = np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mv[ok])(x, y)).astype(int)
ice_el = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset) < 1
in_el = ice_el & np.all(sector[el] == SECTOR, axis=1)
cut_v = np.zeros(x.size, bool)
cut_v[el[ice_el & ~in_el].ravel()] = True
md.inversion.vx_obs = np.nan_to_num(np.asarray(md.inversion.vx_obs, float))
md.inversion.vy_obs = np.nan_to_num(np.asarray(md.inversion.vy_obs, float))
mds = md.extract(in_el)
del md
ev = np.asarray(mds.mesh.extractedvertices).astype(int).ravel() - 1
nv = mds.mesh.numberofvertices
bnd = np.asarray(mds.mesh.vertexonboundary).astype(bool).ravel()
ol = np.asarray(mds.mask.ocean_levelset, float).ravel()
il = np.asarray(mds.mask.ice_levelset, float).ravel()
cut = bnd & cut_v[ev]
margin_fl = bnd & ~cut & (ol < 0)
for k in ('spcvx', 'spcvy', 'spcvz'):
    getattr(mds.stressbalance, k)[margin_fl] = np.nan
ice = il < 0
gr = ice & (ol > 0)
fl = ice & (ol < 0)
H = np.asarray(mds.geometry.thickness, float).ravel()
vobs = np.asarray(mds.inversion.vel_obs, float).ravel()
C0 = np.asarray(mds.friction.coefficient, float).ravel().copy()
B0 = np.asarray(mds.materials.rheology_B, float).ravel().copy()
el_s = np.asarray(mds.mesh.elements).astype(int) - 1
xs, ys = np.asarray(mds.mesh.x, float).ravel(), np.asarray(mds.mesh.y, float).ravel()
print(f'Amundsen subdomain: {nv} vertices (J0 reference reused)', flush=True)

X, Y = xs[el_s], ys[el_s]
det = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
area_e = np.abs(det) / 2


def vgrad(f):
    F = f[el_s]
    gx = ((F[:, 1] - F[:, 0]) * (Y[:, 2] - Y[:, 0]) - (F[:, 2] - F[:, 0]) * (Y[:, 1] - Y[:, 0])) / det
    gy = ((F[:, 2] - F[:, 0]) * (X[:, 1] - X[:, 0]) - (F[:, 1] - F[:, 0]) * (X[:, 2] - X[:, 0])) / det
    w = np.zeros(nv); ax = np.zeros(nv); ay = np.zeros(nv)
    for k in range(3):
        np.add.at(w, el_s[:, k], area_e); np.add.at(ax, el_s[:, k], area_e * gx); np.add.at(ay, el_s[:, k], area_e * gy)
    return ax / w, ay / w


vxo, vyo = np.asarray(mds.inversion.vx_obs, float).ravel(), np.asarray(mds.inversion.vy_obs, float).ravel()
uxx, uxy = vgrad(vxo)
vyx, vyy = vgrad(vyo)
exy = 0.5 * (uxy + vyx)
eps_e = np.sqrt(uxx ** 2 + vyy ** 2 + uxx * vyy + exy ** 2)   # 1/yr (velocities in m/yr)
sp_ = np.maximum(np.hypot(vxo, vyo), 1e-6)
cs_, sn_ = vxo / sp_, vyo / sp_
e_long = uxx * cs_ ** 2 + vyy * sn_ ** 2 + 2 * exy * cs_ * sn_
e_shear = (vyy - uxx) * cs_ * sn_ + exy * (cs_ ** 2 - sn_ ** 2)
marg = gr & (vobs > 20) & (np.abs(e_shear) > EPS_MARGIN) & (np.abs(e_shear) > np.abs(e_long))
area_v = np.zeros(nv)
np.add.at(area_v, el_s.ravel(), np.repeat(area_e / 3, 3))
print(f'margins (grounded, observed shear strain rate > {EPS_MARGIN}/yr and > along-flow): {int(marg.sum())} vertices, {area_v[marg].sum() / 1e6:.0f} km^2 '
      f'({area_v[marg].sum() / area_v[gr].sum():.1%} of grounded area); obs speed there median {np.median(vobs[marg]):.0f} m/yr', flush=True)

# ---- shared settings, as J0 ------------------------------------------------------------------
mds.basalforcings = pyissm.model.classes.basalforcings.default(mds.basalforcings)
mds.basalforcings.groundedice_melting_rate = np.zeros(nv)
mds.basalforcings.floatingice_melting_rate = np.zeros(nv)
mds.transient = pyissm.model.classes.transient.deactivate_all(mds.transient)
mds.stressbalance.restol = 0.01
mds.stressbalance.reltol = 0.1
mds.stressbalance.abstol = np.nan
mds.settings.solver_residue_threshold = 1e-3
mds.verbose.solution = 0
mds.verbose.control = 1
c = pyissm.model.classes.cluster.gadi()
c.codepath = os.environ['ISSM_DIR'] + '/bin'
c.executionpath = EXE
c.storage = 'gdata/au88+gdata/vk83+gdata/av17'
c.moduleuse = ['/g/data/vk83/modules/']
c.moduleload = ['access-issm_ad/2026.05.0']
c.login = 'jh7060'
c.project = 'au88'
c.queue, c.np, c.memory, c.time = 'normal', 48, 190, 2 * 60
mds.cluster = c
mds.settings.waitonlock = 30  # > 0 so ISSM writes the lock file; batch=True skips pyissm's own wait
mask = (vobs > 0) & (ol >= 0)
touch_f = (ol < 0)[el_s].any(axis=1)
gladj = np.zeros(nv, bool)
gladj[el_s[touch_f].ravel()] = True
reg_mask = (ol >= 0) & ~gladj


def wait(tags, hours):
    pending, done, t0 = set(tags), {}, time.time()
    while pending and time.time() - t0 < hours * 3600:
        time.sleep(60)
        for tag in sorted(pending):
            dd = f'{EXE}/{NAME.format(tag)}/{NAME.format(tag)}'
            log = open(dd + '.outlog', errors='replace').read() if os.path.exists(dd + '.outlog') else ''
            if os.path.exists(dd + '.lock') or 'Exit Status:' in log:
                pending.discard(tag)
                done[tag] = os.path.exists(dd + '.lock')
                print(f'{tag}: lock {done[tag]}; {(log.split("Exit Status:")[1].splitlines()[0].strip() if "Exit Status:" in log else "")}', flush=True)
    print('pending (timed out):', sorted(pending), flush=True)
    return done


# ---- friction re-inversions with softened margins ---------------------------------------------
Bcase = {'J0': B0}
for tag, f_ in SOFT.items():
    Bcase[tag] = np.where(marg, f_ * B0, B0)
    m = copy.deepcopy(mds)
    m.materials.rheology_B = Bcase[tag].copy()
    m.inversion = pyissm.model.classes.inversion.m1qn3(m.inversion)
    m.inversion.iscontrol = 1
    m.inversion.maxsteps = 600
    m.inversion.maxiter = 300
    m.inversion.gttol = 1e-8
    m.friction.coefficient = np.where(fl, 0.05, C0)
    m.inversion.control_parameters = ['FrictionCoefficient']
    m.inversion.control_scaling_factors = np.array([1.0])
    m.inversion.min_parameters = np.where(fl, 0.0, 0.05)
    m.inversion.max_parameters = np.where(fl, 0.0, 900.0)
    m.inversion.cost_functions = [101, 103, 501]
    m.inversion.cost_functions_coefficients = np.column_stack(
        [np.where(mask, 10.0, 0.0), np.where(mask, 100.0, 0.0), np.where(reg_mask, 1e-4, 0.0)])
    m.miscellaneous.name = NAME.format(tag)
    print(f'-- submitting inversion {tag} (B x {f_} on margins)', flush=True)
    pyissm.model.execute.solve(m, 'Stressbalance', batch=True, load_only=False, runtime_name=False)
    del m
done = wait(tuple(SOFT), 6)
if not all(done.get(t) for t in SOFT):
    raise SystemExit('an inversion failed; stopping before the forward solves')

field = {}
for tag in CASES:
    d = outbin(J0DIR if tag == 'J0' else f'{EXE}/{NAME.format(tag)}/{NAME.format(tag)}')
    Cn = np.asarray(d['FrictionCoefficient'], float).ravel()
    vel = np.asarray(d['Vel'], float).ravel()
    J = np.asarray(d['J'], float)
    field[tag] = (Cn, vel)
    mm = gr & (vobs > 0)
    rm = lambda q: np.sqrt(np.mean((vel[q] - vobs[q]) ** 2))
    print(f'{tag}: final cost {J.ravel()[-1] if J.ndim < 2 else J[-1, -1]:.5g}; grounded RMSE {rm(mm):.1f} m/yr '
          f'(margins {rm(mm & marg):.1f}, fast > 1000 m/yr {rm(mm & (vobs > 1000)):.1f})', flush=True)

# ---- shelf removal on each case ---------------------------------------------------------------
glv = np.zeros(nv, bool)
glv[el_s[np.any(gr[el_s], axis=1) & np.any(fl[el_s], axis=1)].ravel()] = True
glv &= gr
dist = cKDTree(np.column_stack([xs[glv], ys[glv]])).query(np.column_stack([xs, ys]))[0]


def detached(icem, oln, C):
    ie = np.where(np.all(icem[el_s], axis=1))[0]
    e = el_s[ie]
    ed = np.sort(np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]]), axis=1)
    key = ed[:, 0].astype(np.int64) * nv + ed[:, 1]
    owner = np.tile(np.arange(ie.size), 3)
    order = np.argsort(key, kind='stable')
    same = key[order][1:] == key[order][:-1]
    a_, b_ = owner[order][:-1][same], owner[order][1:][same]
    _, lab = connected_components(coo_matrix((np.ones(a_.size), (a_, b_)), shape=(ie.size, ie.size)), directed=False)
    anchored = np.zeros(lab.max() + 1, bool)
    np.logical_or.at(anchored, lab, np.all((oln[e] > 0) & (C[e] > 0), axis=1))
    held = np.zeros(nv, bool)
    held[e[anchored[lab]].ravel()] = True
    return icem & ~held


removed = np.zeros(nv, bool)
for tag in FWD:
    case = tag[:-2]
    Cn, _ = field[case]
    m = copy.deepcopy(mds)
    m.inversion.iscontrol = 0
    m.stressbalance.requested_outputs = ['default']
    m.friction.coefficient = Cn.copy()
    m.materials.rheology_B = Bcase[case].copy()
    m.cluster.time = 60
    if tag.endswith('F1'):
        gone = fl | detached(ice & ~fl, ol, Cn)
        removed |= gone
        il_ = il.copy(); il_[gone] = 1.0
        m.mask.ice_levelset = il_
        H_ = H.copy(); H_[gone] = float(m.masstransport.min_thickness)
        bed = np.asarray(m.geometry.bed, float).ravel()
        base = np.asarray(m.geometry.base, float).ravel().copy()
        hfl = -H_[gone] * float(m.materials.rho_ice) / float(m.materials.rho_water)
        floats = bed[gone] < hfl
        base[gone] = np.where(floats, hfl, bed[gone])
        oln = ol.copy(); oln[gone] = np.where(floats, -1.0, np.where(ol[gone] > 0, ol[gone], 1.0))
        m.mask.ocean_levelset = oln
        m.geometry.thickness, m.geometry.base, m.geometry.surface = H_, base, base + H_
        for k in ('spcvx', 'spcvy', 'spcvz'):
            s_ = np.asarray(getattr(m.stressbalance, k), float).copy(); s_[gone] = np.nan
            setattr(m.stressbalance, k, s_)
        print(f'   {tag}: removed {int(gone.sum())} vertices ({int(fl.sum())} floating)', flush=True)
    m.miscellaneous.name = NAME.format(tag)
    print(f'-- submitting {tag}', flush=True)
    pyissm.model.execute.solve(m, 'Stressbalance', batch=True, load_only=False, runtime_name=False)
    del m
done = wait(FWD, 4)
v = {t: np.asarray(outbin(f'{EXE}/{NAME.format(t)}/{NAME.format(t)}')['Vel'], float).ravel() for t, okk in done.items() if okk}

# ---- diagnostics ------------------------------------------------------------------------------
keep = gr & ~removed
sgx, sgy = vgrad(np.asarray(mds.geometry.surface, float).ravel())
taud = float(mds.materials.rho_ice) * float(mds.constants.g) * H * np.hypot(sgx, sgy)
Nf = np.maximum(np.asarray(mds.friction.effective_pressure, float).ravel(),
                0.07 * float(mds.materials.rho_ice) * float(mds.constants.g) * H)
yts = float(mds.constants.yts)
trunk = keep & (vobs > 1000) & (dist < 20e3) & (taud > 1e3)
print(f'\n== fast trunks (obs > 1000 m/yr, < 20 km of the GL): {int(trunk.sum())} vertices')
for case in CASES:
    Cn, vel = field[case]
    tb = Cn ** 2 * Nf * vel / yts
    print(f'   {case}: drag/driving median {np.median(tb[trunk] / taud[trunk]):.2f}; C median {np.median(Cn[trunk]):.0f}')

print('\n== shelf removal F1/F0 on grounded ice kept, F0 speed > 100 m/yr: median (flux-weighted, ratio capped at 10)')
for case in CASES:
    if f'{case}F0' not in v or f'{case}F1' not in v:
        print(f'   {case}: forward solve missing')
        continue
    a, b = v[f'{case}F1'], v[f'{case}F0']
    row = []
    for lo, hi in ((0, 5e3), (5e3, 20e3), (20e3, 50e3)):
        mm = keep & (dist >= lo) & (dist < hi) & (b > 100)
        r = a[mm] / b[mm]
        row.append(f'{lo / 1e3:.0f}-{hi / 1e3:.0f} km {np.median(r):.2f} ({np.sum(np.minimum(r, 10) * b[mm] * H[mm]) / np.sum(b[mm] * H[mm]):.2f})')
    row.append(f'trunks {np.median(a[trunk] / b[trunk]):.2f}')
    row.append(f'n > 10x {int((keep & (b > 100) & (a > 10 * b)).sum())}')
    print(f'   {case}: ' + ';  '.join(row))
np.savez(OUT, ev=ev, vobs=vobs, gr=gr, fl=fl, dist=dist, removed=removed, marg=marg, eps_e=eps_e, e_shear=e_shear, e_long=e_long, B0=B0,
         **{f'C_{k}': f[0] for k, f in field.items()}, **{f'vel_{k}': f[1] for k, f in field.items()},
         **{f'fwd_{k}': a for k, a in v.items()})
print(f'\nsaved {OUT}')
print('DONE')
