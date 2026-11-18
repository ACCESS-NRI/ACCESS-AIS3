"""Are Amundsen's trunks held by stiff shear margins (worklog 6.24)? Joint friction + grounded-B
inversion on the Amundsen sector, then the shelf-removal test on the result.

6.24: Amundsen fast trunks carry only ~1/3 of their driving stress on the bed, yet removing every
shelf speeds them up just 8%. Grounded B is the temperature-based B everywhere (only shelf B was
inverted), so the margins cannot soften and the friction inversion may lower trunk drag instead.
Test on the Amundsen subdomain (LARMIP-3 sector 4) of models/AIS3_proj_ssa_init.nc (the SSA 2019
state = the friction inversion's geometry, inverted shelf B, production Budd p=q=1 C, coupling 3 N):
  J0  friction-only inversion, as ssa_friction_inv_reg_lcurve's production point (cf101 10,
      cf103 100 on grounded ice with obs, cf501 1e-4 off the GL, C in [0.05, 900], floating C 0),
      warm-started from the production C. Reference: same subdomain, BCs and iteration budget.
  J1b as J0 plus MaterialsRheologyBbar as a second control: B free in [cuffey(0 C), cuffey(-70 C)]
      (the shelf inversion's bounds) on grounded ice with obs speed > 50 m/yr (trunks and their
      margins); fixed at B0 elsewhere (slow interior, and floating ice at the inverted shelf B);
      cf502 1e-17 (the shelf inversion's chosen point) where B is free; warm-started from B0.
      Control scaling 1 for both: the raw first-iteration gradient norms are then comparable
      (C 3, B 4.4). The first attempt (J1, scaling [1, 1e6], B free on all grounded ice) stopped
      on dxmin after 3 iterations: the scaled B gradient (norm 4.4e6, 79% of it in ice < 50 m/yr)
      made the first step too short. Its run dir is kept. J0 is reused if its run dir has a lock.
Then stress balance only, for each of J0 and J1: F0 shelves as is, F1 all floating ice removed
(front at the GL, unanchored pieces removed, as check_shelf_removal.py).
Subdomain boundary: true ice margin as extract_friction_inversion_domain (floating -> Neumann,
grounded -> Dirichlet obs); cuts along the sector edge -> Dirichlet obs (extract's default).
Run dirs execution_SSA/AIS3_amujoint_*. Writes models/AIS3_amundsen_joint_inv/{J0,J1}.nc and
models/amundsen_joint_inv_test.npz (refuses to overwrite). Read-only for other model files.
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

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
ODIR = f'{M}/AIS3_amundsen_joint_inv'
OUT = f'{M}/amundsen_joint_inv_test.npz'
NAME = 'AIS3_amujoint_{}'
INV, FWD = ('J0', 'J1b'), ('J0F0', 'J0F1', 'J1bF0', 'J1bF1')
SECTOR = 4
for p_ in [OUT, ODIR] + [f'{EXE}/' + NAME.format(t) for t in INV + FWD if t != 'J0']:
    if os.path.exists(p_):
        raise SystemExit(f'{p_} exists; move it aside first')

md = pyissm.model.io.load_model(f'{M}/AIS3_proj_ssa_init.nc')
md.results.TransientSolution = []
assert type(md.friction).__name__ == 'default' and int(md.friction.coupling) == 3
assert np.all(np.asarray(md.friction.p) == 1) and np.all(np.asarray(md.friction.q) == 1)
x, y = np.asarray(md.mesh.x, float).ravel(), np.asarray(md.mesh.y, float).ravel()
el = np.asarray(md.mesh.elements).astype(int) - 1
ds = xr.open_dataset('assets/larmip3/LARMIP3_sectors-1km.nc')
var = next(v for v in ds.data_vars if ds[v].ndim == 2)
mx, my = np.meshgrid(ds['x'].values, ds['y'].values)
mv = ds[var].values.astype(float)
ok = np.isfinite(mv)
sector = np.rint(NearestNDInterpolator(np.column_stack([mx[ok], my[ok]]), mv[ok])(x, y)).astype(int)

# subdomain: ice elements (as extract_friction_inversion_domain) entirely in the sector
ice_el = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset) < 1
in_el = ice_el & np.all(sector[el] == SECTOR, axis=1)
cut_v = np.zeros(x.size, bool)
cut_v[el[ice_el & ~in_el].ravel()] = True       # vertices touching ice outside the sector
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
mds.stressbalance.spcvx[margin_fl] = np.nan
mds.stressbalance.spcvy[margin_fl] = np.nan
mds.stressbalance.spcvz[margin_fl] = np.nan
print(f'Amundsen subdomain: {nv} vertices; boundary {int(bnd.sum())} = sector cut {int(cut.sum())} (Dirichlet obs) '
      f'+ margin floating {int(margin_fl.sum())} (Neumann) + margin grounded {int((bnd & ~cut & (ol >= 0)).sum())} (Dirichlet obs)', flush=True)

ice = il < 0
gr = ice & (ol > 0)
fl = ice & (ol < 0)
H = np.asarray(mds.geometry.thickness, float).ravel()
vobs = np.asarray(mds.inversion.vel_obs, float).ravel()
C0 = np.asarray(mds.friction.coefficient, float).ravel().copy()
B0 = np.asarray(mds.materials.rheology_B, float).ravel().copy()
Bmin, Bmax = pyissm.tools.materials.cuffey(273.15), pyissm.tools.materials.cuffey(273.15 - 70)
print(f'grounded B0 within bounds: {np.mean((B0[gr] >= Bmin) & (B0[gr] <= Bmax)):.4f} '
      f'(B0 median {np.median(B0[gr]):.3g}; bounds {Bmin:.3g}-{Bmax:.3g})', flush=True)
B0c = np.where(gr, np.clip(B0, Bmin, Bmax), B0)

# shared solver/cluster settings
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
c.queue, c.np, c.memory, c.time = 'normal', 48, 190, 8 * 60
mds.cluster = c
mds.settings.waitonlock = 30  # > 0 so ISSM writes the lock file; batch=True skips pyissm's own wait

# cost masks as ssa_friction_inv_reg_lcurve
mask = (vobs > 0) & (ol >= 0)
el_s = np.asarray(mds.mesh.elements).astype(int) - 1
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
                print(f'{tag}: lock {done[tag]}; NaN {"residue is NaN" in log}; '
                      f'{(log.split("Exit Status:")[1].splitlines()[0].strip() if "Exit Status:" in log else "")}', flush=True)
    print('pending (timed out):', sorted(pending), flush=True)
    return done


# ---- inversions -------------------------------------------------------------------------------
Bfree = gr & (vobs > 50)
print(f'B free on {int(Bfree.sum())} of {int(gr.sum())} grounded vertices (obs > 50 m/yr)', flush=True)
subm = {}
for tag in INV:
    m = copy.deepcopy(mds)
    m.inversion = pyissm.model.classes.inversion.m1qn3(m.inversion)
    m.inversion.iscontrol = 1
    m.inversion.maxsteps = 600
    m.inversion.maxiter = 300
    m.inversion.gttol = 1e-8
    m.friction.coefficient = np.where(fl, 0.05, C0)
    cmin = np.where(fl, 0.0, 0.05)
    cmax = np.where(fl, 0.0, 900.0)
    cols = [np.where(mask, 10.0, 0.0), np.where(mask, 100.0, 0.0), np.where(reg_mask, 1e-4, 0.0)]
    if tag == 'J0':
        m.inversion.control_parameters = ['FrictionCoefficient']
        m.inversion.control_scaling_factors = np.array([1.0])
        m.inversion.min_parameters, m.inversion.max_parameters = cmin, cmax
        m.inversion.cost_functions = [101, 103, 501]
    else:
        m.materials.rheology_B = B0c.copy()
        m.inversion.control_parameters = ['FrictionCoefficient', 'MaterialsRheologyBbar']
        m.inversion.control_scaling_factors = np.array([1.0, 1.0])
        m.inversion.min_parameters = np.column_stack([cmin, np.where(Bfree, Bmin, B0c)])
        m.inversion.max_parameters = np.column_stack([cmax, np.where(Bfree, Bmax, B0c)])
        m.inversion.cost_functions = [101, 103, 501, 502]
        cols.append(np.where(Bfree, 1e-17, 0.0))
    m.inversion.cost_functions_coefficients = np.column_stack(cols)
    m.miscellaneous.name = NAME.format(tag)
    subm[tag] = m
    if os.path.exists(f'{EXE}/{NAME.format(tag)}/{NAME.format(tag)}.lock'):
        print(f'-- {tag}: finished run found, reusing it', flush=True)
        continue
    print(f'-- submitting inversion {tag}', flush=True)
    pyissm.model.execute.solve(m, 'Stressbalance', batch=True, load_only=False, runtime_name=False)
done = wait(INV, 20)
if not all(done.get(t) for t in INV):
    raise SystemExit('an inversion failed; stopping before the forward solves')

os.makedirs(ODIR)
field = {}
for tag in INV:
    m = pyissm.model.execute.solve(subm.pop(tag), 'Stressbalance', load_only=True, runtime_name=False)
    r = m.results.StressbalanceSolution
    Cn = np.asarray(r.FrictionCoefficient, float).ravel()
    Bn = np.asarray(r.MaterialsRheologyBbar, float).ravel() if tag == 'J1b' else B0.copy()
    vel = np.asarray(r.Vel, float).ravel()
    J = np.asarray(r.J, float)
    field[tag] = (Cn, Bn, vel)
    mm = gr & (vobs > 0)
    print(f'{tag}: iterations {J.shape[0]}; J first/last {J[0, -1]:.4g} / {J[-1, -1]:.4g}; grounded RMSE {np.sqrt(np.mean((vel[mm] - vobs[mm]) ** 2)):.1f} m/yr '
          f'(start-state RMSE {np.sqrt(np.mean((np.asarray(mds.initialization.vel, float).ravel()[mm] - vobs[mm]) ** 2)):.1f})', flush=True)
    pyissm.model.io.save_model(m, f'{ODIR}/{tag}.nc')
    del m

# ---- shelf removal on both --------------------------------------------------------------------
glv = np.zeros(nv, bool)
glv[el_s[np.any(gr[el_s], axis=1) & np.any(fl[el_s], axis=1)].ravel()] = True
glv &= gr
xs, ys = np.asarray(mds.mesh.x, float).ravel(), np.asarray(mds.mesh.y, float).ravel()
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


removed = None
for tag in FWD:
    jt = tag[:-2]
    Cn, Bn, _ = field[jt]
    m = copy.deepcopy(mds)
    m.inversion.iscontrol = 0
    m.stressbalance.requested_outputs = ['default']
    m.friction.coefficient = Cn.copy()
    m.materials.rheology_B = Bn.copy()
    m.cluster.time = 60
    if tag.endswith('F1'):
        gone = fl | detached(ice & ~fl, ol, Cn)
        removed = gone if removed is None else removed | gone
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
    subm[tag] = m
done = wait(FWD, 4)
v = {}
for tag, okk in done.items():
    if okk:
        m = pyissm.model.execute.solve(subm.pop(tag), 'Stressbalance', load_only=True, runtime_name=False)
        v[tag] = np.asarray(m.results.StressbalanceSolution.Vel, float).ravel().copy()
        del m

# ---- diagnostics ------------------------------------------------------------------------------
keep = gr & ~removed
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


sgx, sgy = vgrad(np.asarray(mds.geometry.surface, float).ravel())
taud = float(mds.materials.rho_ice) * float(mds.constants.g) * H * np.hypot(sgx, sgy)
Nf = np.maximum(np.asarray(mds.friction.effective_pressure, float).ravel(), 0.07 * float(mds.materials.rho_ice) * float(mds.constants.g) * H)
yts = float(mds.constants.yts)
# shear-margin proxy: speed-gradient magnitude of the J0 velocity (1/yr)
ugx, ugy = vgrad(field['J0'][2])
shear = np.hypot(ugx, ugy)

Bn = field['J1b'][1]
fast = keep & (field['J0'][2] > 100)
print('\n== J1 grounded B / B0 on fast grounded ice (J0 speed > 100 m/yr), by speed-gradient quintile (margin proxy):')
qs = np.percentile(shear[fast], [0, 20, 40, 60, 80, 100])
for i in range(5):
    mm = fast & (shear >= qs[i]) & (shear <= qs[i + 1])
    print(f'   |grad u| {qs[i]:.2e}-{qs[i + 1]:.2e} /yr: n {int(mm.sum()):6d}, B/B0 median {np.median(Bn[mm] / B0[mm]):.2f} '
          f'[p10 {np.percentile(Bn[mm] / B0[mm], 10):.2f}, p90 {np.percentile(Bn[mm] / B0[mm], 90):.2f}]')
mm = gr & (vobs > 0)
print(f'   all grounded: B/B0 median {np.median(Bn[gr] / B0[gr]):.2f}; at lower bound {np.mean(np.isclose(Bn[gr], Bmin)):.3f}, '
      f'upper {np.mean(np.isclose(Bn[gr], Bmax)):.3f}')

for jt in INV:
    Cn, _, vel = field[jt]
    tb = Cn ** 2 * Nf * vel / yts
    for nm, mm in (('fast trunks > 1000 m/yr, < 20 km of GL', keep & (vel > 1000) & (dist < 20e3) & (taud > 1e3)),
                   ('grounded > 100 m/yr, < 20 km of GL', keep & (vel > 100) & (dist < 20e3) & (taud > 1e3))):
        print(f'{jt}: drag/driving, {nm}: median {np.median(tb[mm] / taud[mm]):.2f} (n {int(mm.sum())})')

print('\n== shelf removal: F1/F0 speed ratio on grounded ice kept, J0 speed > 100 m/yr; median (flux-weighted, capped at 10) | n > 10x')
for jt in INV:
    if f'{jt}F0' not in v or f'{jt}F1' not in v:
        continue
    a, b = v[f'{jt}F1'], v[f'{jt}F0']
    row = []
    for lo, hi in ((0, 5e3), (5e3, 20e3), (20e3, 50e3), (50e3, 150e3)):
        mm = keep & (dist >= lo) & (dist < hi) & (b > 100)
        r = a[mm] / b[mm]
        fw = np.sum(np.minimum(r, 10) * b[mm] * H[mm]) / np.sum(b[mm] * H[mm])
        row.append(f'{lo / 1e3:.0f}-{hi / 1e3:.0f} km {np.median(r):.2f} ({fw:.2f}) | {int((r > 10).sum())}')
    mm = keep & (dist < 20e3) & (b > 1000)
    row.append(f'fast trunks < 20 km {np.median(a[mm] / b[mm]):.2f} (n {int(mm.sum())})')
    print(f'   {jt}: ' + ';  '.join(row))
np.savez(OUT, ev=ev, vobs=vobs, gr=gr, fl=fl, dist=dist, removed=removed, B0=B0, shear=shear, taud=taud,
         **{f'C_{k}': f[0] for k, f in field.items()}, **{f'B_{k}': f[1] for k, f in field.items()},
         **{f'vel_{k}': f[2] for k, f in field.items()}, **{f'fwd_{k}': a for k, a in v.items()})
print(f'\nsaved {OUT}')
print('DONE')
