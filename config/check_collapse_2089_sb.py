"""Collapse run: the 2089 sub-solve fails on its first stress balance (residue 1.9e-3 at threshold
1e-3; NaN at threshold 1e-2; reproducible, worklog 6.15). Which change to the 2089 starting state
lets the SSA stress balance converge?

Stress-balance-only solves on models/AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2089.nc (the state
the failing solve started from, collapse at 2089 applied):
  V1 as saved                       (should reproduce the failure)
  V2 initialization vx, vy = 0      (the 2088 end velocities, up to 10 km/yr, as the Picard start)
  V3 all unanchored ice removed     (every piece without an anchor element, and every ice vertex in
                                     no fully-ice element -- not only those touching the collapse)
  V4 V2 + V3
Same SSA physics as the projection; melt off (stress balance only). Read-only for model files;
run directories execution_SSA/AIS3_collapse2089_sb_V*. Prints residue/NaN outcome and speed stats.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
import sys
import copy
import numpy as np
import pyissm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, '/g/data/au88/jh7060/ACCESS-AIS3/config')
M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'

md0 = pyissm.model.io.load_model(f'{M}/AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2089.nc')
md0.results.TransientSolution = []
nv = md0.mesh.numberofvertices
el = np.asarray(md0.mesh.elements).astype(int) - 1
fc = np.asarray(md0.friction.coefficient, float).ravel()


def unanchored(ice, ol):
    ie = np.where(np.all(ice[el], axis=1))[0]
    e = el[ie]
    ed = np.sort(np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]]), axis=1)
    key = ed[:, 0].astype(np.int64) * nv + ed[:, 1]
    own = np.tile(np.arange(ie.size), 3)
    o = np.argsort(key, kind='stable')
    same = key[o][1:] == key[o][:-1]
    _, lab = connected_components(coo_matrix((np.ones(same.sum()), (own[o][:-1][same], own[o][1:][same])),
                                             shape=(ie.size, ie.size)), directed=False)
    anch = np.zeros(lab.max() + 1, bool)
    np.logical_or.at(anch, lab, np.all((ol[e] > 0) & (fc[e] > 0), axis=1))
    held = np.zeros(nv, bool)
    held[e[anch[lab]].ravel()] = True
    return ice & ~held


def variant(tag, zero_vel, clean):
    md = copy.deepcopy(md0)
    il = np.asarray(md.mask.ice_levelset, float).ravel().copy()
    ol = np.asarray(md.mask.ocean_levelset, float).ravel().copy()
    if clean:
        gone = unanchored(il < 0, ol)
        il[gone] = 1.0
        md.mask.ice_levelset = il
        H = np.asarray(md.geometry.thickness, float).ravel().copy()
        bed = np.asarray(md.geometry.bed, float).ravel()
        H[gone] = float(md.masstransport.min_thickness)
        hfl = -H * float(md.materials.rho_ice) / float(md.materials.rho_water)
        base = np.asarray(md.geometry.base, float).ravel().copy()
        fl = gone & (bed < hfl)
        base[gone] = np.where(fl[gone], hfl[gone], bed[gone])
        ol[gone] = np.where(fl[gone], -1.0, np.where(ol[gone] > 0, ol[gone], 1.0))
        md.mask.ocean_levelset = ol
        md.geometry.thickness, md.geometry.base, md.geometry.surface = H, base, base + H
        print(f'{tag}: removed {int(gone.sum())} unanchored ice vertices', flush=True)
    if zero_vel:
        for f in ('vx', 'vy', 'vel'):
            setattr(md.initialization, f, np.zeros(nv))
    md.basalforcings = pyissm.model.classes.basalforcings.default()
    md.basalforcings.floatingice_melting_rate = np.zeros(nv)
    md.basalforcings.groundedice_melting_rate = np.zeros(nv)
    md.inversion.iscontrol = 0
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.stressbalance.requested_outputs = ['default']
    md.settings.solver_residue_threshold = 1e-3
    md.verbose.solution = 1
    md.miscellaneous.name = f'AIS3_collapse2089_sb_{tag}'
    md.cluster = pyissm.model.classes.cluster.gadi()  # as in ais_0.1_SSA.py
    md.cluster.codepath = os.environ['ISSM_DIR'] + '/bin'
    md.cluster.executionpath = EXE
    md.cluster.storage = 'gdata/au88+gdata/vk83+gdata/av17'
    md.cluster.moduleuse = ['/g/data/vk83/modules/']
    md.cluster.moduleload = ['access-issm_ad/2026.05.0']
    md.cluster.login = 'jh7060'
    md.cluster.project = 'au88'
    md.cluster.np = 48
    md.cluster.memory = 190
    md.cluster.queue = 'normal'
    md.cluster.time = 60
    md.settings.waitonlock = 180
    try:
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only=False, runtime_name=False)
        s = md.results.StressbalanceSolution
        v = np.asarray(s.Vel, float).ravel()
        ice = np.asarray(md.mask.ice_levelset).ravel() < 0
        print(f'{tag}: CONVERGED; ice speed median {np.nanmedian(v[ice]):.1f}, p99.9 {np.nanpercentile(v[ice], 99.9):.0f}, '
              f'max {np.nanmax(v[ice]):.0f} m/yr; NaN {int(np.isnan(v).sum())}', flush=True)
    except Exception as ex:
        print(f'{tag}: FAILED to load results ({type(ex).__name__}: {ex})', flush=True)
    log = f'{EXE}/{md.miscellaneous.name}/{md.miscellaneous.name}.outlog'
    if os.path.exists(log):
        txt = open(log, errors='replace').read()
        print(f'{tag}: outlog: NaN {"Solver residue is NaN" in txt}; residue-too-high {txt.count("residue too high")}; '
              f'nonlinear cap {txt.count("exceeded")}', flush=True)


print(f'friction coupling {md0.friction.coupling}, p unique {np.unique(np.asarray(md0.friction.p))[:3]}; '
      f'init vel max {np.max(np.asarray(md0.initialization.vel)):.0f}', flush=True)
for tag, zv, cl in [('V1', False, False), ('V2', True, False), ('V3', False, True), ('V4', True, True)]:
    variant(tag, zv, cl)
print('DONE')
