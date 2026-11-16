"""Follow-up to check_collapse_2089_sb.py: a stress-balance-only solve on the 2089 collapse state
converges (V1-V3), yet the transient sub-solve from the same state failed four times on its first
stress balance (worklog 6.15). One-step transients (2089.0 -> 2089.1) with the projection's
transient settings, to find which ingredient matters:
  T1 as saved (static forcing from the checkpoint, ismip6 melt, grounding line on)
  T2 initialization vx, vy = 0
  T3 basal forcing: default class, zero melt
  T4 isgroundingline = 0
All four are submitted at once (batch) and polled. Run dirs execution_SSA/AIS3_collapse2089_tr_T*.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
import time
import copy
import numpy as np
import pyissm

M = '/g/data/au88/jh7060/ACCESS-AIS3/models'
EXE = '/g/data/au88/jh7060/ACCESS-AIS3/execution_SSA'
md0 = pyissm.model.io.load_model(f'{M}/AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2089.nc')
md0.results.TransientSolution = []
nv = md0.mesh.numberofvertices
print(f'basalforcings {type(md0.basalforcings).__name__}; tf columns {np.asarray(md0.basalforcings.tf[0]).shape}; '
      f'smb {np.asarray(md0.smb.mass_balance).shape}; time {md0.timestepping.start_time}-{md0.timestepping.final_time}', flush=True)


def setup(tag, zero_vel=False, no_melt=False, no_gl=False):
    md = copy.deepcopy(md0)
    if zero_vel:
        for f in ('vx', 'vy', 'vel'):
            setattr(md.initialization, f, np.zeros(nv))
    if no_melt:
        md.basalforcings = pyissm.model.classes.basalforcings.default()
        md.basalforcings.floatingice_melting_rate = np.zeros(nv)
        md.basalforcings.groundedice_melting_rate = np.zeros(nv)
    md.inversion.iscontrol = 0
    md.verbose.solution = 1
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.isthermal = 0
    md.transient.isgroundingline = 0 if no_gl else 1
    md.groundingline.migration = 'SubelementMigration'
    md.transient.requested_outputs = ['Thickness', 'MaskOceanLevelset', 'Vel']
    md.timestepping.start_time = 2089.0
    md.timestepping.final_time = 2089.1
    md.timestepping.time_step = 0.1
    md.settings.output_frequency = 1
    md.settings.solver_residue_threshold = 1e-3
    md.miscellaneous.name = f'AIS3_collapse2089_tr_{tag}'
    c = pyissm.model.classes.cluster.gadi()  # as in ais_0.1_SSA.py
    c.codepath = os.environ['ISSM_DIR'] + '/bin'
    c.executionpath = EXE
    c.storage = 'gdata/au88+gdata/vk83+gdata/av17'
    c.moduleuse = ['/g/data/vk83/modules/']
    c.moduleload = ['access-issm_ad/2026.05.0']
    c.login = 'jh7060'
    c.project = 'au88'
    c.np, c.memory, c.queue, c.time = 48, 190, 'normal', 60
    md.cluster = c
    md.settings.waitonlock = 240  # > 0 so ISSM writes the lock file; batch=True skips pyissm's own wait
    return pyissm.model.execute.solve(md, 'Transient', batch=True, load_only=False, runtime_name=False)


runs = {t: setup(t, **kw) for t, kw in [('T1', {}), ('T2', dict(zero_vel=True)),
                                          ('T3', dict(no_melt=True)), ('T4', dict(no_gl=True))]}
pending, t0 = set(runs), time.time()
while pending and time.time() - t0 < 5 * 3600:
    time.sleep(60)
    for t in sorted(pending):
        d = f'{EXE}/AIS3_collapse2089_tr_{t}/AIS3_collapse2089_tr_{t}'
        log = open(d + '.outlog', errors='replace').read() if os.path.exists(d + '.outlog') else ''
        if os.path.exists(d + '.lock') or 'Exit Status:' in log:
            pending.discard(t)
            print(f'{t}: lock {os.path.exists(d + ".lock")}; NaN {"residue is NaN" in log}; residue-too-high '
                  f'{log.count("residue too high")}; nonlinear cap {log.count("exceeded")}; '
                  f'steps {log.count("time [yr]")}', flush=True)
print('pending (timed out):', sorted(pending))
print('DONE')
