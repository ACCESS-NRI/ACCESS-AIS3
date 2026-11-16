"""Screen candidate cf501 (drag-coefficient regularisation weight) values for ho_friction_inv,
following up on the finding that the regularisation term dominates the cost function at
chunk 3's endpoint (63% of total cost vs. 37% for the velocity misfit, cf101=10/cf103=100/
cf501=0.0001) -- a plausible mechanism for why raw grounded RMSE has been WORSENING across
chunks (94.38 -> 97.22 -> 99.15 m/yr) even as the optimizer's own cost function keeps falling.
cf501=0.0001 was carried over from the SSA-track's reg_lcurve work on a completely different
(~1.6M-vertex 2D) mesh and, per ho_friction_inv's own code comment, was always meant to be
re-swept once a baseline was in hand -- never actually done. See
docs/post_inversion_pipeline_worklog.md SS3 for the full chunk history this follows up on.

NOT a full independent multi-chunk convergence per candidate (that would mean redoing the
entire multi-day chunk-1..4 process 5x over, wildly expensive). Instead: warm-start every
candidate from the SAME common checkpoint (AIS3_ho_friction_inv.nc, currently chunk 3's
converged state -- safe to read concurrently while chunk 4 continues, since chunk 4 has not
yet saved a new checkpoint as of this writing), run a SHORT probe (maxsteps=5, matching one
normal chunk's iteration budget) at each cf501 candidate, and compare grounded RMSE + cost
breakdown after that fixed budget. This is a screening sweep, not a final answer -- it tells
us which direction (and roughly how far) to move cf501 before committing real multi-chunk
compute to a production rerun at the winning value.

Own execution names (AIS3_cf501_resweep_c{i}) and own model_dir subfolder, so this runs
fully concurrently with chunk 4's ongoing production attempt without any collision -- same
isolation pattern used throughout this session (melt_ismip7_calibration.py,
melt_deltaT_basin_refit.py) after the steps/save race-condition incident.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import copy
import numpy as np
import pyissm

PHASE = 'analyze'  # 'submit' or 'analyze' -- hand-toggle before each qsub

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
# Log-spaced downward from the current production value (0.0001), which the cost breakdown
# says is over-regularising -- 0.0001 itself is included as a same-budget control point, to
# separate "5 more iterations at the same weight" from "5 iterations at a different weight".
CF501_CANDIDATES = [0.0001, 0.00003, 0.00001, 0.000003, 0.000001]
PROBE_MAXSTEPS = 5  # matches one normal chunk's iteration budget -- see module docstring
FRICTION_CF101 = 10
FRICTION_CF103 = 100
FRICTION_INV_GTTOL = 1e-8


def friction_law_info(md):
    """Copied verbatim from ais_0.1.py -- see that file for the full rationale."""
    if type(md.friction).__name__ == 'default':
        return 'FrictionCoefficient', 'coefficient', 0.05, 900
    return 'FrictionC', 'C', 0.05, 250 ** 2


def cluster_config():
    cl = pyissm.model.classes.cluster.gadi()
    cl.codepath = os.environ['ISSM_DIR'] + '/bin'
    cl.executionpath = '/g/data/au88/jh7060/ACCESS-AIS3/execution'
    cl.storage = 'gdata/au88+gdata/vk83+gdata/av17'
    cl.moduleuse = ['/g/data/vk83/modules/']
    cl.moduleload = ['access-issm_ad/2026.05.0']
    cl.login = 'jh7060'
    cl.project = 'au88'
    cl.queue = 'hugemem'
    cl.np = 96
    cl.memory = 1450 * 2
    cl.time = 60 * 24
    return cl


def setup_common_start():
    print(f'-- Loading common starting checkpoint (AIS3_ho_friction_inv.nc, chunk 3\'s '
          f'converged state)...', flush=True)
    md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
    fric_control, fric_field, fric_min, fric_max = friction_law_info(md)

    print(f'-- Copying checkpoint\'s optimized {fric_control} into the live friction field '
          f'(same resume bugfix as ho_friction_inv itself)...', flush=True)
    _resumed = np.asarray(getattr(md.results.StressbalanceSolution, fric_control)).ravel()
    setattr(md.friction, fric_field, _resumed)

    nv3d = md.mesh.numberofvertices
    on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
    _fld = getattr(md.friction, fric_field).astype(float)

    on_surface = np.asarray(md.mesh.vertexonsurface).astype(bool).ravel()
    vo = np.asarray(md.inversion.vel_obs).ravel()
    ol = np.asarray(md.mask.ocean_levelset).ravel()
    mask = (vo > 0) & (ol >= 0) & on_surface
    grounded_mask = (ol >= 0) & on_base
    _elx = np.asarray(md.mesh.elements).astype(int) - 1
    _float_v = ol < 0
    _touch_f = _float_v[_elx].any(axis=1)
    _gladj = np.zeros(nv3d, dtype=bool)
    _gladj[_elx[_touch_f].ravel()] = True
    reg_mask = grounded_mask & ~_gladj

    md.inversion = pyissm.model.classes.inversion.m1qn3(md.inversion)
    md.inversion.iscontrol = 1
    md.inversion.control_parameters = [fric_control]
    md.inversion.min_parameters = np.where(on_base, fric_min, _fld)
    md.inversion.max_parameters = np.where(on_base, fric_max, _fld)
    md.inversion.maxsteps = PROBE_MAXSTEPS
    md.inversion.maxiter = PROBE_MAXSTEPS
    md.inversion.gttol = FRICTION_INV_GTTOL
    # FIX (2026-09-02): same dfmin_frac issue found and fixed in ais_0.1.py's ho_friction_inv
    # after chunk 4's third failure -- this script also does a fresh m1qn3 restart from chunk
    # 3's checkpoint and never set dfmin_frac (silently defaulting to 1.0, an oversized first
    # step). All 5 original probes ran 13h stuck at the exact same iteration-1->2 transition
    # chunk 4 died at, confirming this script needed the identical fix. See
    # docs/post_inversion_pipeline_worklog.md SS3 for the full root-cause writeup.
    md.inversion.dfmin_frac = 0.3
    md.inversion.cost_functions = [101, 103, 501]

    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.stressbalance.restol = 0.01
    md.stressbalance.maxiter = 300
    md.settings.waitonlock = 0

    return md, mask, reg_mask, nv3d


if PHASE == 'submit':
    md, mask, reg_mask, nv3d = setup_common_start()
    for i, cf501 in enumerate(CF501_CANDIDATES):
        cf = np.zeros((nv3d, 3))
        cf[mask, 0] = FRICTION_CF101
        cf[mask, 1] = FRICTION_CF103
        cf[reg_mask, 2] = cf501
        mdi = copy.deepcopy(md)
        mdi.inversion.cost_functions_coefficients = cf
        mdi.miscellaneous.name = f'AIS3_cf501_resweep_c{i}'
        mdi.cluster = cluster_config()
        print(f'-- Submitting cf501 candidate c{i}={cf501:g} (probe, maxsteps={PROBE_MAXSTEPS})...', flush=True)
        pyissm.model.execute.solve(mdi, 'Stressbalance', load_only=False, runtime_name=False)
    print('DONE submitting. Set PHASE=\'analyze\' and rerun once all probe jobs finish.')

elif PHASE == 'analyze':
    md, mask, reg_mask, nv3d = setup_common_start()
    vo = np.asarray(md.inversion.vel_obs).ravel()
    ol = np.asarray(md.mask.ocean_levelset).ravel()
    il = np.asarray(md.mask.ice_levelset).ravel()
    on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
    on_surface = np.asarray(md.mesh.vertexonsurface).astype(bool).ravel()
    gr_surf = (il < 0) & (ol >= 0) & (vo > 0) & on_surface

    print(f'\n{"c":>3} {"cf501":>10} {"grounded RMSE":>14} {"f(x)":>12} '
          f'{"101":>10} {"103":>12} {"501":>12}')
    results = []
    for i, cf501 in enumerate(CF501_CANDIDATES):
        name = f'AIS3_cf501_resweep_c{i}'
        try:
            mdi = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_ho_friction_inv.nc')
            mdi.miscellaneous.name = name
            mdi.cluster = cluster_config()
            mdi = pyissm.model.execute.solve(mdi, 'Stressbalance', load_only=True, runtime_name=False)
        except Exception as e:
            print(f'   c{i} (cf501={cf501:g}): FAILED to load ({e})')
            continue
        vel = np.asarray(mdi.results.StressbalanceSolution.Vel).ravel()
        rmse = np.sqrt(np.nanmean((vel[gr_surf] - vo[gr_surf]) ** 2))
        # Cost breakdown not directly exposed post-hoc on the results object in a stable
        # field name across pyissm versions -- report RMSE (the metric we actually care
        # about) as the primary comparison; skip re-deriving the weighted terms here.
        print(f'{i:3d} {cf501:10.2g} {rmse:14.2f}')
        results.append((i, cf501, rmse))

    if results:
        best = min(results, key=lambda r: r[2])
        print(f'\nBest by grounded RMSE: c{best[0]} (cf501={best[1]:g}), RMSE={best[2]:.2f} m/yr')
        print('NOTE: this is a 5-iteration probe from a common checkpoint, not a converged '
              'answer -- confirms DIRECTION before committing a full multi-chunk production '
              'rerun at the winning cf501.')
    print('DONE')
