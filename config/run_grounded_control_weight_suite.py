"""Submit a compact control/velocity-weight diagnostic inversion suite.

The already-running grounded-only joint inversion at (101, 103) = (10, 100)
is the sixth cell of this experiment.  This script submits the remaining five:

    controls       (101, 103)
    C only         (10, 100), (10, 5)
    B only         (10, 100), (10, 5)
    C and B                    (10, 5)

The 10/5 pair approximately balances the initial 101 and 103 contributions
observed in the 10/100 run.  Cost functions 101 and 103 are evaluated only on
grounded ice; floating shelves remain in the stress-balance domain.
"""

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyissm


ROOT = Path("/g/data/au88/jh7060/ACCESS-AIS3")
MODEL_DIR = ROOT / "models"
ISSM_DIR = (
    "/g/data/vk83/apps/spack/1.1/release/linux-x86_64/"
    "issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd"
)
os.environ["ISSM_DIR"] = ISSM_DIR
os.chdir(ROOT)


def load_base_model():
    """Build the same corrected model used by the grounded joint baseline."""
    print("-- Loading temperature-derived grounded B --", flush=True)
    md = pyissm.model.io.load_model(MODEL_DIR / "AIS3_param.nc")
    b_prior = np.asarray(md.materials.rheology_B).ravel()
    assert np.nanmax(b_prior) / np.nanmin(b_prior) > 2, "param has uniform B"

    print("-- Applying the inverted shelf B --", flush=True)
    shelf_source = pyissm.model.io.load_model(MODEL_DIR / "AIS3_param.nc")
    shelf_source.mask.ice_levelset = pyissm.model.param.kill_icebergs(shelf_source)
    shelf = shelf_source.extract(
        (shelf_source.mask.ocean_levelset < 0)
        & (shelf_source.mask.ice_levelset < 0)
    )
    load_cluster = pyissm.model.classes.cluster.gadi()
    load_cluster.codepath = ISSM_DIR + "/bin"
    load_cluster.executionpath = str(ROOT / "execution_newB_rheology")
    load_cluster.login = "jh7060"
    load_cluster.project = "au88"
    load_cluster.storage = "gdata/au88"
    shelf.cluster = load_cluster
    shelf.settings.waitonlock = 0
    shelf.inversion.iscontrol = 0
    shelf.miscellaneous.name = "run_001_1_10_1e-17"
    shelf_result = pyissm.model.execute.solve(
        shelf,
        "Stressbalance",
        load_only=True,
        runtime_name=False,
        check_consistency=False,
    )
    shelf_vertices = np.asarray(shelf_result.mesh.extractedvertices).ravel() - 1
    md.materials.rheology_B[shelf_vertices] = np.asarray(
        shelf_result.results.StressbalanceSolution.MaterialsRheologyBbar
    ).ravel()

    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)
    print("-- Applying the consistent 100 m thin-ice floor --", flush=True)
    rho_i = md.materials.rho_ice
    rho_w = md.materials.rho_water
    thickness = np.asarray(md.geometry.thickness).ravel().copy()
    bed = np.asarray(md.geometry.bed).ravel()
    ocean = np.asarray(md.mask.ocean_levelset).ravel()
    print(f"   raising {int((thickness < 100).sum())} vertices", flush=True)
    thickness = np.maximum(thickness, 100.0)
    floating = ocean < 0
    base = np.empty_like(thickness)
    surface = np.empty_like(thickness)
    base[floating] = -thickness[floating] * rho_i / rho_w
    surface[floating] = thickness[floating] * (1.0 - rho_i / rho_w)
    base[~floating] = bed[~floating]
    surface[~floating] = bed[~floating] + thickness[~floating]
    md.geometry.thickness = thickness
    md.geometry.base = base
    md.geometry.surface = surface

    md.inversion.iscontrol = 1
    md.verbose.solution = 0
    md.verbose.qmu = 0
    md.verbose.control = 1
    effective_pressure = md.friction.effective_pressure.copy()
    effective_pressure[effective_pressure < 0] = 0
    md.friction.effective_pressure = effective_pressure
    md.friction.effective_pressure_limit = 0.07

    print("-- Extracting all ice with Neumann shelf fronts --", flush=True)
    ice_elements = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset)
    mds = md.extract(ice_elements < 1)
    boundary = mds.mesh.vertexonboundary.astype(bool)
    ocean = np.asarray(mds.mask.ocean_levelset).ravel()
    shelf_front = boundary & (ocean < 0)
    mds.stressbalance.spcvx[shelf_front] = np.nan
    mds.stressbalance.spcvy[shelf_front] = np.nan
    mds.stressbalance.spcvz[shelf_front] = np.nan
    mds.mask.ice_levelset[shelf_front] = 0

    mds.friction = pyissm.model.classes.friction.default(mds.friction)
    mds.friction.p = np.full(mds.mesh.numberofelements, 3.0)
    mds.friction.q = np.full(mds.mesh.numberofelements, 3.0)
    mds.friction.coupling = 3
    effective_pressure = mds.friction.effective_pressure.copy()
    effective_pressure[effective_pressure < 0] = 0
    mds.friction.effective_pressure = effective_pressure
    mds.friction.effective_pressure_limit = 0.07

    floating = ocean < 0
    mds.friction.coefficient = np.full(mds.mesh.numberofvertices, 1.8)
    mds.friction.coefficient[floating] = 0.05

    mds.transient = pyissm.model.classes.transient.deactivate_all(mds.transient)
    mds.stressbalance.restol = 0.01
    mds.stressbalance.reltol = 0.1
    mds.stressbalance.abstol = np.nan
    mds.settings.solver_residue_threshold = 1e-3
    mds.settings.waitonlock = 0
    return mds


def configure_cluster(md, mode):
    execution_dir = ROOT / f"execution_grounded_suite_{mode}"
    execution_dir.mkdir(exist_ok=True)
    cluster = pyissm.model.classes.cluster.gadi()
    cluster.codepath = ISSM_DIR + "/bin"
    cluster.executionpath = str(execution_dir)
    cluster.storage = "gdata/au88+gdata/vk83"
    cluster.moduleuse = ["/g/data/vk83/modules/"]
    cluster.moduleload = ["access-issm/2025.11.0"]
    cluster.np = 32
    cluster.memory = 190
    cluster.time = 60 * 48
    cluster.login = "jh7060"
    cluster.project = "au88"
    md.cluster = cluster


def configure_controls(md, mode):
    nv = md.mesh.numberofvertices
    ocean = np.asarray(md.mask.ocean_levelset).ravel()
    floating = ocean < 0
    b_current = np.asarray(md.materials.rheology_B).ravel()
    b_min = pyissm.tools.materials.cuffey(273.15)
    b_max = pyissm.tools.materials.cuffey(273.15 - 70)

    md.inversion = pyissm.model.classes.inversion.m1qn3(md.inversion)
    md.inversion.maxsteps = 200
    md.inversion.maxiter = 200

    if mode == "friction":
        md.inversion.control_parameters = ["FrictionCoefficient"]
        lower = np.full(nv, 0.1)
        upper = np.full(nv, 10.0)
        lower[floating] = 0.0
        upper[floating] = 0.0
        md.inversion.min_parameters = lower
        md.inversion.max_parameters = upper
        md.inversion.control_scaling_factors = np.array([1.8])
    elif mode == "rheology":
        md.inversion.control_parameters = ["MaterialsRheologyBbar"]
        lower = np.full(nv, b_min)
        upper = np.full(nv, b_max)
        lower[floating] = b_current[floating]
        upper[floating] = b_current[floating]
        md.inversion.min_parameters = lower
        md.inversion.max_parameters = upper
        md.inversion.control_scaling_factors = np.array(
            [float(np.nanmedian(b_current[~floating]))]
        )
    elif mode == "joint":
        md.inversion.control_parameters = [
            "FrictionCoefficient",
            "MaterialsRheologyBbar",
        ]
        lower = np.zeros((nv, 2))
        upper = np.zeros((nv, 2))
        lower[:, 0] = 0.1
        upper[:, 0] = 10.0
        lower[floating, 0] = 0.0
        upper[floating, 0] = 0.0
        lower[:, 1] = b_min
        upper[:, 1] = b_max
        lower[floating, 1] = b_current[floating]
        upper[floating, 1] = b_current[floating]
        md.inversion.min_parameters = lower
        md.inversion.max_parameters = upper
        md.inversion.control_scaling_factors = np.array(
            [1.8, float(np.nanmedian(b_current[~floating]))]
        )
    else:
        raise ValueError(mode)


base = load_base_model()
ocean = np.asarray(base.mask.ocean_levelset).ravel()
ice = np.asarray(base.mask.ice_levelset).ravel()
velocity = np.asarray(base.inversion.vel_obs).ravel()
grounded_ice = (ice < 0) & (ocean > 0)
observation_mask = (velocity > 0) & grounded_ice
regularization_mask = ocean >= 0

print(
    f"   grounded objective vertices: {int(observation_mask.sum())}; "
    f"grounded observed RMS: {np.sqrt(np.nanmean(velocity[observation_mask] ** 2)):.1f}",
    flush=True,
)

experiments = {
    "friction": [(10.0, 100.0), (10.0, 5.0)],
    "rheology": [(10.0, 100.0), (10.0, 5.0)],
    # (10, 100) joint is already running as job 174385101.
    "joint": [(10.0, 5.0)],
}

for mode, velocity_pairs in experiments.items():
    model = copy.deepcopy(base)
    configure_controls(model, mode)
    configure_cluster(model, mode)

    grids = []
    for run_id, (coefficient_101, coefficient_103) in enumerate(
        velocity_pairs, start=1
    ):
        coefficients = {101: [coefficient_101], 103: [coefficient_103]}
        if mode in {"friction", "joint"}:
            coefficients[501] = [1e-8]
        if mode in {"rheology", "joint"}:
            coefficients[502] = [1e-17]
        grids.append(
            pyissm.inversion.sensitivity.build_parameter_grid(
                coefficients, initial_run_id=run_id
            )
        )
    grid = pd.concat(grids, ignore_index=True)

    coefficient_masks = {101: observation_mask, 103: observation_mask}
    if mode in {"friction", "joint"}:
        coefficient_masks[501] = regularization_mask
    if mode in {"rheology", "joint"}:
        coefficient_masks[502] = regularization_mask

    print(f"-- Submitting {len(grid)} {mode} run(s) --", flush=True)
    pyissm.inversion.sensitivity.parameter_sensitivity(
        model,
        grid,
        output_dir=str(MODEL_DIR / f"AIS3_grounded_suite_{mode}"),
        run=True,
        load_only=False,
        coeff_masks=coefficient_masks,
    )

print("SUBMITTED ALL FIVE DIAGNOSTIC RUNS", flush=True)
