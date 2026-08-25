"""Submit a four-cell, bounded and regularized friction-only experiment."""

from pathlib import Path


REFERENCE = Path(
    "/g/data/au88/jh7060/ACCESS-AIS3/config/"
    "run_grounded_control_weight_suite.py"
)
source = REFERENCE.read_text()

# Reuse the validated model-construction and control-configuration functions,
# but replace the original top-level multi-control campaign with this focused
# friction-only grid.
marker = "\nbase = load_base_model()"
if source.count(marker) != 1:
    raise RuntimeError(f"Expected one top-level base-model marker in {REFERENCE}")
source = source.split(marker, 1)[0]

replacements = {
    'execution_dir = ROOT / f"execution_grounded_suite_{mode}"': (
        'execution_dir = ROOT / "execution_grounded_friction_regularized"'
    ),
    "lower = np.full(nv, 0.1)": "lower = np.full(nv, 0.5)",
    "upper = np.full(nv, 10.0)": "upper = np.full(nv, 5.0)",
}
for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}; found {count}")
    source = source.replace(old, new)

experiment = r'''
base = load_base_model()
ocean = np.asarray(base.mask.ocean_levelset).ravel()
ice = np.asarray(base.mask.ice_levelset).ravel()
velocity = np.asarray(base.inversion.vel_obs).ravel()
grounded_ice = (ice < 0) & (ocean > 0)
observation_mask = (velocity > 0) & grounded_ice

# Do not penalize the imposed C discontinuity across the grounding line.
elements = np.asarray(base.mesh.elements, dtype=int) - 1
element_ocean = ocean[elements]
crosses_grounding_line = np.any(element_ocean < 0, axis=1) & np.any(
    element_ocean >= 0, axis=1
)
gl_adjacent = np.zeros(base.mesh.numberofvertices, dtype=bool)
gl_adjacent[np.unique(elements[crosses_grounding_line].ravel())] = True
regularization_mask = (ocean >= 0) & ~gl_adjacent

print(
    f"grounded objective vertices={int(observation_mask.sum())}; "
    f"regularized vertices={int(regularization_mask.sum())}; "
    f"excluded GL-adjacent vertices={int(gl_adjacent.sum())}",
    flush=True,
)

model = copy.deepcopy(base)
configure_controls(model, "friction")
configure_cluster(model, "friction")

grid = pyissm.inversion.sensitivity.build_parameter_grid(
    {101: [10.0], 103: [0.5, 1.0], 501: [1e-3, 1e-2]}
)
print("-- Submitting four regularized friction-only runs --", flush=True)
print(grid.to_string(index=False), flush=True)
pyissm.inversion.sensitivity.parameter_sensitivity(
    model,
    grid,
    output_dir=str(MODEL_DIR / "AIS3_grounded_friction_regularized"),
    run=True,
    load_only=False,
    coeff_masks={
        101: observation_mask,
        103: observation_mask,
        501: regularization_mask,
    },
)
print("SUBMITTED ALL FOUR REGULARIZED FRICTION RUNS", flush=True)
'''

exec(
    compile(source + marker + experiment.split(marker, 1)[-1], str(REFERENCE), "exec"),
    {"__name__": "__main__"},
)
