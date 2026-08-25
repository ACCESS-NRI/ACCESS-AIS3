"""Render and quantify all four completed regularized friction runs."""

import gc
from pathlib import Path


REFERENCE = Path(
    "/g/data/au88/jh7060/ACCESS-AIS3/config/plot_grounded_joint_10_5.py"
)
template = REFERENCE.read_text()

runs = [
    ("run_001_10_0.5_0.001", "10 / 0.5 / 1e-3", 120),
    ("run_002_10_0.5_0.01", "10 / 0.5 / 1e-2", 94),
    ("run_003_10_1_0.001", "10 / 1 / 1e-3", 156),
    ("run_004_10_1_0.01", "10 / 1 / 1e-2", 67),
]

for run_name, coefficient_label, iteration in runs:
    print(f"\n===== {run_name} =====", flush=True)
    output_name = f"grounded_friction_regularized_{run_name}_diagnostics.png"
    source = template
    replacements = {
        'RUN_NAME = "run_001_10_5_1e-08_1e-17"': f'RUN_NAME = "{run_name}"',
        'OUTPUT = MODEL_DIR / "grounded_joint_10_5_diagnostics.png"': (
            f'OUTPUT = MODEL_DIR / "{output_name}"'
        ),
        'result_cluster.executionpath = str(ROOT / "execution_grounded_suite_joint")': (
            'result_cluster.executionpath = str('
            'ROOT / "execution_grounded_friction_regularized")'
        ),
        'b_inverted = np.asarray(solution.MaterialsRheologyBbar).ravel()': (
            'b_inverted = np.asarray(getattr('
            'solution, "MaterialsRheologyBbar", b_prior)).ravel()'
        ),
        '"Grounded-only JOINT C+B inversion, cf101=10 / cf103=5\\n"': (
            f'"Grounded-only FRICTION inversion, cf101/cf103/cf501='
            f'{coefficient_label}\\n"'
        ),
        '"stopped on dxmin at iteration 104"': (
            f'"stopped on dxmin at iteration {iteration}"'
        ),
    }
    for old, new in replacements.items():
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"Expected one occurrence of {old!r} for {run_name}; found {count}"
            )
        source = source.replace(old, new)
    exec(compile(source, str(REFERENCE), "exec"), {"__name__": "__main__"})
    gc.collect()
