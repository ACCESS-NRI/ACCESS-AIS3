"""Plot and quantify the completed grounded friction 10/100 inversion."""

from pathlib import Path


reference = Path(
    "/g/data/au88/jh7060/ACCESS-AIS3/config/plot_grounded_joint_10_5.py"
)
source = reference.read_text()

replacements = {
    'RUN_NAME = "run_001_10_5_1e-08_1e-17"': 'RUN_NAME = "run_001_10_100_1e-08"',
    'OUTPUT = MODEL_DIR / "grounded_joint_10_5_diagnostics.png"': (
        'OUTPUT = MODEL_DIR / "grounded_friction_10_100_diagnostics.png"'
    ),
    'result_cluster.executionpath = str(ROOT / "execution_grounded_suite_joint")': (
        'result_cluster.executionpath = str(ROOT / "execution_grounded_suite_friction")'
    ),
    'b_inverted = np.asarray(solution.MaterialsRheologyBbar).ravel()': (
        'b_inverted = np.asarray(getattr(solution, "MaterialsRheologyBbar", b_prior)).ravel()'
    ),
    '"Grounded-only JOINT C+B inversion, cf101=10 / cf103=5\\n"': (
        '"Grounded-only FRICTION inversion, cf101=10 / cf103=100\\n"'
    ),
    '"stopped on dxmin at iteration 104"': '"stopped on dxmin at iteration 64"',
}

for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}; found {count}")
    source = source.replace(old, new)

exec(compile(source, str(reference), "exec"), {"__name__": "__main__"})
