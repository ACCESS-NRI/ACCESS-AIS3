"""Submit the corrected joint inversion with a grounded-only velocity misfit.

This is a one-variable isolation test based on the validated ``joint_fz.py``
session harness. Floating ice remains in the stress-balance domain, but cost
functions 101 and 103 are zeroed there. Results use dedicated model and
execution directories, so the all-ice-misfit baseline is not overwritten.
"""

from pathlib import Path


REFERENCE = Path(
    "/scratch/au88/jh7060/tmp/claude-18227/-home-120-jh7060/"
    "167a191a-2bf0-47a6-b6fc-a16ed9084e34/scratchpad/joint_fz.py"
)

source = REFERENCE.read_text()

replacements = {
    "execution_joint_fz": "execution_joint_grounded_misfit",
    "AIS3_joint_fz": "AIS3_joint_grounded_misfit",
    "mask=(mds.inversion.vel_obs>0)": (
        "mask=(np.asarray(mds.inversion.vel_obs).ravel()>0) & gr"
    ),
    "-- submitting {len(grid)} JOINT runs (101/103 velocity-weighting sweep) --": (
        "-- submitting {len(grid)} JOINT run with GROUNDED-ONLY 101/103 misfit --"
    ),
}

for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one occurrence of {old!r} in {REFERENCE}, found {count}"
        )
    source = source.replace(old, new)

exec(compile(source, str(REFERENCE), "exec"), {"__name__": "__main__"})
