### What this step does
Starts the SSA track. It takes the geometry from `AIS3_param.nc` and the inverted ice-shelf B,
then switches friction to Budd p=q=1 using the regularised inversion run
`AIS3_ssa_friction_inv_reg_lcurve/run_001_10_100_0.0001`. It then does a full-continent
forward SSA solve and saves `AIS3_SSA_inverted.nc` plus a velocity plot.

### Before you run
- Run `finalize_p1q1_reg_lcurve.py` once, if that friction run directory doesn't exist.
- **The step floors thin ice at 100 m and does not restore it.** The production file was
  patched afterwards with `restore_floored_thickness_ssa.py` (worklog §4k). After a rerun,
  run that script again, or the 100 m floor ends up in every later SSA run.
- Archive the existing output first (`mv -n`).

### What to check afterwards
- Grounded RMSE against observations. The printed target is about 60.4 m/yr (the
  inversion's own value); the full-continent solve gave 65.67.
- The maximum modelled velocity, and `AIS3_SSA_inverted_velocity.png`.
