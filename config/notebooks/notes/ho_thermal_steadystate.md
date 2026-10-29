### What this step does
Extrudes the inverted SSA state to the HO mesh, solves HO velocity, then a single thermal
steady-state solve. The two solves are separate (velocity, then temperature), not a coupled
`steadystate` iteration. It uses plain temperature (`isenthalpy=0`).

After the solve:
- non-ice temperature is set to 250 K, and the result is clipped at 200 K
- `rheology_B` is recomputed as `cuffey(T)`
- `rheology_B` is then rescaled column by column to the input model's B (the "D2" rescale):
  grounded columns match the deformation-weighted column B, floating columns the plain
  column mean. This keeps HO's vertical profile and restores the inverted ice-shelf B.

### Before you run
- `AIS3_inverted.nc` must exist (from `ssa_inverted_solve`).
- Every later HO file derives from `AIS3_thermal_steadystate.nc`. Archive the existing file
  first (`mv -n`) rather than overwriting it.
- The current `AIS3_thermal_steadystate.nc` was made before the D2 rescale was added
  (2026-10-02). `ho_relaxation` applies D2 downstream instead. A rerun of this step makes that
  later rescale a no-op.

### What to check afterwards
- The printed max velocity after the velocity solve.
- Min, max and mean temperature. Physically: about 200–273 K, warmer at the bed, cold
  surface. A min of millions of K below zero means the non-ice override failed.
- `plot_ais3_thermal_steadystate.py` → `models/ais3_thermal_steadystate_diagnostics.png`.
  The first run had about 21.6% of grounded ice near basal melting.

### Known pitfalls (worklog §2)
- `isenthalpy=1` never converged on this domain. MOLHO was numerically unstable.
- `timestepping.time_step` must be 0, and each solve needs its own `miscellaneous.name`.
- NaN-default scalar fields must be zeroed before extrusion.
- The step needs hugemem. The shared 190 GB `normal` cluster ran out of memory.
