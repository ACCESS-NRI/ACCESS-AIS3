### What this step does
A 1-year HO transient to damp the shock of switching from a diagnostic to a prognostic
state. It is not the historical spin-up. Settings:
- time step 0.02 yr
- RACMO 1979–1994 mean SMB and the calibrated ISMIP6 melt
- sub-element grounding-line migration

Before the solve:
- `mesh.z` is synced to the geometry
- `rheology_B` gets the D2 rescale against `AIS3_inverted.nc`. This fixes HO flowing 10–15%
  too fast (worklog §5c/§5d) and is a no-op once the thermal step has been rerun.

After the solve, `adopt_final_state` and `sync_mesh_z` make the relaxed end state the saved
geometry and initial velocity.

### Before you run
- `AIS3_melt_gamma_tuning.nc` must exist.
- The step overwrites `AIS3_ho_relaxed.nc`, so archive it first (`mv -n`). Earlier versions
  are kept as `_oldB` and `_prerelaxfix`.
- It takes about 8.5 h and 1,220 SU.

### What to check afterwards
- Before the solve, "mesh.z vs geometry before sync" reports 198,014 columns off by up to
  90 m on the current input. That is expected.
- The relaxation diagnostics: the D2 run gave mean |dH| 6.07 m and max 1155 m.
  - The maximum is a known local artifact: a deep inlet on the western Ross Ice Shelf margin
    (worklog §5d).
  - Grounded/floating state changed at about 1.1% of vertices.
- An "adopted final transient state" line must appear before the save.

### Known pitfalls
- Saving the model needs a hugemem outer driver.
- `basalforcings.tf` comes back from `load_model` as lists. The step converts it back to
  arrays.
