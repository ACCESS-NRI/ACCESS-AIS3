### What this step does
Builds `AIS3_melt_final_ssa.nc`, the input to the SSA historical and projection steps, from
the validated SSA melt calibration. There is no solve. It loads `AIS3_SSA_relaxed.nc` as the
model shell and sets ISMIP6 quadratic-local melt:
- IMBIE2 basins and Zhou thermal forcing
- `gamma_0=300` and the SSA track's 16 refit `deltaT` values, hard-coded in the step

### Before you run
- The step refuses to overwrite an existing output. Archive it (`mv -n`) or set
  `AIS3_MELT_TAG`.
- To redo the calibration rather than apply it, run the standalone chain (run guide §6):
  1. `melt_gamma_sweep_ssa.py`
  2. `melt_deltaT_sensitivity_test_ssa.py`
  3. `ssa_melt_deltaT_basin_refit.py`
  4. `ssa_melt_deltaT_refit_recalibrate.py`

  Then copy the new values into this step.

### What to check afterwards
- 30 thermal-forcing depth layers, and `gamma_0=300` in the saved file.

### Known pitfalls
- `AIS3_melt_gamma_tuning_ssa.nc` (`gamma_0=5537.7`) is the superseded coarse pick. Nothing
  should read it.
- At `gamma_0=300`, melt responds to ocean warming only about 4% as strongly as the ISMIP7
  ocean models (worklog §6.1). Projections use `gamma_0=15000` (`AIS3_PROJ_MELT=high`) as the
  high-melt member.
