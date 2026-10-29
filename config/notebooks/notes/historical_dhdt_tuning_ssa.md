### What this step does
A 1995–2019 SSA transient in one run: time step 0.1 yr, annual RACMO SMB, calibrated melt from
`AIS3_melt_final_ssa.nc`. It saves `AIS3_historical_1995_2019_SSA.nc` before running the dH/dt
diagnostic, so a diagnostic crash can't lose the run.

### Before you run
- `AIS3_melt_final_ssa.nc` must exist. Don't use `AIS3_melt_gamma_tuning_ssa.nc`.
- It takes about 6 h and 530 SU. Archive the existing output first (`mv -n`). Projections
  start from this file.

### What to check afterwards
- The printed grounded dH/dt RMSE against CPOM. The reference run gave 1.402 m/yr, but this is
  a spatial RMSE against one CPOM field, so don't treat it as validation.
- For validation use:
  - `check_ssa_historical_fixedcov_basins.py`
  - `check_ssa_historical_timeseries_areaweighted.py`
  - `check_ssa_historical_smb_correlation.py`
- Reference results (worklog §4r–§4t):
  - The area mean is close to CPOM after a start-up transient of about 6 years.
  - There is no interannual skill.
  - The Amundsen sector over-thins, and the run misses its acceleration.
