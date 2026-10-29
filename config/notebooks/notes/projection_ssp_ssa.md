### What this step does
The SSA projection ensemble, from the end of the SSA historical run. The forcing is anomalies
on a calibrated baseline:
- TF = Zhou TF0 + (GCM TF − GCM 1995–2014 mean)
- SMB = RACMO 1995–2014 mean + (dEBM2 GCM SMB − its 1995–2014 mean)

Chunks end at 2019, 2050, then every 25 years to 2300 (2100 for ssp370). Each chunk resubmits
the next through `launch_projection_ssa.pbs`.

Experiments are chosen with the `AIS3_PROJ_*` options:
- scenario and GCM
- melt (`ref`, `high`, `g<i>`)
- forcing (`both`, `ocean`, `smb`)
- LARMIP region and mode
- shelf collapse
- friction (`coupledN`, `coupledN_p3`)

Names and the full matrix are in `docs/projection_experiments.md`.

### Before you run
- `AIS3_historical_1995_2019_SSA.nc` must exist.
- The first chunk of the first experiment builds `AIS3_proj_ssa_init.nc` and
  `AIS3_proj_ssa_baseline.npz`. Don't start a second experiment until both exist.
- Only the first chunk runs inside this notebook. Later chunks run the driver.
- To stop a chain after its current chunk: `touch models/AIS3_proj_ssa_<exp>.STOP`.
- For a 0.2-yr test, set `AIS3_PROJ_SMOKE=1`. It writes `*_smoke` names only.

### What to check afterwards
- Each chunk writes `AIS3_proj_ssa_<exp>_<year>.nc` and `..._<t0>-<t1>_history.npz`.
- Analyse runs with `check_projection_ssa.py`, as sea level relative to CTRL. CTRL drifts by
  about −0.5 mm/yr (worklog §6.4).
- The step stops with `SystemExit` if a solve did not reach the chunk end time (worklog
  §6.10).

### Known pitfalls
- A walltime-killed inner solve leaves the driver waiting out its whole budget. Collapse runs
  use a watched solve instead.
- Slow nodes have caused several 3–5× slow chunks. `AIS3_PROJ_TIME_FACTOR` and
  `AIS3_PROJ_CHUNK_YEARS` give more room (worklog §6.16).
- MRI-ESM2-0 dates switch type after 2262. Thermal-forcing times are decoded as cftime.
