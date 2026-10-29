### What this step does
HO projections from the end of the HO historical run, as a check on the SSA ensemble. The
forcing is anomalies on a calibrated baseline:
- TF = Zhou TF0 + (GCM TF − GCM 1995–2014 mean)
- SMB = RACMO 1995–2014 mean + (GCM SMB − its 1995–2014 mean)

The forcing is built on the base layer and copied up each column. Runs go in 10-year chunks,
and each chunk resubmits the next through `launch_projection_ho.pbs`. Files are
`AIS3_proj_ho_relaxfix_<exp>_*`. Experiment names and options are the same as the SSA step;
see `docs/projection_experiments.md`.

### Before you run
- `AIS3_historical_1995_2019_relaxfix.nc` must have reached 2019.
- The first chunk builds the shared `_init.nc` and `_baseline.npz`. Don't start a second
  experiment until both exist.
- Only the first chunk runs inside this notebook. Later chunks run the driver.
- To stop a chain after its current chunk: `touch models/AIS3_proj_ho_relaxfix_<exp>.STOP`.

### What to check afterwards
For each chunk, the driver should print:
- "VAF change over chunk …"
- "saved checkpoint …"
- "next chunk submitted …"

The step stops with `SystemExit` if the solve did not reach the chunk end time (worklog §6.10).

### Known pitfalls
- `AIS3_PROJ_COLLAPSE` is refused: shelf collapse is not ported to the 3D mesh.
- `IcefrontMassFlux` is not available on 3D elements.
- A failed inner solve leaves the driver waiting out its whole budget. Check chains for
  stalls.
- Cost: about 11.5 h of the 23 h inner limit per chunk, and about 95k SU per run to 2300
  (worklog §6.19).
