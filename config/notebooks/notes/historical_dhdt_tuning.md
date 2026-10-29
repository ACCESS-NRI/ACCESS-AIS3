### What this step does
A 1995–2019 HO transient, compared with CPOM dH/dt:
- time step 0.1 yr with annual output (`output_frequency=10`)
- annual RACMO SMB
- run in 6-year chunks, **one chunk per submission**

The checkpoint is `AIS3_historical_1995_2019{AIS3_HIST_TAG}.nc`, and its `final_time` marks
progress. Each chunk also writes a base-layer history file `..._history_<t0>-<t1>.npz`. The
final chunk prints the grounded dH/dt RMSE against CPOM.

### Before you run
- `AIS3_ho_relaxed.nc` must exist.
- **Use a new `AIS3_HIST_TAG` for every rerun.** Tags in use:
  - `''`: the old unrelaxed-start run
  - `_chunk1rerun`
  - `_relaxfix`: the current reference, from the D2-relaxed state;
    `launch_historical_relaxfix.pbs` sets this tag
- Resubmit after each chunk until t=2019. Archive the launcher's `.out`/`.err` with a chunk
  suffix each time.
- Each chunk takes about 10.5–11.5 h of driver time and 1,500–1,660 SU.

### What to check afterwards
- "Checkpoint completed through t=…" should advance by 6 years.
- A history npz should be written for every chunk.
- The final RMSE: `_relaxfix` 1.593 m/yr, old HO run 1.768, SSA 1.402.
  - This is a spatial RMSE against a single CPOM field. It cannot see a domain-wide bias.
  - For real validation use the area-weighted, fixed-coverage, per-basin time series:
    `check_ho_historical_timeseries_v2.py`, and worklog §4r–§4t for the method.

### Known pitfalls
- `waitonlock` includes the inner job's queue wait. It is 40 h, and the driver walltime 45 h.
- If an inner solve is killed at walltime, the driver keeps waiting out its budget. Watch
  the job.
