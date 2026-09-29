# ACCESS-AIS3 run guide

How to run the AIS3 pipeline on Gadi (project `au88`): which script runs each stage, what it
reads and writes, what resources it needs, and the known failures. It is distilled from
[`inversion_worklog.md`](inversion_worklog.md) and
[`post_inversion_pipeline_worklog.md`](post_inversion_pipeline_worklog.md). Section numbers
(§) refer to those logs, which have the full evidence.

Status is as of **2026-09-29**. The boundary-condition notes are in the root
[`RUN_GUIDE.md`](../RUN_GUIDE.md).

---

## 1. Environment

| item | value |
|---|---|
| Python | `/scratch/au88/jh7060/mamba/envs/pyissm/bin/python` (launchers `unset PYTHONPATH PYTHONSTARTUP PYTHONHOME` first) |
| ISSM build | `ISSM_DIR=/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_...` (set at the top of every script) |
| cluster module | `access-issm_ad/2026.05.0`. The old `access-issm/2025.11.0` is stale and mismatched. |
| PBS storage | `gdata/au88+gdata/vk83+gdata/av17` |
| run directory | `config/`. Submit every launcher from here. |
| model files | `models/` |
| ISSM execution dirs | `execution/` (HO and upstream SSA), `execution_SSA/` (SSA track) |

## 2. How a step runs

Every launcher (`config/launch_*.pbs`) starts a small **outer driver** job. The driver loads
a model, marshals it, and uses pyissm's `cluster` to submit the **inner ISSM compute job**.
Then it waits on the lock file (`md.settings.waitonlock`, in minutes), loads the results and
saves the `.nc`. So each run is two PBS jobs, and the driver's walltime must cover the inner
job's queue wait plus its run time.

The scripts select what to run in three different ways:

| script | how the step is selected |
|---|---|
| `ais_0.1.py` (HO track + upstream SSA inversion) | `qsub -v AIS3_STEPS=<step>[,<step>] <launcher>` overrides the hard-coded `steps` list. `AIS3_HIST_TAG=<suffix>` renames the historical checkpoint and execution dir. |
| `ais_0.1_SSA.py` (SSA track) | Edit `steps = [...]` near line 91 by hand, then `qsub launch_ais_0.1_SSA.pbs`. There is no env override. |
| standalone scripts (melt calibration etc.) | Hand-edit the `PHASE = 'submit'/'analyze'` constants (or `ROUND`/`STEP`) at the top of the file before each `qsub`. `ho_stressbalance_tests.py` is the exception: it reads `PHASE` from the environment (`qsub -v PHASE=...`). |

**Always use `AIS3_STEPS` for `ais_0.1.py`.** Don't edit the shared `steps` list while a job
is queued. The job reads `steps` when Python starts, which can be days after `qsub`. This
has caused two duplicate `ho_friction_inv` submissions that overwrote a running job's
`.bin` files (§4 race incident, §4f). If you must edit the list, confirm the new job has
printed its own step header with `qcat <new-jobid>` before reverting. Never grep a
`.pbs.out` file for this, because it keeps text from earlier runs.

## 3. Pipeline and file flow

```
                         AIS3_mesh.nc -> AIS3_param.nc
                                   |
          SSA inversion (rheology L-curve, friction L-curve / reg L-curve)
                    |                                          |
        ssa_inverted_solve (Schoof)                 ssa_inverted_solve_budd (p=q=1)
            AIS3_inverted.nc                             AIS3_SSA_inverted.nc
                    |                                          |
   HO TRACK (ais_0.1.py)                        SSA TRACK (ais_0.1_SSA.py)
   ho_thermal_steadystate -> AIS3_thermal_steadystate.nc     ssa_relaxation_budd -> AIS3_SSA_relaxed.nc
   ho_friction_inv (chunked) -> AIS3_ho_friction_inv.nc        (geometry deliberately NOT relaxed)
   melt_gamma_tuning -> AIS3_melt_gamma_tuning.nc            melt_gamma_tuning_ssa + refit scripts
     (validated gamma_0=300 + refit deltaT, thickness restored)  finalize_ssa_melt_calibration.py
   ho_relaxation -> AIS3_ho_relaxed.nc                                 -> AIS3_melt_final_ssa.nc
   historical_dhdt_tuning (4 chunks)                          historical_dhdt_tuning_ssa
        -> AIS3_historical_1995_2019{tag}.nc                     -> AIS3_historical_1995_2019_SSA.nc
   projection_ssp (scaffold)                                  projection_ssp_ssa (scaffold)
```

The SSA track starts its downstream steps from the **inverted** geometry, not the relaxed
one, and that is by design (§4z correction). Only HO propagates its relaxed state.

## 4. Upstream: SSA inversion (`ais_0.1.py`)

These steps are done and you shouldn't need to rerun them. The list is here for reference.

| step | output |
|---|---|
| `process_domain`, `mesh`, `param` | `AIS3_mesh.nc`, `AIS3_param.nc` |
| `ssa_rheology_floating_inv_sensit` / `_lcurve` | `AIS3_ssa_rheology_floating_inv_lcurve/run_004_1_10_1e-17/` (the `rheology_lcurve_run` in use) |
| `ssa_friction_inv_sensit` / `_lcurve` / `_reg_lcurve` | `AIS3_ssa_friction_inv_reg_lcurve/<run>/` |
| `ssa_inverted_solve` | `AIS3_inverted.nc` (Schoof, `C_init=5000`, `Cmax=0.85`, grounded RMSE 70.43) |

- `friction_law` must match between `ais_0.1.py` and `ais_0.1_param.py`. Both are set by
  hand and are currently `'schoof'`.
- The Budd p=q=1 field (RMSE 60.4) that the SSA track uses was salvaged into
  `AIS3_ssa_friction_inv_reg_lcurve/run_001_10_100_0.0001/` by `finalize_p1q1_reg_lcurve.py`.
  It has never been produced by `ssa_friction_inv_reg_lcurve` itself (inversion log §7 item 9).
- Settings that must not regress, because each fixed a real failure (inversion log §2, §6):
  - Budd `C_init=1.8` with bounds `(0.1, 10)`
  - `friction_cf101/cf103 = 10/100`
  - cost function 501, not 502
  - `friction_inv_gttol = 1e-8`
  - vertex-based `ocean_levelset < 0` friction mask
  - thickness floor absorbed into the base, not the surface
- `ssa_inverted_solve` floors thin ice at 100 m and never restores it. Downstream files
  were patched afterwards with `restore_floored_thickness*.py` (§4i, §4k). If you rerun this
  step, redo that patch.

## 5. HO track (`ais_0.1.py`)

HO mesh: 23.8 M vertices (the SSA mesh extruded to 15 layers). Every HO step needs
`hugemem`. The inner jobs set this themselves (`md.cluster.queue='hugemem'`, 96 cores,
2900 GB, except the thermal solve at 48). The outer drivers that save large models also need
`hugemem` (48 cpu / 1450 GB). On `normal`, the save step runs out of memory.

### 5.1 `ho_thermal_steadystate` (done)

```bash
qsub -v AIS3_STEPS=ho_thermal_steadystate launch_ho_relaxation.pbs   # any hugemem 48/1450 launcher
```
Reads `AIS3_inverted.nc` and writes `AIS3_thermal_steadystate.nc`. It uses full HO (not
MOLHO) and `isenthalpy=0`, with separate stress-balance and thermal solves. After the solve
it sets non-ice temperature to 250 K, clips at a 200 K floor, and recomputes `rheology_B`
with `cuffey()`. All of these are needed (§2.1–2.2).

### 5.2 `ho_friction_inv`, chunked (paused after chunk 3)

```bash
qsub -v AIS3_STEPS=ho_friction_inv launch_ho_friction_inv.pbs
```
- Each submission runs `maxsteps=5` m1qn3 iterations and resumes from
  `AIS3_ho_friction_inv.nc` if the file exists. Otherwise it starts from
  `AIS3_thermal_steadystate.nc`.
- On resume, the code copies `results.StressbalanceSolution.FrictionC` into `md.friction.C`.
  `save_model`/`load_model` don't carry that field over, and before this fix every resumed
  chunk silently restarted from scratch (§3.3).
- The run is chunked because a job killed at walltime saves nothing, and hugemem at 96 cores
  is capped at 24 h.
- **Status:** chunk 4 failed 5 times (hangs or SIGBUS at iteration 2's velocity solve, about
  30k SU in total). It was paused on 2026-09-16 (§4h, §4j). The file on disk is chunk 3
  after smoothing. The unsmoothed chunk 3 is in `..._chunk3_backup_presmoothing.nc`.
- Things already tried that did not fix chunk 4:
  - `restol=0.001`: made iterations take over 21 h.
  - `maxiter=300`
  - `dfmin_frac=0.3`
  - cf501 from 1e-6 to 1e-4
  - targeted smoothing of the C field

### 5.3 `melt_gamma_tuning`: apply the validated melt calibration

```bash
qsub -v AIS3_STEPS=melt_gamma_tuning launch_ho_relaxation.pbs      # hugemem 48/1450 driver; no inner solve
```
Reads `AIS3_ho_friction_inv.nc` and `AIS3_param.nc`, and writes `AIS3_melt_gamma_tuning.nc`
(or `AIS3_melt_gamma_tuning{AIS3_MELT_TAG}.nc`). No cluster solve is involved. The step:
- restores real thickness where the 100 m floor is still in the geometry (formerly
  `restore_floored_thickness.py`), then runs `sync_mesh_z`
- configures ismip6 melt with the validated values hard-coded in the step: `gamma_0=300`, the
  16 per-basin refit `deltaT` values, IMBIE2 basins and the Zhou TF climatology (formerly
  `finalize_melt_calibration.py`)
- refuses to overwrite an existing output. Archive the old file with `mv -n` first, or set
  `AIS3_MELT_TAG`.

Before 2026-09-29 this step ran a per-vertex RMSE sweep of `gamma_0`. That metric is gamed
by turning melt off almost everywhere (§4b), so the sweep was removed.

**How the values were derived.** Only rerun this chain if you want to redo the calibration
itself. All of these scripts read `AIS3_ho_friction_inv.nc`. Hand-edit
`PHASE`/`ROUND`/`STEP` before each `qsub`, then copy the new values into the step:

| # | script (launcher `launch_<name>.pbs`) | control | produces |
|---|---|---|---|
| 1 | `melt_ismip7_calibration.py` | `PHASE` submit → analyze | 9-member `gamma_0` ensemble `execution/AIS3_melt_ismip7_g{0..8}` + Mathiot cold/warm runs |
| 2 | `melt_deltaT_sensitivity_test.py` | `PHASE` submit → analyze | +1 °C uniform shift run (initial per-basin slopes) |
| 3 | `melt_deltaT_basin_refit.py` | `ROUND` 0…5 with `STEP` (0 = bootstrap + submit round 1; N = analyze round N + submit N+1) | `models/deltaT_refit_state.json`, `execution/AIS3_deltaT_refit_g{i}_r{N}` |
| 4 | `melt_deltaT_refit_recalibrate.py` | — | J1/J2 scores per candidate. `gamma_0=300` wins, and the J1+J2 Monte Carlo is unanimous. |

Each forward solve takes about 4.5 min and costs about 22 SU. `finalize_melt_calibration.py` and
`restore_floored_thickness.py` produced the current production file. The step above replaces
them.

> **Open (§5c):** the step keeps `md.friction.C` as loaded and does not copy over `FrictionC`,
> which reproduces what the current HO runs used. Every downstream HO file therefore carries the
> **chunk-2** friction field, one chunk behind. Grounded C differs by a median of 0.1%, so the
> effect is probably small. Decide the friction source once the §5c tests are in.

### 5.4 `ho_relaxation` (rerun 2026-09-29)

```bash
qsub -v AIS3_STEPS=ho_relaxation launch_ho_relaxation.pbs      # hugemem 48/1450, 45 h
```
- Reads `AIS3_melt_gamma_tuning.nc` and writes `AIS3_ho_relaxed.nc`.
- Runs 1 year with the RACMO 1979–1994 mean SMB and the calibrated ismip6 melt.
- Calls `sync_mesh_z` before the solve. After the solve it calls `adopt_final_state` and
  `sync_mesh_z` again, so the relaxed geometry, ocean mask and velocities are what gets saved.
- The pre-fix file is kept as `AIS3_ho_relaxed_prerelaxfix.nc`.
- Took about 8h46m and 1262 SU.

### 5.5 `historical_dhdt_tuning`, chunked (1995–2019)

```bash
# one submission = one 6-year chunk; resubmit after each chunk until t=2019
qsub -v AIS3_STEPS=historical_dhdt_tuning,AIS3_HIST_TAG=_relaxfix launch_historical_dhdt_tuning.pbs
```
- Starts from `AIS3_ho_relaxed.nc`. Checkpoints to `AIS3_historical_1995_2019{tag}.nc`, and
  that file's `timestepping.final_time` is the progress marker. There are 4 chunks:
  1995→2001→2007→2013→2019.
- Each chunk also writes base-layer history to
  `AIS3_historical_1995_2019{tag}_history_{t0}-{t1}.npz`. The checkpoint is overwritten every
  chunk, and without these files the per-year history is lost (§4v).
- On resume, the code carries `Thickness` and `MaskOceanLevelset` forward, recomputes
  base/surface hydrostatically so that base ≥ bed, and runs `sync_mesh_z`.
- **Always set a new `AIS3_HIST_TAG` for a rerun.** Without a tag the run resumes from, or
  overwrites, the finished `AIS3_historical_1995_2019.nc`.
- Per chunk: about 13–17 h and 2000–2500 SU, plus hugemem queue waits of up to about 44 h.
  That is why `waitonlock=40 h` and the launcher walltime is 45 h (§4p).
- The final chunk runs the dH/dt comparison against CPOM.
- `launch_historical_chunk1_rerun.pbs` runs chunk 1 only (tag `_chunk1rerun`). It refuses to
  start if that file already exists.
- **Status:** the `_relaxfix` rerun is **on hold** until the HO too-fast-flow question (§7)
  is settled.

### 5.6 `projection_ssp` (scaffold, untested)
Interpolation of time-varying TF and SMB onto the mesh is still a TODO here. The SSA copy
already implements it.

## 6. SSA track (`ais_0.1_SSA.py`)

The mesh is 1.59 M vertices (2D), so every inner job uses the plain `cluster` (48 cores /
190 GB / `normal`). Workflow: edit `steps`, then `qsub launch_ais_0.1_SSA.pbs` (outer driver
on `normal`, 8 cpu, 190 GB, 25 h).

| # | step or script | reads → writes | status |
|---|---|---|---|
| 1 | `ssa_inverted_solve_budd` | `AIS3_param.nc` + p=q=1 friction run → `AIS3_SSA_inverted.nc` (then `restore_floored_thickness_ssa.py`) | done |
| 2 | `ssa_relaxation_budd` | → `AIS3_SSA_relaxed.nc` (20 yr, zero SMB and melt, about 9.5 h) | done; geometry not adopted, by design |
| 3 | `melt_gamma_tuning_ssa` (9-point sweep, J1) | → `AIS3_melt_gamma_tuning_ssa.nc` | done, but this is the coarse pick (5537.7), **not** the one used downstream |
| 4 | `melt_deltaT_sensitivity_test_ssa.py` (`PHASE`) | +1 °C slopes | done |
| 5 | `ssa_melt_deltaT_basin_refit.py` (`ROUND`/`STEP`, as in HO) | `deltaT_refit_state_ssa.json`, `execution_SSA/AIS3_ssa_deltaT_refit_g{i}_r{N}` | done |
| 6 | `ssa_melt_deltaT_refit_recalibrate.py` | J2 minimum at `gamma_0=300` | done |
| 7 | `finalize_ssa_melt_calibration.py` | → `AIS3_melt_final_ssa.nc` | done |
| 8 | `historical_dhdt_tuning_ssa` | `AIS3_melt_final_ssa.nc` → `AIS3_historical_1995_2019_SSA.nc` (one shot, about 6 h, about 530 SU) | done |
| 9 | `projection_ssp_ssa` | TF/SMB interpolation smoke-tested (`test_ssa_projection_interp.py`). SMB anomaly method and elevation feedback still TODO | scaffold |

The historical step must read `AIS3_melt_final_ssa.nc`. `AIS3_melt_gamma_tuning_ssa.nc` holds
the superseded calibration (§4n).

## 7. Current state and next actions (2026-09-29)

1. **HO stress-balance tests (§5c): running.** The submit phase finished and inner jobs
   180097005/180097089 are on hugemem. The tests are: A = HO as is, B = SSA `rheology_B` in
   all 15 layers, C = unsmoothed chunk-3 `FrictionC`. When they finish:
   ```bash
   qsub -v PHASE=analyze launch_ho_stressbalance_tests.pbs
   ```
   If B brings HO surface speed down to about the observed speed, softer HO rheology is the
   cause of HO flowing 10–15% too fast (§5a).
2. Depending on that result, fix HO rheology or friction first (possibly by resuming chunk 4
   or re-inverting). Then graft `FrictionC` properly (§5.3). Then rerun
   `ho_relaxation` → historical with a new tag.
3. Still open: the `projection_ssp` scaffolds, the HO `cf101/cf103/cf501` re-sweep, and
   inversion log §7. That includes the Siple Coast trunk deficit and promoting Budd p=q=1
   through `ssa_friction_inv_reg_lcurve`.

## 8. Gadi resource limits (project `au88`, measured, not documented anywhere)

| queue | ncpus | max walltime | memory |
|---|---|---|---|
| hugemem | 48 (1 node) | 48 h | ≤ ~1450–1500 GB per node |
| hugemem | 96 (2 nodes) | 24 h | 2900 GB |
| hugemem | 144+ | 5 h | — |
| normal | any | — | 192 GB per node ceiling. Memory can't be decoupled from ncpus, and more ranks need *more* memory (768 cores ran out of memory at 2.8 TB). |

- hugemem accepts only multiples of 48 cpus.
- **pyissm does not report `qsub` rejections.** A rejected inner job just waits forever on a
  lock file. If a wait looks too long, run `qsub` on the generated
  `execution/<name>/<name>.queue` directly (with a `timeout`).

## 9. Checklist of known pitfalls

| symptom or situation | cause and fix |
|---|---|
| step "runs" but never submits | `waitonlock=0` + `load_only=True` only *loads* a finished run. Use one `solve(..., load_only=False)` with `waitonlock=<minutes>`. This bug has shown up in 6 steps. |
| resumed inversion restarts from scratch | Copy `results.StressbalanceSolution.<control>` into `md.friction.<field>` after `load_model`. |
| marshalling `TypeError` in `basalforcings.tf` | After `load_model`, `md.basalforcings.tf = [np.asarray(t, float) for t in md.basalforcings.tf]`. |
| `KeyError: '101'` when marshalling a forward run | Leftover inversion config. Set `md.inversion.iscontrol = 0`. |
| `smb.mass_balance not found in binary file` | Set it, even with `issmb=0`. |
| `base < bed` consistency error | Recompute base/surface hydrostatically from thickness and mask. Don't trust the solver's `Base`. |
| HO geometry edited by a script | Run `sync_mesh_z(md)`, or `mesh.z` goes stale (198k columns were off by up to 90 m, §5a). |
| out-of-memory in an outer driver | Strip unused `md.results.TransientSolution` before `deepcopy`. Use hugemem for HO saves. |
| `xr_to_mesh ... must be 2D` | Pass `mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=-1)`. Also `smbgl.squeeze('height')`. |
| 2D field has the wrong shape on the HO mesh | Replicate up the columns with `mesh._project_3d(..., type='element'|'node', layer=0)`. |
| same `miscellaneous.name` used for two solves | They overwrite each other's staging files. Use a distinct name per solve. |
| overwriting outputs | **Never delete.** Archive with `mv -n <f> <f>_<tag>` or write reruns under a new tag. |

**Validation:** the headline "grounded dH/dt RMSE" (HO 1.768, SSA 1.402 m/yr) is a
per-vertex spatial RMSE against only the 2019 CPOM slice. It cannot see a domain-wide bias.
For validation, use the area-weighted, fixed-coverage, per-basin series instead:
`check_ssa_historical_fixedcov_basins.py`, `check_ho_historical_timeseries_v2.py` and
`check_ssa_historical_smb_correlation.py` (§4r–§4t, §5b).

## 10. Monitoring

```bash
qstat -u jh7060                 # both outer and inner jobs
qcat -o <jobid>                 # live stdout of a running job (ISSM writes .outlog only at exit)
tail config/launch_<name>.pbs.out   # outer driver log (holds text from earlier runs too)
ls execution/<run_name>/        # inner job files: .queue, .bin, .outlog/.errlog
```
