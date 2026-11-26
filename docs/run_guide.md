# ACCESS-AIS3 run guide

How to run the AIS3 pipeline on Gadi (project `au88`): which script runs each stage, what it
reads and writes, what resources it needs, and the known failures. It is distilled from
[`inversion_worklog.md`](inversion_worklog.md) and
[`post_inversion_pipeline_worklog.md`](post_inversion_pipeline_worklog.md). Section numbers
(§) refer to those logs, which have the full evidence.

Status is as of **2026-09-29**. The boundary-condition notes are in the root
[`BOUNDARY_CONDITIONS.md`](../BOUNDARY_CONDITIONS.md).

---

## 1. Environment

Paths below are relative to the repository root, `<repo>`.

**Site settings live in one file, [`config/site.env`](../config/site.env).** Edit it when you
run AIS3 under another NCI account, project or Python environment:

| setting | what it is | default |
|---|---|---|
| `AIS3_PROJECT` | NCI project charged for the jobs | `au88` |
| `AIS3_STORAGE` | storage the jobs mount | `gdata/au88+gdata/vk83+gdata/av17` |
| `AIS3_PYTHON` | Python with pyissm and ccdtools (and jupyter + a `pyissm` kernel for notebooks) | the maintainer's env on /scratch/au88 |
| `AIS3_ISSM_DIR` | ISSM build (ACCESS-NRI, vk83); must match the module the inner jobs load | `issm-git.2026.05.18_...` |
| `AIS3_ISMIP_DIR` | ISMIP forcing tree | `/g/data/au88/ismip6/2300/forcings` |

Each line is `KEY=${KEY:-default}`, so an environment variable of the same name overrides it
(`qsub -v AIS3_PYTHON=...`, or `export` before a notebook session). Found automatically, not
set there: the repository root (the folder above `config/`; override with `AIS3_ROOT`) and
the NCI username (`$USER`; override with `AIS3_USER`). Python reads the settings through
[`config/ais3_site.py`](../config/ais3_site.py) (`import ais3_site as site`; run it to print
them); every launcher sources `site.env` after changing into `config/`.

| item | value |
|---|---|
| cluster module | `access-issm_ad/2026.05.0`. The old `access-issm/2025.11.0` is stale and mismatched. |
| run directory | `config/`. Submit launchers from here (the repo root also works); `#PBS -o/-e` log paths are relative to it. |
| model files | `models/` |
| ISSM execution dirs | `execution/` (HO and upstream SSA), `execution_SSA/` (SSA track) |

**Another project or storage.** `#PBS -P` and `#PBS -l storage` lines cannot read a file, so
submit with [`config/submit`](../config/submit) instead of `qsub`: it takes the same
arguments and adds `-P $AIS3_PROJECT -l storage=$AIS3_STORAGE`, which override those lines.
Chunked runs that resubmit themselves add the same flags.

**Not converted:** the early grounded-friction experiment scripts (`run_grounded_*`,
`run_joint_grounded_misfit_test.py`, `plot_grounded_*`, `plot_friction_run.py`) and most
untracked one-off check scripts still hard-code paths. `ais_0.1_param.py` reads one input
(`Antarctica_SubglacialHydrology.nc`) from `/g/data/au88/lb9857/`, which needs au88 access.

## 2. How a step runs

Every launcher (`config/launch_*.pbs`) starts a small **outer driver** job. The driver loads
a model, marshals it, and uses pyissm's `cluster` to submit the **inner ISSM compute job**.
Then it waits on the lock file (`md.settings.waitonlock`, in minutes), loads the results and
saves the `.nc`. So each run is two PBS jobs, and the driver's walltime must cover the inner
job's queue wait plus its run time.

The scripts select what to run in three different ways:

| script | how the step is selected |
|---|---|
| `ais_0.1.py` (HO track + upstream SSA inversion) | `qsub -v AIS3_STEPS=<step>[,<step>] <launcher>` overrides the hard-coded `steps` list. `AIS3_HIST_TAG=<suffix>` renames the historical checkpoint and execution dir. `AIS3_HIST_FRICTION=budd_nopins` (fresh start only, needs a new tag) swaps in the SSA Budd p=q=1 friction and releases all velocity pins. With it, `AIS3_HIST_FRICTION_C=<npz>:<key>` replaces that friction coefficient (e.g. the `ho_friction_fixedpoint.py` result) and `AIS3_HIST_MELT=hm<gamma_0>` takes gamma_0 and deltaT from `models/melt_calibration_hm_ho.json`. |
| `ais_0.1_SSA.py` (SSA track) | Same `AIS3_STEPS` override as `ais_0.1.py`: `qsub -v AIS3_STEPS=<step> launch_ais_0.1_SSA.pbs`. Otherwise edit the `steps` list by hand. Variants, each tagging every file it writes: `AIS3_SSA_CALIB=_hm<gamma_0>` (history-matched melt, whole chain), `AIS3_SSA_FRICTION=p1_flotN/p3_fixedN/p3_flotN/p5_flotN`, `AIS3_SSA_OCEAN=main/cold/warm/vary` (OCX pathways) and `AIS3_SSA_FRONT=greene` (observed front) for the historical run. `AIS3_HIST_SUBMIT_ONLY=1` then `AIS3_HIST_LOAD_ONLY=1` splits a historical run so the big-memory driver does not sit idle (worklog 6.32). |
| standalone scripts (melt calibration etc.) | Hand-edit the `PHASE = 'submit'/'analyze'` constants (or `ROUND`/`STEP`) at the top of the file before each `qsub`. `ho_stressbalance_tests.py` is the exception: it reads `PHASE` from the environment (`qsub -v PHASE=...`). |

**Always use `AIS3_STEPS` for `ais_0.1.py`.** Don't edit the shared `steps` list while a job
is queued. The job reads `steps` when Python starts, which can be days after `qsub`. This
has caused two duplicate `ho_friction_inv` submissions that overwrote a running job's
`.bin` files (§4 race incident, §4f). If you must edit the list, confirm the new job has
printed its own step header with `qcat <new-jobid>` before reverting. Never grep a
`.pbs.out` file for this, because it keeps text from earlier runs.

### Step notebooks

`config/notebooks/<driver>/<NN>_<step>.ipynb` has one runnable notebook per step of
`ais_0.1.py` and `ais_0.1_SSA.py`. Each one holds the driver's setup code followed by that
step's code, split into cells. Its header lists what the step reads and writes, its `AIS3_*`
options with their defaults, and the `qsub` command with the step's resources:

```bash
cd config/notebooks
qsub -q hugemem -l ncpus=48,mem=1450GB,walltime=45:00:00 -v NB=ais_0.1/16_ho_relaxation.ipynb launch_notebook.pbs
```

The executed copy goes to `executed/<name>.<jobid>.ipynb` next to the notebook.

The notebooks are generated by `config/make_step_notebooks.py`, and the drivers remain the
source of truth. After changing a driver, rerun it from `config/` with
`<pyissm-env>/bin/python make_step_notebooks.py`. Don't edit the notebooks by hand. The
production steps also get hand-written notes (what the step does, what it needs, what to check,
pitfalls) from `config/notebooks/notes/<step>.md`; edit those files, then regenerate.

To test the notebooks without running the pipeline, use `config/notebooks/test_notebooks.py`.
Submit it with `test_notebooks.pbs` (the commands are in its header). There are three levels:
- `static`: every notebook is valid and its code matches its driver. Takes seconds.
- `setup`: runs each notebook's setup cells, with no step.
- `dry`: runs each step up to its first cluster solve. There, the solve is replaced by the
  model consistency check plus marshalling to `/dev/null`. The run also stops at the first
  `save_model`, file write or `qsub`, so nothing is written to `models/` or `execution/`. The HO
  notebooks need hugemem at this level, because they load the full HO models.

Results go to `config/notebooks/test_runs/` (ignored by git). The older
hand-ported notebooks in `config/` (`ais_0.1_*.ipynb`, August 2026) are out of date.

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
   melt_gamma_tuning -> AIS3_melt_gamma_tuning.nc            melt_gamma_tuning_ssa -> AIS3_melt_final_ssa.nc
     (validated gamma_0=300 + refit deltaT, thickness restored)  (validated gamma_0=300 + refit deltaT)
   ho_relaxation -> AIS3_ho_relaxed.nc
   historical_dhdt_tuning (4 chunks)                          historical_dhdt_tuning_ssa
        -> AIS3_historical_1995_2019{tag}.nc                     -> AIS3_historical_1995_2019_SSA.nc
   projection_ssp (scaffold)                                  projection_ssp_ssa (chained 50-yr chunks to 2300)
```

The SSA track starts its downstream steps from the **inverted** geometry, not the relaxed
one, and that is by design (§4z correction). Only HO propagates its relaxed state.

## 4. Upstream: SSA inversion (`ais_0.1.py`)

Model states are not distributed: `models/` is not in git, so a new checkout starts empty and
you produce every state yourself, running the steps below in order and then one of the two
tracks (§5 HO, §6 SSA). The maintainer's run of these steps is complete; the table lists
what each step writes and the run settings in use.

**Ice-shelf rheology B.** Every step that needs the shelf B (`load_shelf_rheology_B()` in both
drivers, `ssa_inverted_solve`, the forward checks) reads
`models/AIS3_ssa_rheology_floating_inv_lcurve/run_004_1_10_1e-17/`, the step's own output at
cf101=1, cf103=10, cf502=1e-17. The maintainer's copy was saved there from the one-off
`execution_newB_rheology` run by `finalize_newB_rheology.py`; today's step builds identical
inputs, so a run from scratch produces the same field. The June 2026 `run_004`, made before the
thickness-floor fixes, is kept as `run_004_1_10_1e-17.jun30_before_geometry_fixes`.
`AIS3_inverted.nc` and the existing HO track were built with that June field, while the friction
inversions and the SSA track used the newer one (worklog 2026-11-16).

**Two inputs the steps do not yet produce on their own.** In the maintainer's run they came
from one-off runs, so a run from scratch stops at, or differs from, these points until they are
folded into the pipeline:

- **Schoof friction field** (`friction_lcurve_run = 'schoof_m1qn3_tightrestol_cmax2.0'`, used by
  `ssa_inverted_solve` while `friction_law = 'schoof'`). It came from a one-off m1qn3 run whose
  script was lost. The step `ssa_friction_inv_schoof` (launcher
  `launch_ssa_friction_inv_schoof.pbs`) now rebuilds it from the Budd p=q=1 field below: submit
  it once to start the inversion (48 cores, about 36 h), and again when that has finished to
  save `AIS3_ssa_friction_inv_reg_lcurve/schoof_from_budd_cmax2.0/`. Then set
  `friction_lcurve_run` to that name. The result is not bit-identical to the maintainer's field
  (the original start field and grounded surface could not be fully recovered; worklog
  2026-11-17). Through `ssa_inverted_solve` it gives a grounded RMSE of 75.7 m/yr, against 72.0
  for the maintainer's field on the same code. To compare fields without overwriting
  `AIS3_inverted.nc`, pass `AIS3_INVERTED_FRICTION_RUN=<run>` and `AIS3_INVERTED_TAG=_<tag>`
  (output `models/AIS3_inverted_<tag>.nc`).
- **Budd p=q=1 field** for the SSA track: see the bullet below the table.

| step | output |
|---|---|
| `process_domain`, `mesh`, `param` | `AIS3_mesh.nc`, `AIS3_param.nc` |
| `ssa_rheology_floating_inv_sensit` / `_lcurve` | `AIS3_ssa_rheology_floating_inv_lcurve/run_004_1_10_1e-17/` (the `rheology_lcurve_run` in use) |
| `ssa_friction_inv_sensit` / `_lcurve` / `_reg_lcurve` | `AIS3_ssa_friction_inv_reg_lcurve/<run>/` |
| `ssa_friction_inv_schoof` | `AIS3_ssa_friction_inv_reg_lcurve/schoof_from_budd_cmax2.0/` (HO track only) |
| `ssa_inverted_solve` | `AIS3_inverted.nc` (Schoof, Cmax 2.0, grounded RMSE 70.43) |

- `friction_law` must match between `ais_0.1.py` and `ais_0.1_param.py`. Both are set by
  hand and are currently `'schoof'`.
- The Budd p=q=1 field (RMSE 60.4) that the SSA track uses was salvaged into
  `AIS3_ssa_friction_inv_reg_lcurve/run_001_10_100_0.0001/` by `finalize_p1q1_reg_lcurve.py`,
  from a one-off sweep (`execution_p1q1_reg_sweep2/`, script not in the repository). It has
  never been produced by `ssa_friction_inv_reg_lcurve` itself (inversion log §7 item 9). From
  scratch, run that step with the Budd settings (`friction_law = 'budd'` in both files) for the
  `cf501 = 1e-4` point and check the grounded RMSE against 60.4.
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
- **Pin-free variant:** `launch_historical_budd_nopins.pbs` (tag `_budd_nopins`,
  `AIS3_HIST_FRICTION=budd_nopins`) replaces the HO Schoof field and its ~176k pinned columns with
  the SSA track's Budd p=q=1 field and N, with no pins (`use_ssa_budd_friction_no_pins`). The
  Schoof field cannot run unpinned, and the pins build km-scale thickness spikes during the
  historical run (worklog §6.21, §6.23, §6.27). Submit the four chunks as an `afterok` chain with
  per-chunk `-o`/`-e` logs (command in the launcher header). Started 2026-11-18.
- **Refined friction and gamma_0 = 4000:** `launch_historical_budd_nopins_hm4000.pbs` (tag
  `_budd_nopins_hm4000`) adds `AIS3_HIST_FRICTION_C=models/ho_friction_fixedpoint_cont.npz:C7` and
  `AIS3_HIST_MELT=hm4000` (worklog §6.38, §6.42). Started 2026-11-24.
- `launch_historical_chunk1_rerun.pbs` runs chunk 1 only (tag `_chunk1rerun`). It refuses to
  start if that file already exists.
- **Status:** the `_relaxfix` rerun is **on hold** until the HO too-fast-flow question (§7)
  is settled.

### 5.6 `projection_ssp` (scaffold, untested)
Interpolation of time-varying TF and SMB onto the mesh is still a TODO here. The SSA copy
already implements it.

## 6. SSA track (`ais_0.1_SSA.py`)

The mesh is 1.59 M vertices (2D), so every inner job uses the plain `cluster` (48 cores /
190 GB / `normal`). Workflow: `qsub -v AIS3_STEPS=<step> launch_ais_0.1_SSA.pbs` (outer driver
on `normal`, 8 cpu, 190 GB, 25 h).

| # | step or script | reads → writes | status |
|---|---|---|---|
| 1 | `ssa_inverted_solve_budd` | `AIS3_param.nc` + p=q=1 friction run → `AIS3_SSA_inverted.nc` (then `restore_floored_thickness_ssa.py`) | done |
| 2 | `ssa_relaxation_budd` | → `AIS3_SSA_relaxed.nc` (20 yr, zero SMB and melt, about 9.5 h) | done; geometry not adopted, by design |
| 3 | `melt_gamma_tuning_ssa` | `AIS3_SSA_relaxed.nc` → `AIS3_melt_final_ssa.nc` (`AIS3_MELT_TAG` suffix). Applies the hard-coded validated `gamma_0=300` + refit `deltaT`; no solve; refuses to overwrite | done (current file built by `finalize_ssa_melt_calibration.py`) |
| 4 | `historical_dhdt_tuning_ssa` | `AIS3_melt_final_ssa.nc` → `AIS3_historical_1995_2019_SSA.nc` (one shot, about 6 h, about 530 SU) | done |
| 5 | `projection_ssp_ssa` | historical end state → `AIS3_proj_ssa_<exp>`; chained 50-yr chunks to 2300 via `launch_projection_ssa.pbs`. See [`projection_experiments.md`](projection_experiments.md) | implemented 2026-09-29 |

Before 2026-09-29, `melt_gamma_tuning_ssa` ran a J1-only `gamma_0` sweep with the published
`deltaT`. It wrote the superseded pick (5537.7) to `AIS3_melt_gamma_tuning_ssa.nc`, which is still
on disk and must not be used (§4n). The validated values came from this chain. Only rerun it if
you want to redo the calibration itself, then copy the new values into the step:

| # | script | control | produces |
|---|---|---|---|
| 1 | `melt_gamma_sweep_ssa.py` | `qsub -v PHASE=submit` → `PHASE=analyze` | 9-member `gamma_0` ensemble at the published `deltaT`, `execution_SSA/AIS3_melt_gamma_tuning_ssa_g{0..8}`. Refuses if those dirs exist; archive them first with `mv -n` |
| 2 | `melt_deltaT_sensitivity_test_ssa.py` | `PHASE` (hand-edited) | +1 °C run, compared against `g6` for the slopes |
| 3 | `ssa_melt_deltaT_basin_refit.py` | `ROUND`/`STEP`, as in HO | round 0 bootstraps from `g0`–`g8`; `deltaT_refit_state_ssa.json`, `execution_SSA/AIS3_ssa_deltaT_refit_g{i}_r{N}` |
| 4 | `ssa_melt_deltaT_refit_recalibrate.py` | — | J2 minimum at `gamma_0=300` |

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
3. Still open: the HO `projection_ssp` scaffold, the HO `cf101/cf103/cf501` re-sweep, and
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
qstat -u $USER                  # both outer and inner jobs
qcat -o <jobid>                 # live stdout of a running job (ISSM writes .outlog only at exit)
tail config/launch_<name>.pbs.out   # outer driver log (holds text from earlier runs too)
ls execution/<run_name>/        # inner job files: .queue, .bin, .outlog/.errlog
```
