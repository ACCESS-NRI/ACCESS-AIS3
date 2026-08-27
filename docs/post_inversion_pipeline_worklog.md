# ACCESS-AIS3 post-inversion pipeline work log

Design, debugging, and validation of the pipeline stages that run after the SSA
friction/rheology inversion documented in [`inversion_worklog.md`](inversion_worklog.md):
higher-order (HO) thermal spin-up → HO friction re-inversion → melt calibration →
relaxation → historical tuning → future projections. Mirrors Felicity's validated MATLAB
pipeline (`/g/data/au88/jr5971/access-dev2.git/antarctica-issm/matlab_fm/runme.m`)
structurally, adapted to pyISSM.

---

## 1. Stage status

| stage | `steps` name | status | output |
|---|---|---|---|
| 1. HO thermal spin-up | `ho_thermal_steadystate` | **validated** | `AIS3_thermal_steadystate.nc` |
| 2. HO friction re-inversion | `ho_friction_inv` | **in progress** (chunked, see §3.4) | `AIS3_ho_friction_inv.nc` |
| 3. Ocean melt (gamma) calibration | `melt_gamma_tuning` | scaffold, untested | `AIS3_melt_gamma_tuning.nc` |
| 4. Post-calibration relaxation | `ho_relaxation` | scaffold, untested | `AIS3_ho_relaxed.nc` |
| 5. Historical run (1995–2019) vs dH/dt | `historical_dhdt_tuning` | scaffold, untested | `AIS3_historical_1995_2019.nc` |
| 6. Future projections (SSP-forced) | `projection_ssp` | scaffold, untested | `AIS3_projection_{gcm}_{scenario}.nc` |

Scaffolded stages have correct model loading / solver setup / save pattern and real data
paths, but open science decisions are marked `# TODO:` in the code rather than silently
resolved — do not treat their intermediate outputs as validated.

---

## 2. Stage 1 — HO thermal steady-state

### 2.1 Design decisions

- **Full HO, not MOLHO.** MOLHO's reduced vertical shape function was tested first and
  produced numerically unstable thermal advection; switching to full HO (`set_flow_equation
  (md, HO='all')`) fixed it. Costs more memory/compute (independent DOF per layer vs. a
  basal+shear split), which is why this stage needed the `hugemem` queue.
- **Decoupled velocity → thermal, not coupled `steadystate` Picard iteration.** Matches
  Felicity's actual validated pipeline. Two separate solves (`'stressbalance'` then
  `'thermal'`) instead of one `'steadystate'` call.
- **`isenthalpy=0` (plain temperature), not `isenthalpy=1` (phase-change-aware enthalpy).**
  `isenthalpy=1` was tested exhaustively — mesh resolution, HO vs MOLHO, SUPG vs artificial
  diffusivity stabilization, `penalty_lock`, and loosened `solver_residue_threshold` — and
  never converged on this domain; loosening the threshold made convergence *worse*, not
  more permissive, revealing it affects the solver's internal step size, not just a
  pass/fail bar. `isenthalpy=0` converges cleanly. Known limitation: it doesn't track
  basal melt/refreeze or cap exactly at pressure melting on its own — handled instead by
  post-solve clipping (§2.2).

### 2.2 Bugs found and fixed

1. **`waitonlock=0` + `load_only=True` never submits.** `pyissm.model.execute.solve()`'s
   `load_only=True` branch only loads an already-finished run's results; with
   `waitonlock=0` nothing is ever submitted. This exact pattern recurred **four times**
   across the pipeline (`ssa_inverted_solve`, `ssa_relaxation`, `ho_thermal_steadystate`,
   `ho_friction_inv`) — check for it first in any step that seems to hang doing nothing.
   Fix: synchronous submit-and-wait (`load_only=False`, `waitonlock=<minutes>`).
2. **`'SteadystateSolution'` isn't a valid `solve()` string.** The string-to-analysis
   mapping is case-insensitive on short/long names (`'steadystate'`) but not on the
   PascalCase result-class name.
3. **`initialization.waterfraction`/`watercolumn` bare-scalar NaN defaults don't survive
   extrusion.** `_project_3d`'s special-case for size-1 inputs leaves the wrong shape
   post-extrude. Fix: zero them before extrusion.
4. **`basalforcings.{groundedice,floatingice}_melting_rate` hit the same NaN-default
   marshalling bug**, caught only at job-submission time, not by Python-side consistency
   checks. Same fix.
5. **`timestepping.time_step` must be `0` for steadystate-family solves** — `AIS3_inverted.nc`
   carried a stale nonzero value from upstream.
6. **Shared `md.miscellaneous.name` between the velocity and thermal solve steps** corrupted
   the second solve's staging files (overwrote the first's), producing an immediate false
   "Recovery solver failed". Fix: distinct names per solve step.
7. **Uniform-cold initial temperature stalled convergence.** Fixed by priming a
   depth-varying initial guess (linear surface → pressure-melting-minus-1K at bed) before
   solving.
8. **Non-ice `spctemperature` Dirichlet pin is load-bearing, not optional.** Removing it
   produced a min temperature of −8.4 million K. ISSM's `spc*` constraints are
   penalty-based, not exact elimination, and this pin dominates the penalty at small scale.
9. **The same pin doesn't reliably hold at full continental scale.** Of 114,802 "bad" 2D
   columns found in a post-solve diagnostic (7.2% of the domain), 99.87% were non-ice —
   the penalty method doesn't dominate strongly enough across ~114K scattered small
   non-ice inclusions continent-wide (it worked fine on the much smaller/homogeneous
   PIG/Thwaites test population). Fixed with a **post-solve Python override**
   (force non-ice temperature to 250K) rather than relying on the solver constraint.
10. **220K clip floor over-warmed genuinely cold locations** (Dome A ≈214.7K, Vostok
    ≈217.9K annual mean, both real, both below 220K). Diagnostic showed nodes < 200K ≈
    nodes < 100K (only ~800 apart) — the bad population is overwhelmingly catastrophic
    (millions of K), not gently cold, so lowering the floor to 200K was safe. Confirmed
    4,255 genuinely cold ice nodes preserved instead of clipped after the fix.
11. **Full continental HO run OOM'd (SIGKILL) at 190GB/`normal` queue.** HO's ~23.8M-node
    mesh needs far more memory than the 2D SSA-tuned cluster config. Fixed with a
    stage-local `hugemem` queue override (not the shared `cluster` object).

### 2.3 Result

Converged, saved to `AIS3_thermal_steadystate.nc`. Post-solve: non-ice override applied,
200K floor clip, `rheology_B` recomputed from the new depth-resolved temperature via
`cuffey()`. Visually verified physically sensible: correct base>surface temperature
gradient direction, correct fast-flow channel patterns, ~21.6% of grounded ice near basal
melting (glaciologically plausible for West Antarctica). Plot:
`models/ais3_thermal_steadystate_diagnostics.png`.

---

## 3. Stage 2 — HO friction re-inversion

### 3.1 Bug: cost function evaluated at the wrong vertices

`cost_functions=[101, 103, 501]` — 101/103 (`SurfaceAbsVelMisfit`/`SurfaceLogVelMisfit`,
`pyissm/model/inversions.py:13-14`) are evaluated against satellite-observed **surface**
speed, at **surface** vertices. The friction control itself is a **basal** parameter, so the
first version of this step reused `on_base` for the cost-function coefficient mask too —
correct for the control's own bounds, wrong for where the misfit is evaluated. m1qn3 was
consequently optimizing pure regularization smoothness and never actually fitting observed
velocity, which showed up as **exactly `0`/`0` printed contributions for cf101/cf103** in
the m1qn3 cost table (caught because the user asked why those were zero, not by an
automated check). Fixed: `mask` (for cf101/cf103) uses `on_surface`; `reg_mask` (for cf501,
`DragCoefficientAbsGradient`, a genuinely basal quantity) stays on `on_base`. Verified on a
PIG/Thwaites subdomain test first (`RMSE=417.77 m/yr`, genuinely converged at iteration 34,
not budget-truncated) before trusting it at continental scale.

### 3.2 NCI resource constraints discovered (empirically, not documented anywhere)

- **`hugemem` per-node memory cap ≈1450–1500GB.** A flat `-l mem=X` on a multi-node request
  divides across nodes, so memory must scale with node count to preserve per-node headroom
  (confirmed via direct `qsub` bisection: 1470/1490/1500GB accepted per node, 2900GB on a
  single node rejected).
- **Project-level walltime cap shrinks sharply with `hugemem` core count** (project `au88`,
  confirmed via direct `qsub` testing): 48–96 cores → 24h allowed; 144+ cores → only 5h.
  This is a hard institutional policy. A 192-core/48h request silently hung the launcher
  ("waiting for lock file" forever) because `pyissm.model.execute.solve()` **does not
  surface `qsub` submission failures** — the rejected request never created a job, and the
  launcher just polled for a lock file from a job that never existed. **Whenever a
  launcher's wait loop seems unusually long, test the generated `.queue` file directly with
  `qsub` (with a `timeout`) rather than waiting longer.**
- Settled on **96 cores / 2 hugemem nodes / 2900GB / 24h** as the working full-continental
  config.

### 3.3 Bug: 24h walltime kill loses all progress, no checkpoint

The first full-continental attempt at 96 cores made real progress — m1qn3 cost dropped
`3.0065e+07 → 3.211e+06` (89%) over 12 iterations — but was killed by the 24h cap mid
iteration 13. Because `pyissm.model.execute.solve()` only writes results after the
`mpiexec` process returns normally, and ISSM's m1qn3 loop has no built-in mid-run
checkpoint, **all 12 iterations of progress were lost** — no `AIS3_ho_friction_inv.nc` was
ever written.

**Fix — chunked warm-restart** (`config/ais_0.1.py`, `ho_friction_inv` step):
- `md.inversion.maxsteps = md.inversion.maxiter = 8` per job (~2h/iteration observed, so
  ~16h per chunk, leaving margin within the 24h cap).
- The step now checks for an existing `AIS3_ho_friction_inv.nc` and resumes from it
  (warm-starting the friction field from the last completed chunk) if present, else falls
  back to `AIS3_thermal_steadystate.nc` for a fresh start. Repeated 24h submissions
  accumulate progress instead of re-doing it.
- Cost: m1qn3's internal quasi-Newton curvature estimate is **not** preserved across
  restarts (no ISSM mechanism to serialize it) — each chunk starts a fresh approximation.
  Real but unavoidable overhead of this workaround; the alternative (no checkpoint at all)
  is strictly worse.
- The maxiter-100 nonlinear-iteration-exceeded warning inside the stress balance solve,
  seen at inversion iteration 2 of the first attempt, did **not** recur in iterations 3–12
  — treated as a transient early-inversion effect (large friction perturbation away from
  optimum), not a persistent problem, but worth re-checking if it reappears.

### 3.4 Progress so far

| chunk | job | outcome | cost f(x) | notes |
|---|---|---|---|---|
| 1 (unchunked, 500-step budget) | 177331341 | killed at 24h, iter 13 | 3.0065e7 → 3.211e6 | **no checkpoint saved — lost** |
| 1 (chunked, maxsteps=8) | 177469174 | completed, exit 0, 21h6m | 3.0065e7 → 4.9165e6 | RMSE=94.38 m/yr, saved |
| 2 (chunked, maxsteps=8, resumed) | 177574965 | running | — | in progress |

RMSE=94.38 m/yr after chunk 1 is notably better than the PIG/Thwaites subdomain test's
converged 417.77 m/yr — likely because the continental average includes large slow-interior
regions that pull the mean down; not directly comparable to a fast-flow-only subdomain.
Cost was still dropping steadily (not flat) after chunk 1, so expect several more chunks
before this reaches `gttol=1e-8` or plateaus.

Cost weights (`friction_cf101=10`, `friction_cf103=100`, cf501=`0.0001`) are carried over
unchanged from the SSA inversion as a starting prior. At full continental scale cf501
dominates the total cost by several orders of magnitude more than in the PIG/Thwaites test
(`2.912e+07` vs `19.61`/`9.467e+05` at iteration 1) — may just reflect vertex-count scaling
in the regularization sum, or may mean the fit term is getting swamped continent-wide.
**Not yet investigated — re-sweep deferred until convergence, per user direction ("run
full continental now, tune later").**

---

## 4. Stages 3–6 — scaffolds

Not yet run. Each has correct model-loading/solver/save mechanics and real data source
paths, with open science decisions left as explicit `# TODO:`s rather than resolved
silently:

- **Stage 3 (`melt_gamma_tuning`)**: ISMIP6 basalforcings parameterization seeded from the
  published `gamma0`/`deltaT_basin` prior
  (`coeff_gamma0_DeltaT_quadratic_local_median.nc`), basin IDs derived from that same file
  (sidesteps a Mouginot-vs-Rignot basin-set compatibility question), Zhou ocean thermal
  climatology interpolated onto the mesh. Calibration target: ITS_LIVE ice-shelf melt
  observations. TODO: verify `elements2d`/`x2d`/`y2d` naming from `Model.extrude()` at run
  time; verify the ITS_LIVE `melt` variable's exact name/units.
- **Stage 4 (`ho_relaxation`)**: ~1 year transient, shock-damping only (not the historical
  spin-up itself). Kept separate from the existing `ssa_relaxation` baseline so
  `AIS3_relaxed.nc` stays available for comparison.
- **Stage 5 (`historical_dhdt_tuning`)**: 1995–2019 transient (bounded by data — MIPKIT's
  `dhdt_cpom` observational record doesn't reach 2025; state this caveat plainly, don't
  substitute a scenario silently), time-varying SMB from RACMO annual means, compared
  against MIPKIT's `dhdt_cpom`. TODO: the actual tuning loop (re-run stage 3/`cf501` at a
  few values, keep whichever minimizes mismatch RMSE) is not automated — each candidate
  requires re-running the full multi-stage chain, a substantial cost to automate blindly.
- **Stage 6 (`projection_ssp`)**: scenario-forced projection runs from the historical
  end-state, using real ISMIP7 forcing on disk
  (`/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/{CESM2-WACCM,MRI-ESM2-0}/{scenario}/`).
  TODO: end year (2100 vs. the 2300 the data actually supports — depends on what protocol
  this feeds), which of 6 candidate SMB parameterizations to use, whether to enable
  thermal physics, and the time-varying mesh interpolation for `tf`/SMB (currently loads
  the raw xarray objects but doesn't yet build the time-indexed ISSM-format arrays).

---

## 5. Open items

- Finish chunked `ho_friction_inv` convergence (§3.4), then re-sweep `cf101`/`cf103`/`cf501`
  weights for HO if warranted by the converged RMSE/friction-field pattern.
- Investigate the localized bad-fit region visible in the southeast of the PIG/Thwaites
  test's residual plot (`models/ais3_ho_friction_inv_pigthwaites_diagnostics.png`) — not
  blocking, not yet looked into.
- Stages 3–6 need real runs, not just scaffolding, once stage 2 lands.
