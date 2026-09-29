# ACCESS-AIS3 post-inversion pipeline work log

Design, debugging, and validation of the pipeline stages that run after the SSA
friction/rheology inversion documented in [`inversion_worklog.md`](inversion_worklog.md):
higher-order (HO) thermal spin-up → HO friction re-inversion → melt calibration →
relaxation → historical tuning → future projections. Mirrors Felicity's validated MATLAB
pipeline (`/g/data/au88/jr5971/access-dev2.git/antarctica-issm/matlab_fm/runme.m`)
structurally, adapted to pyISSM.

---

## 1. Stage status (updated 2026-09-29)

HO track (`config/ais_0.1.py`):

| stage | `steps` name | status | output |
|---|---|---|---|
| 1. HO thermal spin-up | `ho_thermal_steadystate` | **validated** | `AIS3_thermal_steadystate.nc` |
| 2. HO friction re-inversion | `ho_friction_inv` | **paused after chunk 3** (chunk 4 failed 5×, §4h/§4j); grounded RMSE ~94–99 m/yr. Downstream HO steps may be using chunk 2's friction, not chunk 3's — being checked (§5c) | `AIS3_ho_friction_inv.nc` (chunk 3, smoothed); `..._chunk3_backup_presmoothing.nc` (unsmoothed) |
| 3. Ocean melt calibration | `melt_gamma_tuning` (now builds the validated calibration, §5d) | **calibrated**: gamma_0=300 with per-basin refit deltaT (§4b–4c); thickness restored (§4i) | `AIS3_melt_gamma_tuning.nc` |
| 4. Post-calibration relaxation | `ho_relaxation` | **rerun 2026-09-29** (1 yr, RACMO 1979–1994 SMB + calibrated melt); relaxed state now saved as the model state, vertical mesh synced (§4y, §4z) | `AIS3_ho_relaxed.nc` (old: `..._prerelaxfix.nc`) |
| 5. Historical run (1995–2019) vs dH/dt | `historical_dhdt_tuning` | **complete but from the unrelaxed start with a stale vertical mesh** (§4m–§4u, §5a); HO flows ~10–15% too fast (§5a–§5c). Relaxed-start rerun (`AIS3_HIST_TAG=_relaxfix`) on hold until §5c is resolved | `AIS3_historical_1995_2019.nc`; chunk-1 rerun `..._chunk1rerun.nc` |
| 6. Future projections (SSP-forced) | `projection_ssp` | scaffold, untested; TF/SMB-onto-mesh interpolation still a TODO here (implemented in the SSA scaffold) | `AIS3_projection_{gcm}_{scenario}.nc` |

SSA track (`config/ais_0.1_SSA.py`; SSA deliberately starts downstream steps from the inverted,
not the relaxed, geometry — by design, see the §4z correction):

| stage | `steps` name | status | output |
|---|---|---|---|
| 1. Inversion + solve (Budd p=q=1) | `ssa_inverted_solve_budd` | **done**; thickness restored (§4k) | `AIS3_SSA_inverted.nc` |
| 2. Relaxation (20 yr, zero SMB/melt) | `ssa_relaxation_budd` | **done**; not propagated downstream, by design | `AIS3_SSA_relaxed.nc` |
| 3. Ocean melt calibration | `melt_gamma_tuning_ssa` (now builds the validated calibration, §5d) | **calibrated**: gamma_0=300, refit deltaT (§4k, §4n) | `AIS3_melt_final_ssa.nc` |
| 4. Historical run (1995–2019) | `historical_dhdt_tuning_ssa` | **done**: area-mean close to CPOM after a ~6-yr start-up transient; no interannual skill; misses the West Antarctic acceleration (§4q–§4t, §5b) | `AIS3_historical_1995_2019_SSA.nc` |
| 5. Future projections | `projection_ssp_ssa` | scaffold; time-varying TF/SMB interpolation implemented and smoke-tested, not run. Uses absolute `acabf` from placeholder `SDBN1-8000m`; SMB anomaly method and elevation feedback still TODO | `AIS3_projection_ssa_{gcm}_{scenario}.nc` |

Scaffolded stages have correct model loading / solver setup / save pattern and real data
paths, but open science decisions are marked `# TODO:` in the code rather than silently
resolved — do not treat their intermediate outputs as validated.

Validation caveat for both tracks: the production "grounded dH/dt mismatch RMSE" (HO 1.768,
SSA 1.402 m/yr) is a per-vertex spatial RMSE of the 24-yr mean rate against only CPOM's 2019
slice. It cannot see a domain-wide bias and is not a temporal validation. Use the area-weighted,
fixed-coverage, per-basin time series instead (§4r–§4t, §5b).

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

**Second bug found (chunk 2, 2026-08-27): the resume never actually resumed.** After
chunk 1 saved, chunk 2's iteration 1 cost matched chunk 1's own iteration 1 to 4
significant figures (`3.0065e+07`, identical) — the checkpoint file was being loaded, but
`save_model()`/`load_model()` do **not** round-trip `md.friction.<field>` from
`md.results.StressbalanceSolution.<fric_control>` — the optimized value only ever lives
in `.results` (same convention `ssa_inverted_solve` already handles explicitly elsewhere
in this file for the SSA-stage graft, which is what gave this bug away). Every resumed
chunk was silently restarting from the pre-chunk-1 field, burning a full chunk's compute
(and a walltime-cap-length delay) for zero progress. Fixed by explicitly copying
`md.results.StressbalanceSolution.<fric_control>` into `md.friction.<fric_field>`
immediately after loading a resumed checkpoint, guarded by a `resuming` flag so a fresh
start (no checkpoint yet) is unaffected. Confirmed fixed: the next resubmission's
iteration 1 landed at `f(x)=4.261e6`, close to (and improving on) chunk 1's own endpoint
of `4.9165e6`, not a reset.

**Third issue (chunk 2, resource tuning): tried `normal` queue to dodge `hugemem`
congestion, OOM'd.** `hugemem` was badly congested (chunk 2 queued for hours, estimated
start ~4h out) — tried switching to `normal` at 768 cores/3072GB (16 nodes × 192GB,
`pbsnodes`-confirmed per-node spec) to match `hugemem`'s ~2900GB footprint with more,
smaller nodes. Accepted at submission (a direct `qsub` dry-run confirmed no
`hugemem`-style walltime-vs-core-count cliff on `normal`), but OOM'd within 2 minutes
(SIGKILL, 2.8TB used of 3.0TB requested at 768 ranks — up sharply from 1.39-1.53TB at 96
ranks on `hugemem`). Confirmed this is a dead end, not just under-provisioned: (a) memory
need is **not** flat with rank count — it grew with the 8x rank increase, consistent with
[`inversion_worklog.md`](inversion_worklog.md) §8's per-rank-overhead finding for the
SSA-scale mesh; (b) Gadi's PBS ties memory strictly to a fixed `ncpus`/48-derived node
count on `normal` — a direct `qsub` test requesting the *same* 768 cores with more memory
(6144GB, i.e. "same cores, more memory-only nodes") was flatly rejected ("384.0GB per
node exceeds what normal can provide"), so node count can't be decoupled from core count
here. Net: getting more memory on `normal` requires more ranks, which then need even more
memory — a losing spiral, since `normal`'s 192GB/node ceiling is ~7.5x below `hugemem`'s
~1450GB/node. Reverted to `hugemem`/96 cores; the queue-wait is the lesser problem.

### 3.4 Progress so far

| chunk | job | outcome | cost f(x) | RMSE (m/yr) | notes |
|---|---|---|---|---|---|
| 1 (unchunked, 500-step budget) | 177331341 | killed at 24h, iter 13 | 3.0065e7 → 3.211e6 | — | **no checkpoint saved — lost** |
| 1 (chunked, maxsteps=8) | 177469174 | completed, exit 0, 21h6m | 3.0065e7 → 4.9165e6 | 94.38 | saved |
| 2 (chunked, `normal` 768c/3072GB) | 177583055 | **OOM'd, 2min, SIGKILL** | — | — | dead end, see above — reverted to `hugemem` |
| 2 (chunked, `hugemem`, resume bug present) | 177587209 | completed, exit 0, but re-ran ch.1's exact trajectory | 3.0065e7 → 3.2379e6 | — | **wasted — resume bug, see above** |
| 2 (chunked, `hugemem`, resume bugfixed) | 177599817 | completed, exit 0, 8h27m | 4.261e6 → 3.2379e6 | **97.22** | saved, genuine continuation confirmed |
| 3 (chunked, `hugemem`) | 177657113 | completed, exit 0, 7h53m | 3.0771e6 → 2.8087e6 (via a mid-chunk spike to 3.92e7, iter 2) | **99.15** | saved, see mid-chunk note below |
| 4 (chunked, `hugemem`, restol=0.01/maxiter default) | 177735239 | **killed at 24h walltime, exit -29** | 2.7542e6 → stuck (spike to 5.26e7, iter 2, never recovered) | — | **no checkpoint saved — lost, see fix below** |
| 4 (chunked, `hugemem`, restol=0.001/maxiter=300/maxsteps=5) | 177819609 | **killed manually at 21h11m, iter 2 never finished** | 2.7554e6 → stuck | — | **no checkpoint saved — lost, restol reverted, see below** |
| 4 (chunked, `hugemem`, restol=0.01 (reverted)/maxiter=300/maxsteps=5) | 177904997 | **killed at 24h walltime again, exit -29, iter 2 exceeded maxiter=300 without converging** | 2.7542e6 → stuck | — | **no checkpoint saved — lost, third failure, see below** |
| 4 (chunked, `hugemem`, dfmin_frac=0.3 added) | 178011317 | running | — | — | in progress, resuming from chunk 3, targets the actual restart-step mechanism |

**Chunk 3 had a large mid-chunk cost spike that fully recovered.** Iteration 1 opened at
`f(x)=3.0771e6` (a genuine, correctly-resumed continuation of chunk 2's `3.2379e6`
endpoint), but iteration 2 jumped to `f(x)=3.9227e7` (12.75x higher, grad norm 277) with
no nonlinear-solver warning this time (unlike chunk 1's transient iteration-2 issue) —
consistent with the known cost of this chunked-restart design: m1qn3's internal
quasi-Newton curvature estimate isn't preserved across restarts, so each fresh chunk can
take a poorly-calibrated first step. It recovered over the next several iterations
(3.71e7 → 2.06e7 → 5.58e6 → 3.06e6 → 3.03e6 → 2.81e6), finishing genuinely below both
chunk 1 and chunk 2's endpoints. Worth watching whether this recurs on future chunks; if
it becomes a pattern rather than a one-off, the fix would likely be damping the first
post-restart step size rather than accepting the full spike-and-recover cycle each time.

**Chunk 4 confirmed it's a pattern, and this time it didn't recover — a full 24h walltime
loss.** Job 177735239: iteration 1 opened at `f(x)=2.7542e6` (correctly resumed from
chunk 3's `2.8087e6`), iteration 2 spiked to `5.2552e7` (19x, grad norm 286) — this time
*with* `maximum number of nonlinear iterations (100) exceeded` warnings (the chunk-1-style
variant, not chunk-3's warning-free one). Unlike chunk 1 and chunk 3, it never recovered:
iteration 3's nonlinear solve kept hitting the 100-iteration cap repeatedly and the job was
killed at the 24h wall-clock limit still stuck there (`walltime 86422 exceeded limit
86400`) — 2-3 iterations consumed the entire budget instead of the usual ~8. Because the
process was killed rather than returning normally, **no checkpoint was saved** — the full
24h (job cost ~6916 SU) produced zero progress; `AIS3_ho_friction_inv.nc` on disk is still
chunk 3's checkpoint, confirmed unchanged.

**Root cause and fix, informed directly by this project's own SSA-era history**
(`docs/inversion_worklog.md`, the `ssa_friction_inv_reg_lcurve` notebook): `ho_friction_inv`
was still using `stressbalance.restol = 0.01` — the *exact* value that project's own
p=q=1 friction inversion work root-caused as "too loose", letting the nonlinear forward
solve settle into different Newton states for identical `C` and corrupting the cost/gradient
signal m1qn3's line search relies on (fixed there with `restol=0.001`). `ho_friction_inv`
also never set `stressbalance.maxiter` at all, silently inheriting ISSM's implicit default
of 100 — exactly the number in the "(100) exceeded" messages. Both chunk 1's and chunk 4's
post-restart spikes carried this exact warning; chunk 3's warning-free spike (which
recovered cleanly) suggests the mechanism isn't the *only* source of post-restart
instability, but it's a real, previously-validated-elsewhere contributor. Fixed: tightened
`restol` to `0.001` and explicitly raised `stressbalance.maxiter` to `300`, giving a
genuinely-converging-but-slow nonlinear solve room to actually finish instead of being
truncated into a bad state. Also reduced the per-chunk `maxsteps`/`maxiter` from 8 to 5 as
extra margin, since individual iterations may now legitimately take longer under the
higher cap. Resubmitted (job 177819609), resuming from chunk 3's checkpoint (2.8087e6) —
chunk 4's failed attempt cost compute but no data.

**`restol=0.001` fix reverted — likely too aggressive for this mesh, not just wrong
timing.** The retry (177819609) resumed correctly (iteration 1 opened at `2.7554e6`,
matching chunk 3's endpoint) but iteration 2's nonlinear solve — the same post-restart
spike pattern as before — ran **21+ hours without finishing**, on track to be killed at
the 24h cap with nothing saved again. Killed manually rather than waiting out the
inevitable. The SSA-era precedent for `restol=0.001` (`docs/inversion_worklog.md`) was
validated on a ~1.6M-vertex 2D mesh; this is a 23.8M-node 3D HO mesh where the linear
solve is already ~92% of total runtime (confirmed in earlier chunks' own timing
breakdowns) — a 10x tighter tolerance costs far more per extra Newton iteration here.
Before the fix, spikes recovered within a handful of iterations, comfortably inside
budget (chunks 1, 3); after it, a single iteration ran over 21h and still didn't finish —
worse than the failure the fix was meant to solve. Also: the original chunk-4 failure
(hitting the implicit `maxiter=100` cap) is equally explained by "simply needed more
iterations" as by the SSA-era "converging to a corrupted Newton state" mechanism — no
direct evidence here it was the latter. Reverted `restol` to `0.01`; kept `maxiter=300`
(the uncontroversial half of the original fix — pure headroom, no added per-iteration
cost). Resubmitted (job 177904997), resuming from chunk 3's checkpoint — the second
`restol=0.001` attempt cost ~21h of compute but no data, same as the first chunk-4
failure it was meant to prevent.

**Third chunk-4 failure (2026-09-02), and the actual root cause found.** Job 177904997 ran
the full 24h, iteration 1 opened correctly (`f(x)=2.7542e6`, matching chunk 3's endpoint),
but iteration 2's nonlinear stress-balance solve hit `maximum number of nonlinear iterations
(300) exceeded` and never recovered — killed at walltime, exit -29, zero checkpoint saved
(~2284 SU burned, ~6923 SU on the compute job itself). This ruled out "just needed more
headroom" as the explanation: 3x the original iteration cap (100→300) still wasn't enough,
so both prior fixes (tighter `restol`, then raised `maxiter`) were treating the symptom, not
the cause. **Root cause**: `pyissm`'s `inversion.m1qn3` class has a `dfmin_frac` parameter
("expected reduction of cost function during the first step", default **1.0**) that
`ho_friction_inv` never set. At 1.0, m1qn3 calibrates its very first step assuming a 100%
cost reduction is achievable — a wildly oversized step. Since every chunk restart loses
m1qn3's internal curvature estimate (no serialization mechanism across resubmissions), this
exact miscalibration recurs at **every** chunk boundary — consistent with chunks 1 and 3
both seeing large iteration-2 cost spikes (12–19x) that happened to recover, and chunk 4
(three separate attempts) landing somewhere the forward nonlinear solve genuinely can't
converge from. **Fix**: added `md.inversion.dfmin_frac = 0.3`, a much more conservative
expected first-step reduction, so the post-restart step is properly scaled instead of
overshooting — targets the actual mechanism rather than giving the solver more room to fail
slowly. Resubmitted (job 178011317), resuming from chunk 3's checkpoint.

RMSE ticked **up** slightly across both chunk 1→2 (94.38 → 97.22) and chunk 2→3
(97.22 → 99.15) even though total cost keeps falling — consistent with cf501
(regularization) dominating the cost by orders of magnitude at full continental scale
(flagged below): the optimizer may be trading fit for smoothness at this stage, not
diverging. A direct look at the full-domain residual map (`ais3_ho_friction_inv_diagnostics.png`,
±100 m/yr range) shows why: the bulk interior fits essentially perfectly (white/near-zero),
but faint, coherent positive-residual bands trace the fast-flow trunks and coastal margins
-- exactly where cf501's gradient penalty would be expected to resist C sharpening enough
to match the misfit term's preference. PIG/Thwaites specifically (`check_pig_residual.py`,
`ais3_ho_friction_inv_pig_zoom.png`) has RMSE=193.49 m/yr, ~2x the continental figure, with
a tight cluster of its worst points (up to 1572 m/yr off, model ~2-5x too fast) right at the
trunk near the grounding line (x=-1,608,000, y=-263,600) -- the same "localized bad-fit
region" already flagged in the original PIG/Thwaites subdomain test. Worth revisiting once
this converges: the deferred cf101/cf103/cf501 re-sweep (§ below) is the natural next step
if this pattern persists.

RMSE=94.38 m/yr after chunk 1 is notably better than the PIG/Thwaites subdomain test's
converged 417.77 m/yr — likely because the continental average includes large slow-interior
regions that pull the mean down; not directly comparable to a fast-flow-only subdomain.
Cost was still dropping steadily (not flat) after chunk 2, so expect several more chunks
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
  observations. **Three gaps closed (2026-08-30), still never actually run:**
  (1) `elements2d`/`x2d`/`y2d` naming confirmed correct by reading `Model.extrude()`
  directly (`Model.py:815-818` unconditionally sets these on any extruded model) — the
  runtime fallback never actually triggers, not a real risk after all.
  (2) **Real bug fixed**: the code used `'melt'` (dims `time,y,x`, quarterly 1992-2017,
  a 3D field) as the calibration target, but `xr_to_mesh` assumes a 2D `(y,x)` rectilinear
  grid — checked the actual file, switched to `'melt_mean'` (the dataset's own
  pre-computed time-mean, exactly the right steady-state target; coordinate names `x`/`y`
  confirmed to match `xr_to_mesh`'s defaults).
  (3) **No hugemem override** — this step operates on the same ~23.8M-node mesh that
  OOM'd `ho_thermal_steadystate` outright at the shared cluster's default 190GB/normal
  config; added the same 96-core/2900GB/hugemem override proactively (mirrors
  `ho_thermal_steadystate`'s forward-solve config, since this step has no adjoint solve
  and is likely lighter — untested, revisit if oversized). Also swapped a per-gamma-value
  `md.extract(all-true-mask)` "cheap copy" (which still triggers a full ISSM
  mesh-connectivity rebuild) for `copy.deepcopy()` (genuinely cheap, correct since every
  sweep point shares the identical mesh).

  **First actual run (2026-08-31), launched alongside `ho_friction_inv` chunk 4 against
  chunk 3's stable checkpoint** (independent parameter, no need to wait for full friction
  convergence): submission phase (`launch_melt_gamma_submit.pbs`, toggling the shared
  `steps`/`save` globals to `['melt_gamma_tuning']`/`False` just for this one job, reverted
  back to `['ho_friction_inv']`/`True` once the job was confirmed *running* — not just
  queued — so a concurrently-queued `ho_friction_inv` resubmission couldn't read the wrong
  settings) loaded cleanly, confirmed 16 basins and correctly interpolated the Zhou
  climatology onto all 23,833,290 mesh vertices, but **failed model-consistency on the
  first sweep run**: `basalforcings.basin_id has shape (3173063,), expected (44422882,)`.
  Real bug: `basin_id` was computed per-2D-element (matching `elements2d`'s count) but
  ISSM expects it per-3D-element (`numberofelements2d` x 14 vertical layers =
  3,173,063 x 14 ≈ 44,422,882) — basin identity is horizontal-only but still needs
  replicating up every column. Fixed with `mesh._project_3d(md, vector=basin_id_2d,
  type='element', layer=0)`, the project's own established 2D→3D replication utility
  (same one used elsewhere for vertex fields, just with `type='element'`). Failed cheaply
  — the consistency check runs before any of the 5 sweep jobs actually submit, so no
  compute was wasted, just a ~5min round trip.

  **Second bug on retry**: consistency check passed, but marshalling crashed
  (`KeyError: '101'` in `class_utils.marshall_inversion_cost_functions`) — a *third*
  distinct bug, not a recurrence. `AIS3_ho_friction_inv.nc` still carries its own m1qn3
  control-inversion config (`iscontrol=1`, `cost_functions=[101,103,501]`) inherited from
  the friction inversion; this step never reset it, unlike `ho_relaxation`/
  `historical_dhdt_tuning`, which explicitly clear it (`md.inversion.iscontrol = 0`)
  before their own forward-only solves. Added the same line here. Also failed cheaply —
  crashed during local marshalling, before any job reached the cluster.

  **Third attempt: marshalling/submission fixes held, but 4 of 5 sweep runs then crashed
  mid-solve — a fourth distinct bug.** All 5 marshalled and submitted cleanly to `hugemem`
  (jobs 177826799/857/887/929/965, 96 cores/2900GB/24h each, ~3min/~415GB actually used —
  confirms this step is far lighter than the friction inversion, as expected). But 4 of 5
  (g1-g4; g5 was killed once the pattern was clear) crashed within ~3 minutes, exit 1:
  `FetchDataToInput error message: "md.smb.mass_balance" not found in binary file`.
  Masstransport reads `smb.mass_balance` as an input even with `issmb=0` —
  `AIS3_ho_friction_inv.nc` never had it populated (`ais_0.1_param.py` never sets it,
  a gap `ssa_relaxation` already hit and fixed with a zero placeholder for its own short
  diagnostic run). Applied the identical fix (`md.smb.mass_balance =
  np.zeros(md.mesh.numberofvertices)`) — defensible here too, since this solve's entire
  purpose is evaluating the basalforcings melt flux, not simulating real SMB. Cleaned up
  and resubmitted the full 5-run sweep (job 177827321 spawning fresh
  `AIS3_melt_gamma_tuning_g{1..5}` jobs); in progress. Once these finish, rerun with
  `steps=['melt_gamma_tuning']`/`save=True` (phase 2) to load results, compare against
  ITS_LIVE `melt_mean`, and save the best `gamma_0` to `AIS3_melt_gamma_tuning.nc`.

  **Running tally for this stage's first real run: 4 distinct bugs found and fixed**
  (`basin_id` 2D→3D shape, `spcthickness` bare-scalar-NaN, stale `iscontrol`/
  `cost_functions`, missing `smb.mass_balance`) — every one a genuine gap this mesh/step
  combination had simply never exercised before, not a repeat of the same mistake. Matches
  this project's broader pattern: a "scaffold" step with plausible-looking code reliably
  turns up several such gaps on its first real run against full-scale data.

  **Sweep 1 result (all 5 runs succeeded, ~4.5min/22SU each — confirms this step is far
  lighter than the friction inversion, as expected with no adjoint solve):**

  | gamma_0 | melt RMSE vs ITS_LIVE `melt_mean` (m/yr) |
  |---|---|
  | 5537.7 (0.5x prior) | **12.60** |
  | 8306.6 (0.75x) | 14.21 |
  | 11075.5 (1.0x, published prior) | 15.91 |
  | 13844.3 (1.25x) | 17.69 |
  | 16613.2 (1.5x) | 19.53 |

  RMSE increases monotonically across the *entire* tested range — the "best" point sits
  right at the sweep's low edge, meaning the sweep never bracketed an actual minimum. The
  step's own logic saved `gamma_0=5537.73` as "best" to `AIS3_melt_gamma_tuning.nc`, but
  that's only the best of five points on a still-falling curve, not a validated optimum —
  flagged to the user rather than treated as a finished calibration.

  **Sweep 2 (lower range, extending the trend)**: `gamma_grid` changed to
  `{0.05, 0.1, 0.15, 0.2, 0.3, 0.4}x` the published prior (6 points), to actually find
  where RMSE turns over — or confirm it doesn't, which would itself be informative
  (RMSE monotonically falling all the way toward `gamma_0->0` could mean most of the
  domain has near-zero true melt and the metric is being gamed by turning melt off almost
  everywhere, rather than reflecting a physically meaningful optimum — worth checking the
  *spatial* residual structure, not just the aggregate RMSE, once this sweep is in).
  Old `execution/AIS3_melt_gamma_tuning_g{1..5}` directories cleaned up first (already
  consumed into the saved `.nc`, and would otherwise name-collide with the new sweep's own
  `g1`-`g6`). Submitted (job 177837962, spawning 6 fresh hugemem runs); in progress.

  **Process note for reuse**: `melt_gamma_tuning`, like the older `ssa_friction_inv_lcurve`
  -style steps, uses the shared `steps`/`save` globals at the top of `ais_0.1.py` as its
  submit/load toggle — a genuinely different, valid pattern from `ho_friction_inv`'s
  single-shot synchronous submit-and-wait. Since those globals are shared with the
  concurrently-running `ho_friction_inv` chunk cycle, each toggle here was done carefully:
  set `steps=['melt_gamma_tuning']`/`save=False`, submit, wait for the job to reach `R`
  (running) — not just queued — then revert back to `steps=['ho_friction_inv']`/
  `save=True`.

  **This "wait for `R`" protocol turned out to be insufficient (2026-09-01) — a real race
  actually happened, not just a theoretical risk.** PBS marking a job `R` only means the
  scheduler allocated resources and started the outer shell script; it does **not** mean
  the Python process inside has reached the module-level `steps`/`save` lines yet (conda
  activation, module loads, etc. all happen first). The melt-gamma phase-2 submission
  (job 177904323) reached `R`, was reverted immediately after per the old protocol, but its
  Python process hadn't actually read `steps` yet — it read the already-reverted
  `['ho_friction_inv']` and silently ran a **second, unintended `ho_friction_inv` attempt**,
  resuming from the same checkpoint and marshalling into the *same* `execution/
  AIS3_ho_friction_inv/` directory as the real, concurrently-submitted chunk-4 retry
  (177904997). The two marshalling writes collided — the duplicate's own hugemem compute
  job (177904855) died within minutes (most likely read a partially-overwritten `.bin`
  file), while the legitimate chunk-4 job (177905055, marshalled second) survived
  uncorrupted. The stray outer job (177904323) was left hung waiting on its dead
  duplicate's lock file — killed manually rather than let it burn its walltime budget for
  nothing.

  **Protocol fixed**: instead of polling `qstat` for `R`, poll `qcat <jobid>` for the
  step's own distinctive first print line (e.g. `"OCEAN BASAL-MELT GAMMA CALIBRATION"`) —
  genuine evidence the Python process has executed past the module-level globals, not just
  that PBS handed it a node. Used successfully for the corrected melt-gamma resubmission
  (job 177906268) with no recurrence. **Any future `steps`/`save` toggle on this shared
  script should use this stronger check, not the `qstat` `R` state alone.**
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

## 4a. ISMIP7-informed melt calibration (`config/melt_ismip7_calibration.py`)

The `melt_gamma_tuning` sweeps above (§4) use a crude per-vertex RMSE-vs-ITS_LIVE metric —
never bracketed a real minimum and doesn't match how the field actually calibrates this
parameter. The user supplied the real ISMIP7 AIS ocean focus group draft protocol
(`ISMIP7_AIS_ocean_focus_group_Protocol_Sept_2025.pdf`, Sept 2025) and asked for something
"somewhat ISMIP7 compliant, if it produces better calibrated results." Wrote a standalone
script (own local `steps`/`PHASE` variables, isolated from `ais_0.1.py`'s shared globals per
the race-condition lesson in §4) implementing the protocol's real 3-term objective:
`I = a1*J1/median(J1) + a2*J2/median(J2) + a3*J3/median(J3)`, Monte Carlo-sampled weights
(10,000 samples, ≥ the protocol's minimum), reporting the 5th/50th/95th percentile winning
`gamma_0`.

- **J1** — basin-aggregated (16 official IMBIE2 basins,
  `imbie2/basin_numbers_ismip8km_v2.nc`) mean absolute error vs. the Paolo/Adusumilli
  present-day melt target (`meltobs/Melt_Paolo_Err_Adusumilli_imbie2_v3.csv`).
- **J2** — BFRN-bin-weighted fit (`bfrns/BFRN_ismip8km_v2.nc`), target approximated by
  distributing each basin's observed total proportionally to floating area within each bin
  (the protocol doesn't fully specify per-bin target derivation — flagged as an
  approximation, not silently treated as exact).
- **J3** — warming-sensitivity fit against Mathiot et al. 2023 NEMO cold/warm ocean-model
  states (`ocean_modelling_data/Mathiot_NEMO_{cold,warm}_{TF,m}.nc`). Only run once at the
  published prior `gamma_0=11075.45`; each ensemble member's own cold/warm response is
  estimated by exploiting the ISMIP6 quadratic-local melt formula's own exact linear
  dependence on `gamma_0` (confirmed empirically: `BMB = 0.1507 * gamma_0` Gt/yr, exact
  across the full tested range 100–25000) rather than running 18 more cluster jobs.

9-point `gamma_0` ensemble (Zhou climatology) + 2 Mathiot cold/warm sensitivity runs, all 11
forward-solve cluster jobs completed cleanly on the first submission.

**Known gap, stated plainly in the module docstring, not hidden**: the protocol wants
`deltaT_basin` re-derived per candidate `gamma_0` (each value should get its own basin
corrections re-fit to match present-day melt before scoring). Not implemented — would need
either ~9x more cluster runs (root-finding deltaT per basin per candidate) or a from-scratch
reimplementation of ISSM's internal melt formula unverifiable against the C++ source here.
`deltaT_basin` is held fixed at the published prior for every member.

**Bug found and fixed before the J3-affecting run**: units mismatch on the Mathiot reference
melt fields. `Mathiot_NEMO_{cold,warm}_m.nc`'s `melt_rate` variable is `kg/m2/a` (mass flux,
confirmed via the file's own `units` attribute), not m ice/yr as the original code assumed
(comment said "m/yr presumably" — never verified against the file). The shared
`basin_aggregated_bmb()` helper already multiplies by `rho_ice` internally (expects m ice/yr
in), so feeding it the raw kg/m2/a field double-applied the density conversion. First run
produced `ref total=1,032,352 Gt/yr` (cold) / `13,671,674 Gt/yr` (warm) — obviously
unphysical (continent-wide totals should be O(1,000–15,000) Gt/yr even for an extreme
warm-ocean sensitivity test) — and J3 came out nearly flat across the whole `gamma_0` range
(456,618–459,489), i.e. numerically included but contributing no real discriminating signal.
Fixed by dividing the raw grid by `rho_ice` before interpolation
(`config/melt_ismip7_calibration.py`, J3 block). Rerun (job 177923264, only reloading
already-completed cluster results, ~34min) produced sane totals (`ref total=1125.8` cold /
`14909.1` warm Gt/yr, same order of magnitude as `param total`) and a J3 curve with real
structure (U-shaped, minimum near `gamma_0=3000`).

**Final result (2026-09-01)**:

| gamma_0 | J1 (basin MAE, Gt/yr) | J2 (BFRN-weighted MAE) | J3 (warming-sensitivity MAE) |
|---|---|---|---|
| 100.0 | 53.118 | 505.910 | 489.562 |
| 300.0 | 51.234 | 482.663 | 466.502 |
| 1000.0 | 44.639 | 401.296 | 385.794 |
| 3000.0 | 26.236 | 168.819 | **260.697** (J3 min) |
| 5537.7 (0.5x prior) | **16.151** (J1 min) | **139.820** (J2 min) | 360.562 |
| 8306.6 | 25.687 | 448.607 | 584.885 |
| 11075.5 (published prior) | 51.632 | 769.857 | 864.318 |
| 15000.0 | 88.406 | 1226.040 | 1293.889 |
| 25000.0 | 182.108 | 2388.422 | 2404.799 |

Monte Carlo (10,000 samples): **5th percentile gamma_0 = 3000.0**, **50th = 5537.7**,
**95th = 5537.7** m/yr. J1 and J2 both minimize at 5537.7 (half the published ISMIP6 prior)
and dominate most MC draws; only when a sample weights J3 heavily does the answer shift down
toward 3000, where J3 alone bottoms out. This is a genuinely more complete, protocol-shaped
result than §4's RMSE sweep (basin-aggregated Gt/yr metric, real observational targets,
buttressing-region weighting, ocean-model sensitivity term, percentile uncertainty) — subject
to the fixed-`deltaT_basin` gap above.

---

## 5. Open items

- Finish chunked `ho_friction_inv` convergence (§3.4), then re-sweep `cf101`/`cf103`/`cf501`
  weights for HO if warranted by the converged RMSE/friction-field pattern.
- Investigate the localized bad-fit region visible in the southeast of the PIG/Thwaites
  test's residual plot (`models/ais3_ho_friction_inv_pigthwaites_diagnostics.png`) — not
  blocking, not yet looked into.
- Stages 3–6 need real runs, not just scaffolding, once stage 2 lands.

## 4c. deltaT_basin re-derivation reopened, gamma_0 revised to 300

§4a/4b's `gamma_0=5537.7` held `deltaT_basin` fixed at the published prior — initially
**deliberately deferred** (2026-09-01), judged not worth dozens of cluster round-trips
unless basin-specific accuracy was actually needed. **Reopened the same day** after
`melt_deltaT_sensitivity_test.py` (a cheap +1°C uniform shift at the published `gamma_0`,
run while genuinely idle) came back showing `deltaT_basin` has enormous, highly
basin-dependent leverage — total continental melt more than quadrupled (1669.5 → 7010.3
Gt/yr) from a 1°C shift, with individual basins swinging 2–14x. This meant the earlier
"probably fine as-is" framing was wrong: `deltaT_basin` is the dominant lever on melt
magnitude, not a minor nuisance parameter next to `gamma_0`'s gentle linear scaling.

**`melt_deltaT_basin_refit.py`**: real per-basin secant root-finding against the
Paolo/Adusumilli target, via actual ISSM forward solves (not a from-scratch reimplementation
of ISSM's internal melt formula — unverifiable against the C++ source here). Exploited two
already-established facts to make this affordable: (1) melt scales exactly linearly in
`gamma_0` for any fixed `deltaT_basin`, so the existing g0-g8 ensemble gave a free, exact
bootstrap point per candidate; (2) the +1°C sensitivity test gave an initial per-basin slope
estimate for the first Newton step, rescaled per candidate via the same linearity. Basins are
independent (each `deltaT_basin[b]` only affects basin b's own melt), so all 16 basins × 9
`gamma_0` candidates were root-found simultaneously, one forward solve per candidate per
round. Converged in 3–5 rounds for 8/9 candidates (all 16 basins within 5%-or-1 Gt/yr
tolerance); only the extreme `gamma_0=100` candidate fell short (14/16) after hitting the
5-round cap — not important, far from the eventual winner.

**`melt_deltaT_refit_recalibrate.py`**: recomputed J1/J2 using each candidate's own refit
melt field. Key conceptual point: with `deltaT_basin` fit per basin, **J1 becomes ~0 for
every converged candidate by construction** — it's solved by design, no longer a
discriminator. The real signal moves to J2 (within-basin/BFRN-bin spatial pattern, which a
spatially-uniform-per-basin `deltaT_basin` cannot correct). J2 showed a sharp, clean minimum
at `gamma_0=300` (J2=12.4 vs. 45.4–346.5 for every other candidate — not a marginal
preference). The J1+J2-only Monte Carlo (methodologically clean, no stale metric mixed in)
was unanimous: 5th/50th/95th percentile `gamma_0` = **300/300/300**.

J3 (warming sensitivity) was left **stale** — still the fixed-deltaT value from §4a, not
recomputed under each candidate's refit `deltaT_basin` (would need 18 more solves: Mathiot
cold/warm per candidate). Mixing stale J3 into the full objective pulls the 95th percentile
up to 3000 in the MC output, but this is an artefact of combining a refit metric with a
stale one, not a real signal. **Decision (2026-09-02): accept `gamma_0=300` from the clean
J1+J2 result rather than spend 18 more solves on J3** — the J2 signal is already sharp and
unanimous; revisit only if downstream work later needs the warming-sensitivity term trusted
in detail.

**`finalize_melt_calibration.py`** (rewritten): salvages g1's own converged refit run
(`AIS3_deltaT_refit_g1_r5`, `gamma_0=300` with its own per-basin-refit `deltaT_basin`,
already solved — no new compute needed) into `models/AIS3_melt_gamma_tuning.nc`, superseding
the fixed-deltaT `gamma_0=5537.7` version that file held since §4a. **`gamma_0=5537.7` is now
superseded, not the working value — use 300.**

## 4b. Spatial cross-check: RMSE sweep vs. ISMIP7 calibration (confirms §4's RMSE result invalid)

§4's RMSE sweep and §4a's ISMIP7 calibration disagreed by 10x on the best `gamma_0`
(553.77 vs. 5537.7) — never reconciled at the time. §4 had already flagged the RMSE result
as suspect (monotonically falling toward `gamma_0->0`, possibly gaming the metric by turning
melt off almost everywhere). Wrote `config/compare_melt_spatial_residual.py` to check
directly: loaded both saved results (`AIS3_melt_gamma_tuning.nc` for 553.77,
`execution/AIS3_melt_ismip7_g4` for 5537.7), compared each against ITS_LIVE `melt_mean` at
every floating 2D mesh vertex (351,332 of them), computing RMSE (as a consistency check
against the documented values — reproduced 10.11 exactly), bias, and the fraction of
floating area with near-zero (< 0.1 m/yr) simulated melt. Plot saved to
`models/melt_spatial_residual_comparison.png`.

**Confirmed the metric-gaming suspicion.** gamma_0=553.77: RMSE=10.11 m/yr but
77.8% of the floating domain has near-zero simulated melt — visually, an almost
uniform near-zero field with no correspondence to observed melt hotspots at all.
gamma_0=5537.7: RMSE=12.55 m/yr (nominally worse) but only 53.0% near-zero, and the
spatial map shows melt concentrated in the same locations ITS_LIVE actually observes
elevated melt (Antarctic Peninsula region and a cluster on the East Antarctic coast).
The RMSE sweep's "better" score comes from predicting almost no melt anywhere, which
trivially lowers squared error against a target dominated by small/near-zero values,
not from reproducing real spatial structure.

**Conclusion: `AIS3_melt_gamma_tuning.nc` (gamma_0=553.77) is not a valid calibration
result and should not be used.** The ISMIP7-informed calibration (§4a, gamma_0=5537.7 at
median/95th percentile) is the credible one — basin-aggregated Gt/yr fitting against real
observational targets (Paolo/Adusumilli) turned out to be robust to exactly the failure
mode that broke the naive per-vertex RMSE metric. **Superseded again by §4c — the final
working value is `gamma_0=300` with per-basin-refit `deltaT_basin`.**

## 4d. Stages 4/5 (`ho_relaxation`, `historical_dhdt_tuning`) — real bugs found on first run

Both were previously logged as "scaffold, correct model loading/solver setup" but had never
actually been run. First real run (2026-09-02, `steps=['ho_relaxation']`, using the finalized
`gamma_0=300` melt calibration) found three genuine bugs, none hypothetical:

1. **Never actually submitted.** Both steps had `waitonlock=0` combined with `if save: ...
   load_only=True` — the exact "never submits" bug already found and fixed in
   `ssa_inverted_solve`/`ssa_relaxation`/`ho_thermal_steadystate` earlier this project, just
   never backported to these two. `load_only=True` only loads an already-finished prior run
   (`pyissm/model/execute.py:1078-1082` unconditional early return); since neither step had
   ever run before, this would have silently failed to load anything. Fixed to the single
   synchronous submit-and-wait call (`load_only=False`, `waitonlock=1440` minutes) used
   everywhere else in the pipeline.
2. **Undersized cluster.** Both steps used the shared default `cluster` (48 cores/190GB/
   normal) — sized for the SSA track's much smaller mesh, not the ~23.8M-node HO mesh these
   steps actually operate on. Every other HO-scale step (`ho_thermal_steadystate`,
   `ho_friction_inv`, `melt_gamma_tuning`) already needed the hugemem override after OOM'ing
   outright at this config; applied the same validated 96-core/2900GB/hugemem config here.
3. **`basalforcings.tf` round-trip bug (found via a real marshalling crash, first run of the
   fixed step).** `tf` is a list of per-depth-layer `(nv+1, 1)` arrays — real numpy arrays
   wherever freshly built (`melt_ismip7_calibration.py`, `finalize_melt_calibration.py`), but
   `save_model`/`load_model`'s round-trip through netCDF/HDF5 for this ragged
   list-of-matrices ("MatArray") field returns each entry as a plain Python list instead.
   Marshalling's scaling step (`yts * data[i][-1, :]`) needs numpy elementwise multiply and
   crashed (`TypeError: can't multiply sequence by non-int of type 'float'`). Never caught
   before because no earlier step reloaded a model with `tf` already populated and then
   re-marshalled it for a fresh solve — `ho_relaxation` is the first one (loads
   `AIS3_melt_gamma_tuning.nc`, whose `tf` was saved by `finalize_melt_calibration.py`).
   `historical_dhdt_tuning` will hit the identical issue loading `AIS3_ho_relaxed.nc` (itself
   saved by `ho_relaxation`) — fixed proactively there too, not waiting to rediscover it.
   Fix: `md.basalforcings.tf = [np.asarray(t, dtype=float) for t in md.basalforcings.tf]`
   right after each `load_model()` call, before any further use.

Resubmitted (job 178024331) with all three fixes; in progress.

## 4e. SSA track: melt calibration + historical tuning built out (`config/ais_0.1_SSA.py`)

Stages 3 (`melt_gamma_tuning_ssa`) and 4 (`historical_dhdt_tuning_ssa`) had been bare
`# TODO, not implemented` scaffolds since the SSA track was first created. Built out both,
mirroring the HO track's now-working equivalents:

- **`melt_gamma_tuning_ssa`**: J1-only (basin-aggregated Gt/yr vs. Paolo/Adusumilli, official
  IMBIE2 basins) rather than the HO track's full J1+J2+J3+Monte Carlo — J1 alone already
  proved robust on HO (§4b: not gamed the way naive per-vertex RMSE was), and this track is
  explicitly meant to be the cheap/simple one. No vertical-layer projection machinery needed
  (unlike HO's `on_base`/`x2d`/`elements2d` — the SSA mesh IS the 2D mesh already,
  `md.mesh.x`/`y`/`elements` are directly usable). `deltaT_basin` held fixed at the published
  prior — same starting-point gap HO had before §4c, not re-derived here; HO's experience
  showed this can matter a lot, so treat this track's melt calibration as a reasonable
  working value, not a final answer, same caveat as HO's §4a stood at initially. Same 9-point
  `gamma_0` grid as HO. Loads `AIS3_SSA_relaxed.nc` (this track's stage order puts relaxation
  *before* melt tuning, opposite of HO — pre-existing design in this file, not changed).
  Saves `AIS3_melt_gamma_tuning_ssa.nc`.
- **`historical_dhdt_tuning_ssa`**: direct mirror of `ais_0.1.py`'s `historical_dhdt_tuning`
  (RACMO SMB 1995-2019, MIPKIT `dhdt_cpom` comparison, same 2019-not-2025 data-coverage
  caveat). Loads `AIS3_melt_gamma_tuning_ssa.nc`, includes the `tf` round-trip fix (§4d)
  proactively since it reloads a model with `tf` already populated.

Both use the plain `cluster` (48 cores/190GB/normal) config, not hugemem — the SSA mesh is
~15x smaller than HO's (no vertical layers) — and both were written with the synchronous
submit-and-wait pattern from the start (`waitonlock` minutes + one `load_only=False` call),
avoiding the "never actually submits" bug §4d found the hard way on HO.

`melt_gamma_tuning_ssa`'s gamma_0 sweep submitted (job 178024976); in progress. Neither SSA
stage has completed a real run yet as of this writing.

## 4f. Race-condition incident #2, and why the "wait for confirmation line" protocol failed

§3.4 already documented one race-condition incident and a fix: don't trust PBS `R` state
alone, poll `qcat <jobid>` for the step's own distinctive first print line before reverting
the shared `steps`/`save` globals. **That exact protocol still produced a duplicate
`ho_friction_inv` submission today (2026-09-02)**, because of a gap the first fix didn't
cover: the confirmation grep was run against `launch_ho_relaxation.pbs.out` — a **file path**,
reused across every submission through that launcher — not the specific job's own live
output. That file still contained `POST-CALIBRATION RELAXATION` from the *first* (failed)
`ho_relaxation` attempt (job 178022903, which printed that header before crashing on the
`tf` marshalling bug). When the second `ho_relaxation` submission (job 178024331) was
queued, the grep matched the **stale leftover text from the first attempt**, not fresh
output from the second — reporting "confirmed" before job 178024331 had actually started
executing. `steps` was reverted to `ho_friction_inv` prematurely; when 178024331 actually
began running (some time later), it read the now-reverted value and launched a duplicate
`ho_friction_inv` attempt (compute job 178024418) straight into `execution/
AIS3_ho_friction_inv/`, overwriting the `.bin`/`.queue`/`.toolkits` files chunk 4's
already-running, legitimate attempt (jobs 178011317/178011517, started hours earlier) had
marshalled from.

**Caught before real damage**: 178024418 was still `Q` (queued, never started running) when
noticed — `qdel`'d before it could write results into the same directory chunk 4's real
compute job depends on. Chunk 4's `.bin` file WAS overwritten on disk, but its compute job
had already loaded its inputs into memory hours earlier and doesn't re-read that file
mid-run, so it should be unaffected (confirmed still healthy in `qstat` afterward). The
stray outer driver (178024331, left polling for a lock file that would never appear since
its compute job was deleted) was also `qdel`'d.

**Root cause of the protocol failure**: confirming "has this process read the current file
state" by grepping a **shared, reused output path** is unreliable whenever that same path
was used by an earlier run of the same step — stale content can produce a false-positive
match. **Fixed protocol**: confirm via `qcat <specific-new-job-id>` instead of grepping any
`.out` file path — `qcat` reads that exact job's own live buffer, scoped to its PBS job ID,
so it cannot be fooled by leftover content from a previous, differently-numbered submission
to the same launcher. Used successfully for the corrected third `ho_relaxation` resubmission
(job 178025567), including an explicit check for the *wrong* header
(`HIGHER-ORDER (HO) FRICTION RE-INVERSION`) so a repeat of this exact failure mode would be
caught immediately rather than silently reported as "confirmed." **Any future `steps`/`save`
toggle on `ais_0.1.py` should use job-ID-scoped `qcat`, never a `.out`/`.err` file path, for
confirmation.**

## 4g. `melt_gamma_tuning_ssa` OOM on first run — stale transient history

First real run of §4e's `melt_gamma_tuning_ssa` OOM-killed (175.89GB/190GB) after
successfully submitting only 1 of 9 gamma_0 candidates. Cause: `AIS3_SSA_relaxed.nc` carries
its full 400-timestep transient history (Vel/Thickness/Surface/Base/MaskOceanLevelset x 400
steps x ~1.6M vertices) — none of it is used by the melt calibration (only
geometry/mask/mesh/materials at the relaxed endpoint matter), but it was carried along and
`copy.deepcopy()`'d fresh every submit-loop iteration, ballooning memory each pass. Tried
bumping the outer driver's memory first (380GB/16cpu) — rejected outright by PBS (`normal`
queue caps at 192GB/node regardless of `ncpus`, same ceiling documented in SS3.2 for the HO
track). Real fix: `md.results.TransientSolution = []` right after loading, before any
copying — strips the unused bulk at the source rather than requesting more memory to hold
data nothing reads. Applied in both the submit loop and the analyze loop's per-candidate
reload. Resubmitted (job 178025724) at the original 190GB/8cpu config.

## 4h. Chunk 4, attempt 5: friction field smoothing (root-cause diagnosis + targeted fix)

Following §3's dfmin_frac (attempt 4) and cf501 (5 probes, all reg weights 1e-6 to 1e-4)
fixes both being **ruled out by direct evidence** (all 6 configurations — chunk 4 itself plus
5 cf501 probes — hung identically at iteration 2's "computing velocities" step, walltime-
killed at 24h), inspected chunk 3's saved `FrictionC` field directly
(`inspect_chunk3_friction_field.py`) rather than guessing another optimizer setting:

- 14.61% of base vertices sit at/near the friction floor (0.05) — **not a bug**, confirmed
  100% floating (vs. 22.1% of the domain overall); floating ice has ~zero basal friction by
  physical construction.
- Neighbour-to-neighbour `C` ratio up to **169,470x**, with 184,906 edges (1.94% of all
  ~9.5M base-element edges) showing >10x jumps. 83.9% of ALL grounded/floating-boundary
  edges have >10x jumps (largely expected — friction genuinely drops near the grounding
  line) — but 48,098 edges (26% of all big jumps) are big jumps **within the same regime**
  (both grounded, or both floating) — genuine roughness/noise, not a physically-expected
  discontinuity.
- Diagnosis: chunk 3's friction field already carries this roughness from prior chunks'
  optimization; any further m1qn3 gradient step (regardless of size or regularisation
  weight — consistent with both dfmin_frac and cf501 failing identically) risks pushing
  already-near-discontinuous values into a state the Newton solve can't resolve. This tracks
  with `inversion_worklog.md` §5.4's prior finding that Schoof was ruled out on physics
  grounds for the Siple Coast trunk for a related grounding-line Coulomb-cap reason.

**Fix (`smooth_chunk3_friction_field.py`)**: targeted, not blanket — median-of-self-and-
same-regime-neighbours smoothing applied ONLY to the 26,327 vertices (1.66%) touching an
anomalous same-regime edge (>10x jump, both grounded or both floating). The physically-
expected grounded/floating discontinuity itself is left untouched, as is the rest of the
(well-converged) domain. 15,061 vertices actually changed value. Original checkpoint backed
up to `AIS3_ho_friction_inv_chunk3_backup_presmoothing.nc` before overwriting (modifies
`md.results.StressbalanceSolution.FrictionC`, the field `ho_friction_inv`'s resume logic
actually reads, not `md.friction.C`).

Resubmitted (job 178174322/178174433) against the smoothed checkpoint; iteration 1 opened at
`f(x)=2.755e6` (essentially unchanged from `f(x)=2.7542e6`, as expected for a targeted,
minor-footprint fix). In progress as of this writing — several hours into iteration 2 with
no result yet, genuinely unresolved.

## 4i. `AIS3_ho_relaxed.nc`: coastal ring artifact traced to unrestored 100m thickness floor

Deeper inspection of the salvaged `AIS3_ho_relaxed.nc` (prompted by a visual spatial plot,
not just the summary percentiles) showed a **solid, saturated ring of strong thinning around
the entire coastal margin** — not scattered noise. `p1`/`p5` of the dH distribution were both
exactly `-99.00m`, a flat plateau rather than organic scatter.

**Traced to source** (`check_ho_relaxed_coastal_ring.py`): `ssa_inverted_solve`
(`ais_0.1.py:1398-1409`) floors thin ice at exactly 100m — a deliberate, documented
numerical-stability measure to keep geometry self-consistent with what
`ssa_friction_inv_reg_lcurve` actually solved the friction inversion against. Legitimate for
the inversion itself. **The bug**: the real pre-floor thickness (tracked locally as
`_H_orig`, used only for Schoof thin-ice velocity pinning at line 1467) is never restored
afterward, and never saved to any downstream file — the floored 100m value gets baked into
`AIS3_inverted.nc`'s geometry and propagates unchanged through every subsequent step (SSA
relaxation, HO thermal steady-state, HO friction re-inversion, melt calibration) all the way
to `ho_relaxation`, the first step to ever enable `masstransport` with real dynamics on this
geometry. `masstransport.min_thickness=1.0` (ISSM's real internal floor, far below the
artificial 100m) then let these cells evolve freely: of the 2,973,450 vertices (12.48% of
the ENTIRE domain) sitting at the artificial floor, **57.2% collapsed to exactly 1.0m within
a single simulated year** — not a physically real signal, a numerical artifact of the
mismatch between an artificial uniform starting thickness and whatever velocity/flux these
cells actually have.

**Fix (`restore_floored_thickness.py`)**: recovers real thickness from `AIS3_param.nc` (the
original parameterized model, before ANY flooring was ever applied — the only place the true
value still exists) via nearest-neighbour (x,y) lookup (safe across any mesh
extraction/reordering that happened in between), patches `AIS3_melt_gamma_tuning.nc`
(`ho_relaxation`'s actual input) in place, recomputing surface/base with the same
hydrostatic-floating/bed-grounded formula `ssa_inverted_solve` itself used. Restored
thickness at the 2.97M previously-floored vertices: min=9.65m, max=100.49m, mean=19.11m —
genuinely thin coastal ice, confirming the diagnosis (real values are far below the
artificial 100m floor, not close to it). No arbitrary re-flooring applied — ISSM's own
`min_thickness=1.0` is the physically appropriate floor from here on. 0 vertices with
thickness <=0 after restoration. Original backed up to
`AIS3_melt_gamma_tuning_prethicknessfix_backup.nc`.

Resubmitted `ho_relaxation` (job 178197570/178199985) against the corrected input; in
progress. **`AIS3_ho_relaxed.nc` as it currently stands (from the pre-fix input) should not
be trusted for `historical_dhdt_tuning` — wait for this rerun.**

**Outcome (2026-09-04/05)**: the corrected-thickness rerun (178197570/178199985) completed
its actual relaxation successfully and printed real diagnostics confirming the fix helped —
mean |dH| dropped from 13.00m (broken thickness) to **6.62m** (corrected thickness), a
genuine reduction in the spurious coastal signal. But the outer driver's `save_model()` call
was OOM-killed again (178.68GB/190GB, same failure mode as the first `ho_relaxed` salvage) —
`launch_ho_relaxation.pbs` was never actually updated to hugemem after that first incident,
only the standalone one-off salvage script was. **Fixed properly this time**: updated
`launch_ho_relaxation.pbs` itself to hugemem/48cpu/1450GB, so this doesn't recur a third
time. Salvaged this run's results from the raw outbin via the existing `salvage_ho_relaxed.py`
(job 179123703); in progress.

## 4j. Chunk 4, attempt 5 (smoothed friction): failed differently — SIGBUS, not a hang

Resubmitted against §4h's smoothed checkpoint. This time did NOT hang indefinitely like every
prior attempt — instead crashed at 17h27m into the 24h budget with `SIGBUS (Bus error)` on
MPI rank 71 (`gadi-hmem-clx-0013`), still at "computing new velocity" (iteration 2, same
stage as every previous failure). No further diagnostic detail available in the outlog.
Genuinely ambiguous: could be a transient hardware/node issue unrelated to the friction field
(SIGBUS can also arise from memory-mapped I/O running out of backing storage, distinct from
the OOM SIGKILL seen elsewhere), or a new failure mode the smoothing fix inadvertently
produced — no way to distinguish from one data point. **Decision (2026-09-16): pause chunk 4,
don't spend more compute on it right now** — 5 consecutive failures (~30,000+ SU total, two
distinct failure signatures) without a confirmed fix; revisit later rather than keep guessing.

## 4k. SSA cascade: same thickness-flooring bug confirmed, full redo not yet committed

`check_ssa_relaxed_coastal_ring.py` confirmed `AIS3_SSA_relaxed.nc` has the **identical**
artificial-100m-floor collapse found in HO (§4i) — 198,230 vertices (12.48%, matching HO's
own 12.48% almost exactly), 57.2% of those collapsing to ~-99m dH within the relaxation.
Since SSA's stage order is `ssa_inverted_solve_budd` → `ssa_relaxation_budd` →
`melt_gamma_tuning_ssa` → deltaT_basin refit (relaxation BEFORE melt tuning, opposite of
HO), the entire SSA melt-calibration chain already completed (9-candidate sweep, full
5-round secant refit, gamma_0=300 result) was built on this corrupted geometry.

Full fix would mean: patch `AIS3_SSA_inverted.nc`, re-run `ssa_relaxation_budd` (~9.5h),
then redo `melt_gamma_tuning_ssa` and the entire deltaT_basin refit chain — a substantial
recompute of work already done. **Decision (2026-09-16): quick screening check first**
(`check_ssa_thickness_fix_melt_impact.py`, job 179123951) rather than committing blind —
checks what fraction of the floored vertices are actually floating (melt only applies to
floating ice; if mostly grounded, the bug may not have meaningfully affected melt
calibration at all) and estimates the draft-depth shift for whichever are floating (draft
depth determines which Zhou TF layer gets sampled, so a real thickness correction can shift
melt at those specific vertices). Result pending as of this writing.

**Screening result: NOT a minor edge case.** 62.1% of the floored vertices (123,196) are
floating — 59.74% of ALL floating area in the domain. Mean draft-depth shift: 89.6m
(artificial) → 12.2m (real), a 77.5m shoaling. Given Antarctic thermal-forcing profiles
typically warm with depth, sampling at the artificial (too-deep) draft plausibly
overestimated melt across the majority of the ice-shelf area. **Decision: committed to the
full cascade** — patch `AIS3_SSA_inverted.nc`, re-run `ssa_relaxation_budd`, redo
`melt_gamma_tuning_ssa` and the full deltaT_basin refit chain.

**Cascade executed (2026-09-16/17)**:
1. `restore_floored_thickness_ssa.py` — same technique as HO's fix, patched
   `AIS3_SSA_inverted.nc` (198,230 vertices restored, mean 19.11m, same real-thickness
   values as HO's own fix — expected, same underlying `AIS3_param.nc` source and largely
   overlapping vertex set).
2. `ssa_relaxation_budd` re-run (job 179131514) — **succeeded cleanly this time** (SSA's
   smaller mesh doesn't hit the save-memory OOM HO's own relaxation did). Diagnostics: mean
   |dH| improved 32.90m → **26.19m**, consistent with the HO pattern (less spurious coastal
   signal with corrected thickness).
3. `melt_gamma_tuning_ssa` re-run (9 candidates) — **result barely moved**: every candidate
   shifted <1% (e.g. gamma_0=5537.7: 834.7→837.3 Gt/yr, J1=16.151→16.023), best still
   gamma_0=5537.7. Despite the dramatic draft-depth shift, the affected vertices are
   shallow-draft/thin coastal ice, which was never a major melt contributor in absolute
   terms — the aggregate basin totals don't notice much even though the underlying physics
   changed substantially at those specific points.
4. `melt_deltaT_sensitivity_test_ssa` re-run — also nearly identical (5347.12 vs 5340.82
   Gt/yr per degC).
5. Full deltaT_basin refit chain (round 0 onward) resubmitted on the corrected geometry
   (job 179230152) — completed. `ROUND`/`STEP` cycled through the full 0→5 sequence again:
   convergence pattern matched the original (pre-fix) run almost exactly at every round
   (g3/g4/g5 converged round 3, g2/g6/g7 round 4, g1/g8 round 5, g0 unconverged at the
   `MAX_ROUNDS=5` cap — same basins, same rounds as before).
6. Final recalibration (`ssa_melt_deltaT_refit_recalibrate.py`, job 179305728) —
   **confirmed `gamma_0=300.0`** (J2=12.624, J1=0.8455) on the corrected geometry, matching
   the original corrupted-geometry result (J2=12.428, J1=0.8642) almost exactly.
   **Definitively closed: the SSA melt calibration conclusion (`gamma_0=300`) is unchanged
   by the thickness-flooring fix.** Both tracks (HO and SSA) now agree on `gamma_0=300`.

## 4l. `historical_dhdt_tuning`: first real run, RACMO shape bug found and fixed

First actual run of this step (loads the now-fixed `AIS3_ho_relaxed.nc`) — matches this
project's consistent pattern of scaffold code finding genuine bugs on first execution.
`points_to_mesh(racmo_x, racmo_y, smb_yr_myr, ...)` crashed: `smbgl` carries a singleton
`height` dimension (`dims: time, height, rlat, rlon`) that survives `sum('time')`, leaving a
3D `(1, rlat, rlon)` array where `lat`/`lon` (hence `racmo_x`/`racmo_y`) are 2D `(rlat,
rlon)` — shape mismatch. Fixed with `.squeeze('height')` before `.to_numpy()`. Also created
`launch_historical_dhdt_tuning.pbs` sized for hugemem/48cpu/1450GB from the start (240
timesteps, more than `ho_relaxation`'s 50 that already needed hugemem for its own save step
— no reason to wait and rediscover that a third time). Resubmitted (job 179231120).

## 4m. `historical_dhdt_tuning`: hard 24h/96-cpu walltime ceiling found; chunked warm-restart implemented

Job 179231120 (the outer driver) hit its own 25h PBS walltime and died (exit -29); its
inner compute job (179249841) independently hit ITS OWN 24h walltime cap (exit -29) after
reaching **iteration 160 of 240** (66.7%) — no checkpoint saved (this step had no
resume/chunking mechanism, unlike `ho_friction_inv`), losing the full ~24h / ~6917 SU of
compute. At the observed rate (~9 min/iteration), the full 240-timestep run needs ~36h.

Initial fix attempt (just raising `md.cluster.time`/`md.settings.waitonlock`/the launcher's
PBS walltime to 48h) turned out to be based on a wrong assumption. Direct `qsub` probing
(dry-run submissions of a trivial script at increasing `walltime`/`ncpus` on `hugemem`)
revealed a **hard, project-level NCI ceiling that cannot be raised by any walltime setting**:

| ncpus (hugemem) | max walltime |
|---|---|
| 48 (1 node) | 48h |
| 96 (2 nodes) | 24h |
| 192 (4 nodes) | 5h |

(hugemem also only accepts node-multiples of 48 cpus — 56/64/72/80 are rejected outright.)
This step's inner solve uses `md.cluster.np = 96`, i.e. exactly the 24h-capped tier — this
is the actual root cause of the original failure, not an under-sized budget. Dropping to 48
cpus would double the cap to 48h but also roughly double per-iteration time (fewer ranks on
the same 23.8M-node mesh), netting out no better.

**Decision (via `AskUserQuestion`, chose "Implement chunked warm-restart"):** mirror
`ho_friction_inv`'s own chunked-resubmission pattern (§ above), but chunked over wall-clock
**time** rather than m1qn3 **iterations**. Implemented in `ais_0.1.py`'s
`historical_dhdt_tuning` block:
- `CHUNK_YEARS = 6.0` (60 timesteps/chunk @ 0.1yr) — ~9h observed per chunk, comfortable
  margin under the 24h/96-cpu cap. 4 chunks total: 1995→2001→2007→2013→2019.
- Checkpoint/resume uses the **same file** (`AIS3_historical_1995_2019.nc`) as both
  in-progress checkpoint and final result — same convention `ho_friction_inv` uses for
  `AIS3_ho_friction_inv.nc`. `md.timestepping.final_time` on the loaded checkpoint IS the
  progress marker (always accurate, since `solve()` only writes results after the mpiexec
  process returns normally — a walltime-killed chunk saves nothing, so a resumed checkpoint
  is always a fully-completed chunk boundary; worst-case loss is now ONE chunk, not the
  whole run).
- Each chunk carries `Thickness`/`Surface`/`Base`/`MaskOceanLevelset` from its last
  timestep forward as the next chunk's initial condition (`md.geometry.*`,
  `md.mask.ocean_levelset`). `ice_levelset` is left untouched (calving isn't modelled, only
  grounding-line migration). `md.transient.requested_outputs` explicitly set to
  `['default', 'Vel', 'Thickness', 'Surface', 'Base', 'MaskOceanLevelset']` (previously
  unset) so these fields are actually available to carry forward.
- Per-chunk cluster budget: `md.cluster.time = 60*14` (14h), `waitonlock = 60*15` (15h) —
  margin under the 24h cap. Outer launcher's own PBS walltime (`launch_historical_dhdt_tuning.pbs`)
  sized to 20h (that job only requests 48 cpus itself, a *separate* PBS submission from the
  96-cpu inner solve, so it sits in the 48h tier — 20h just needs to cover setup + waiting
  out one chunk + save, comfortably under its own ceiling).
- Final chunk (reaching `t=2019.0`) additionally runs the dH/dt-vs-`dhdt_cpom` diagnostic,
  using `AIS3_ho_relaxed.nc`'s geometry as H0 (the true 1995 start) rather than this chunk's
  own `TransientSolution[0]` (which only spans the last chunk's own window).

**Chunk 1 (job 179337124, t=1995→2001): succeeded.** Exit 0, 17h16m walltime used of the
20h budget, 2486.96 SU, checkpoint saved cleanly to `AIS3_historical_1995_2019.nc` (159GB).
Confirms the chunking mechanism works end-to-end.

**Chunk-1 sanity check** (`check_historical_chunk1.py`): first attempt (job 179486566)
crashed on the dH/dt-vs-`dhdt_cpom` comparison — see the `xr_to_mesh` bug below, same root
cause as found in the production code. Resubmitted (job 179494421) after the fix, succeeded
cleanly. Findings:
- Field sanity: clean — zero NaN/Inf across Vel/Thickness/Surface/Base, zero negative
  thickness.
- Grounding-line migration over the 6 years: 130,380 vertices newly floating vs **484,350
  newly grounded** — a real, persistent asymmetry (survived the rerun, not an artifact of
  the crashed attempt). Likely explanation: `AIS3_ho_relaxed.nc` only had 1 year of
  relaxation after the thickness-flooring fix (§4i) was applied, so a lot of geometric
  adjustment may still be working through the system in these early chunk-1 years rather
  than reflecting a genuine historical grounding-line trend. Flagged to watch across chunks
  2-4 — if the asymmetry shrinks, that supports "settling"; if it persists at this scale
  throughout, worth a deeper look.
- dH distribution: median ≈ -0.30m (flat), IQR -8.65 to +8.51m (bulk of the ice sheet
  stable), but wide tails (p0=-1544.68m, p100=+1012.11m) — consistent with a handful of
  vertices still settling, matching the GL-migration finding above.
- Early dH/dt vs `dhdt_cpom` (6-of-24-year rate, directional check only): RMSE=3.758 m/yr,
  mean bias=+0.340 m/yr (sim slightly less negative than obs). Small relative to typical
  dynamic-region dH/dt magnitudes — an encouraging early signal, though not the official
  comparison (that happens once all 24 years complete).

**Bug found and fixed: `xr_to_mesh` crash on `dhdt_cpom` (3 locations).** The dH/dt
diagnostic code (`mipkit['dhdt_cpom'].isel(cpom_dhdt_time=-1)` computed but then the
UN-sliced 3D `mipkit` dataset passed to `xr_to_mesh` anyway) crashes with `"variable
'dhdt_cpom' must be 2D on a rectilinear grid"` — `xr_to_mesh` requires 2D input, `dhdt_cpom`
carries a `cpom_dhdt_time` dimension. This exact broken pattern existed in **three places**:
`ais_0.1.py`'s final-chunk diagnostic (would have crashed chunk 4 right after the full
24-year run finished — checkpoint save happens first, so no compute lost, just the
diagnostic printout), `ais_0.1_SSA.py`'s `historical_dhdt_tuning_ssa`, and
`check_historical_chunk1.py`. Fixed all three: wrap the already-sliced 2D field back into a
one-variable Dataset (`mipkit[['dhdt_cpom']].isel(cpom_dhdt_time=-1)`) before passing it in,
preserving `x1km`/`y1km` coords. Also reordered `historical_dhdt_tuning_ssa` to save its
transient results *before* running the diagnostic (mirroring the HO chunked step's own
save-then-diagnose ordering), so a future diagnostic-only crash there can't lose a completed
run either.

**Chunk 2 (job 179486058, t=2001→2007): submitted, running** (queued ~44h before starting —
hugemem congestion has gotten noticeably worse than chunk 1's queue wait). Chunks 3
(2007→2013) and 4 (2013→2019, final) to follow the same check-then-resubmit pattern; the
shared `steps=['historical_dhdt_tuning']` toggle in `ais_0.1.py` stays set until the full
run completes, then reverts to `['ho_friction_inv']`.

## 4n. SSA `historical_dhdt_tuning_ssa`: found using the wrong (superseded) melt calibration

First attempt (job 179492355) crashed immediately: `AttributeError: 'default' object has no
attribute 'tf'`. Root cause: `melt_gamma_tuning_ssa`'s save/analyze branch reloads its best
candidate straight from `AIS3_SSA_relaxed.nc` and only grafts on the `TransientSolution`
results fetched via `load_only=True` — it never reapplies the ismip6 basalforcings
config (`gamma_0`/`tf`/`delta_t`/`basin_id`) that was actually used to produce those
results, since that config only ever existed on the `_copy.deepcopy` submitted separately
and doesn't round-trip through `save_model`/`load_model`. So `AIS3_melt_gamma_tuning_ssa.nc`
was saved with `basalforcings` still at its raw, unconfigured `default` type. Every other
downstream consumer (`melt_deltaT_sensitivity_test_ssa.py`, `ssa_melt_deltaT_basin_refit.py`)
never hit this because they reconstruct basalforcings from scratch themselves rather than
trusting this file's saved config — `historical_dhdt_tuning_ssa` was the first to assume
otherwise. **Fixed** in `ais_0.1_SSA.py`'s `melt_gamma_tuning_ssa` save branch: re-apply the
full ismip6 config to the winning candidate before saving. Regenerated the file (job
179494110) — succeeded, but revealed a second, more significant problem.

**Second, deeper problem: that "best" candidate is the wrong config entirely.**
`AIS3_melt_gamma_tuning_ssa.nc` only ever held `melt_gamma_tuning_ssa`'s own COARSE 9-point
sweep pick (`gamma_0=5537.7`, published/unrefit `deltaT_basin`) — it was never updated after
`ssa_melt_deltaT_basin_refit.py`'s per-basin secant refit chain converged on the actual
accepted answer, `gamma_0=300` with a per-basin REFIT `deltaT_basin` (confirmed best-by-J2
in `ssa_melt_deltaT_refit_recalibrate.py`, job 179305728, §4k). Neither refit script ever
saved a full model file — they only tracked numeric state (`deltaT_refit_state_ssa.json`)
and pulled raw melt-rate results via `load_only=True` against each candidate/round's own
execution folder, never persisting a genuinely complete, self-consistent "final calibrated"
model anywhere. Had this gone unnoticed, `historical_dhdt_tuning_ssa` would have produced a
historical run using a known-superseded melt configuration.

**Fix**: new script `finalize_ssa_melt_calibration.py` reconstructs the winning candidate's
(i=1, `gamma_0=300`, `final_round=5`, converged=True) full ismip6 basalforcings config from
`deltaT_refit_state_ssa.json`'s `history[-1]['delta_t']` (the refit per-basin array),
mirroring `ssa_melt_deltaT_basin_refit.py`'s own `setup_base_model()` exactly, then pulls
its already-completed solve (`AIS3_ssa_deltaT_refit_g1_r5`) via the same `load_only=True`
technique and saves a genuinely complete model to `AIS3_melt_final_ssa.nc`. Ran successfully
(job 179502636): `gamma_0=300.0`, `delta_t` range `[0.869, 12.259]`°C, floating melt rate
0–13.945 m/yr (mean 0.270). Updated `historical_dhdt_tuning_ssa` to load
`AIS3_melt_final_ssa.nc` instead of `AIS3_melt_gamma_tuning_ssa.nc`. Resubmitted (job
179504719) — hit a third, unrelated bug (below).

Job 179504719 crashed fast: `points_to_mesh` shape mismatch in the RACMO SMB loop — the
exact same `smbgl` singleton-`height`-dimension bug already found and fixed on the HO
track's identical code (§4l), just never ported over to `ais_0.1_SSA.py`'s own copy. Fixed
with the same `.squeeze('height')` before `.to_numpy()`. Resubmitted (job 179562597).

## 4o. HO chunk 2: warm-restart geometry inconsistency found and fixed

Chunk 2's first real attempt (job 179486058) failed fast, before any solve: `"Model
consistency error: base < bed on one or more vertices"`. Checkpoint from chunk 1 was
untouched (the crash was in the pre-solve consistency check, before `save_model` would ever
run again) — no compute or progress lost.

Root cause: §4m's chunk-resume logic carried forward the raw solver-output
`Surface`/`Base`/`MaskOceanLevelset` fields verbatim as the next chunk's initial condition.
At vertices whose grounding state flipped during the chunk (484,350 went floating→grounded
in chunk 1 alone, per the chunk-1 sanity check above), the solver's own output `Base` can
still reflect the OLD floating-basis draft depth rather than snapping exactly to bed at the
moment the vertex is reclassified as grounded — a self-consistency gap in ISSM's raw output
itself, not safe to trust verbatim across a chunk boundary the way `Thickness` and
`MaskOceanLevelset` are.

**Fix**: still carry `Thickness` and `MaskOceanLevelset` forward as the real prognostic
state, but RECOMPUTE `Base`/`Surface` from them via the same hydrostatic-floating/bed-
grounded formula used throughout this pipeline (`restore_floored_thickness.py`'s
`base = max(-H*rho_ice/rho_water, bed)` for floating, `base = bed` for grounded,
`surface = base + H`), which guarantees `base >= bed` everywhere by construction rather than
trusting solver output. Resubmitted chunk 2 (job 179562596) — **succeeded**: exit 0, "Chunk
complete (t=2007.0 of 2019.0)", checkpoint saved.

**SSA historical run also succeeded** (job 179562597, after the RACMO SMB fix above) — this
one completes in a single shot (no chunking, SSA mesh is ~15x smaller and fits comfortably
under `normal`'s walltime tiers). Saved to `AIS3_historical_1995_2019_SSA.nc`. **Official
result: grounded dH/dt mismatch RMSE vs `dhdt_cpom` = 1.402 m/yr.** SSA track's historical
validation is complete.

## 4p. HO chunk 2's near-miss: `waitonlock` had almost no real margin

Chunk 2's outer driver (job 179562596) used 19h49m of its 20h PBS walltime — cutting it
extremely close. Traced to `md.settings.waitonlock = 60*15` (900 min): the "waiting for lock
file" counter it bounds includes the inner solve job's PBS QUEUE wait, not just its actual
compute time, and chunk 2 needed 899 of those 900 minutes to finish — one minute from the
outer driver giving up on an inner job that was about to complete successfully. hugemem
queue congestion has ranged from a few hours to ~44h across this session's chunks (§4m); 15h
of combined queue-wait+compute margin doesn't cover that variance.

**Fix**: bumped `waitonlock` to `60*40` (40h) and the launcher's own PBS walltime to 45h
(still under the 48h/48-cpu hard ceiling from §4m), leaving ~5h for the outer's own
setup/load-results/save-checkpoint overhead beyond the 40h wait budget. Chunk 3 (job
179637209) had already been submitted with the old, tighter settings before this was caught
— since it was still queued (zero compute sunk), cancelled and resubmitted (job 179637270)
with the fixed walltime rather than risk a repeat or an outright failure.

## 4q. SSA historical run: independent full validation (`check_historical_ssa.py`)

Deeper sanity check of `AIS3_historical_1995_2019_SSA.nc` (job 179637343), mirroring
`check_historical_chunk1.py`'s approach, adapted for SSA's 2D mesh and single-shot (not
chunked) 24-year run. Confirms the production script's result independently rather than
just trusting its inline printout.

- **Field sanity**: clean — zero NaN/Inf across Thickness/Vel/Surface/Base, zero negative
  thickness. All fields were actually present in `TransientSolution` despite
  `historical_dhdt_tuning_ssa` never explicitly setting `requested_outputs` — ISSM's
  defaults covered it here (unlike the HO chunked step, which does set it explicitly since
  it needs specific fields to carry state across chunk boundaries).
- **Grounding-line migration over the full 24 years**: 13,017 newly floating vs 40,128
  newly grounded — same directional asymmetry as HO chunk 1 (§4m sanity check), proportionally
  similar scale relative to SSA's ~1.59M-vertex mesh. Consistent with genuine
  settling/adjustment near the grounding zone rather than something SSA-specific or a bug.
- **dH distribution**: median exactly 0.00m over 24 years (remarkably stable), IQR -11.99 to
  +14.85m. Tails still wide (p0=-1436.23m, p100=+864.78m) — matches the "quiet interior,
  noisy coastal margin" pattern seen throughout this pipeline.
- **dH/dt vs MIPKIT's `dhdt_cpom`, independently recomputed**: RMSE=1.402 m/yr, bias
  (sim-obs)=+0.236 m/yr — matches the production script's inline result exactly.
- **Spatial plot** (`historical_ssa_dH_spatial.png`): flat/white interior, speckled
  red/blue coastal margin concentrated on West Antarctica and a few East Antarctic spots,
  two standout strong-thickening hotspots (Antarctic Peninsula tip, one East Antarctic
  coastal point). Same qualitative pattern as HO chunk 1's plot, just accumulated over the
  full 24 years (mean|dH|=25.64m vs chunk 1's 16.71m over 6 years) — no runaway divergence,
  a reassuring sign of stability over the full historical period.

**SSA track's `historical_dhdt_tuning_ssa` is fully complete and independently verified.**
RMSE=1.402 m/yr is a genuinely good result: SSA's simplified physics reproduces observed
elevation-change rates to within 1.4 m/yr over a real 24-year forced transient, using a
friction/melt calibration fit independently (to velocity and basal melt, never to dH/dt
directly) — this is the out-of-sample validation this whole step exists to provide (see the
step's own purpose, discussed in conversation: friction inversion matches a velocity
snapshot, melt calibration matches basal melt rates, neither directly tests whether the
model's *evolution over time* looks like reality; this does).

## 4r. SSA historical run: time-series check corrects §4q's "good validation" framing

§4q called RMSE=1.402 m/yr "a genuinely good validation result". That was overstated. Two
problems with the number, then a second correction to my own first correction:

**Problem 1 — the metric compares different quantities.** The production diagnostic is a
per-vertex spatial RMSE of the simulated 24-year MEAN rate `(H_2019-H_1995)/24` against only
`dhdt_cpom`'s LAST (2019) slice (`.isel(cpom_dhdt_time=-1)`), not a temporal comparison. It is
dominated by local noise of order +-1 m/yr, so it cannot see (or rule out) a domain-wide bias.
`dhdt_cpom` has 27 annual slices (1993-2019) that evolve smoothly (looks like a multi-year
trend product, not independent annual snapshots).

**Problem 2 — first time series (`check_ssa_historical_timeseries.py`) was misleading.**
Unweighted vertex mean over all grounded vertices (sim) vs. over obs-finite grounded vertices
(obs): sim +0.30 -> +0.06 m/yr, obs -0.06 -> -0.20 m/yr, "wrong sign", RMSE 0.2685, corr 0.722.
I read this as the model gaining mass while observations lose it, and hypothesised drift toward
equilibrium from an initial imbalance. **That conclusion was an artifact** of (a) an unweighted
vertex mean on a mesh that is much finer at the margins/fast flow, so it overweights those
regions, and (b) mismatched coverage between the sim and obs averages.

**Redo (`check_ssa_historical_timeseries_areaweighted.py`, job 179716441):** area-weighted
(vertex control-volume area), with sim and obs averaged over the SAME vertices each year
(grounded AND obs finite). Grounded = 77.4% of vertices but 76.5% of mesh area (12.02 Mkm^2).
- Area-weighted mean 1996-2019: sim +0.0025 m/yr, obs -0.0017 m/yr; mean bias +0.0042 m/yr;
  RMSE 0.0145 m/yr. The sign disagreement is gone: both are ~0 in the area mean.
- Unweighted (same common mask) RMSE is 0.1703 m/yr — 12x larger — so the earlier large
  bias was mostly a weighting effect.
- **Correlation = -0.134**: the model has no skill at reproducing the observed interannual
  variability. Sim starts negative (-0.024 in 1996) and drifts positive; obs wanders around zero
  (-0.012..+0.013). Notable sim spike in 2016 (+0.024, ~+258 Gt/yr integrated) that obs
  (+0.005) doesn't show; 2016 RACMO mesh-mean SMB was also the anomalous high (0.640 vs ~0.53).
  The drift-to-equilibrium hypothesis from the first pass is NOT supported and is withdrawn.
- Caveat on coverage: the common mask grows from ~7.8 Mkm^2 (1996-2010) to ~11.5 Mkm^2
  (2011+), i.e. CPOM coverage expands (CryoSat-2 era) — the yearly means/Gt-yr values are over
  different areas across the record, so the time-series shape is partly a coverage artifact.
  Integrated Gt/yr (sim +32.0, obs -17.1 Gt/yr mean) uses non-firn-corrected obs and is only
  indicative, not comparable to IMBIE totals.

**Where SSA historical validation actually stands:** area-mean dH/dt magnitude is consistent with
observations (both ~0, well within obs noise), which is a real if modest positive. It does NOT
demonstrate skill in interannual variability (corr -0.13). Not yet done: fixed-area (constant
coverage) comparison, comparison to a 1995-2019 mean/trend product instead of annual slices,
per-basin comparison, and a firn/SMB-response-aware treatment of the altimetry. The same caveat
(spatial RMSE vs only the last dhdt_cpom slice) applies to the HO number when chunk 4 finishes.

## 4s. SSA historical: fixed-coverage, per-basin comparison (follow-up to 4r)

`check_ssa_historical_fixedcov_basins.py` (job 179718665). Fixes 4r's coverage problem: every
year uses the same vertices — grounded AND `dhdt_cpom` finite in all 24 years (1996-2019):
599,285 vertices, 7.56 Mkm^2, 62.9% of grounded area (the rest is mostly areas without early
altimetry coverage). All means area-weighted.

- Whole domain: sim mean -0.0086 m/yr, obs -0.0026 m/yr (-59.8 vs -18.2 Gt/yr over the fixed
  area); bias -0.0060, RMSE 0.0109, time corr +0.136. On a fixed area both are NEGATIVE; the
  model thins ~3x more than CPOM in the mean.
- Trends (fit to the printed series): sim +0.00034 m/yr per yr, obs -0.00069. The opposite
  trends are NOT a coverage artifact — they survive fixed coverage. But the sim trend excluding
  1996-1999 is -0.00007 (flat): the sim's upward trend is entirely its first ~4 years.
- Spatial pattern (area-weighted correlation across vertices): 1996-2019 mean-dH/dt map 0.166,
  trend map -0.005. Weak skill in the mean pattern, none in where things are changing.
- Per basin (IMBIE2 numbering; no names in the source files, locations from basin centroids):
  13/16 basins same sign as obs. Opposite sign: basins 0 (~12E), 6 (~179W), 11 (~77W).
  Nearly all the mass loss is in basins 8 (~139W), 9 (~115W, Amundsen Sea sector) and 10
  (~94W): sim -20.9/-84.9/-6.1 Gt/yr vs obs -11.1/-65.0/-0.9 — the model OVER-thins them, and
  their per-basin time correlations are ~0 (-0.07, -0.03, 0.43). The large East Antarctic
  basins are small in both, with per-basin corr 0.4-0.8 (consistent with 4t: SMB-driven).
- So the near-zero whole-domain mean in 4r did hide a regional error, but not an opposite-sign
  cancellation: West Antarctic over-thinning offset by coverage-dependent interior thickening.

## 4t. Why the SSA sim trend opposes CPOM: SMB correlation test

`check_ssa_historical_smb_correlation.py` (job 179720452). Correlates area-weighted grounded
sim dH/dt with the area-weighted RACMO SMB the run was actually forced with (read from the saved
`md.smb.mass_balance`), and splits sim dH/dt = SMB + residual (residual = -div(flux), i.e.
dynamics). Note: this script averages sim over ALL grounded vertices but obs over obs-finite
vertices (different areas), so its sim-vs-obs line (+0.279) is not comparable to 4s; the
sim-vs-SMB numbers use the same area on both sides and are the point of the test.

- Sim vs SMB (average of year and previous year, which the H(yr)-H(yr-1) interval spans):
  levels r=0.846, first differences r=0.972. Residual std 0.0039 vs SMB std 0.0060. The sim's
  year-to-year signal is almost entirely surface forcing; with near-steady dynamics this is
  close to built in (dH/dt = SMB - div(flux)).
- Obs vs SMB: levels r=0.443, first differences r=0.388 — altimetry has an SMB/firn signal
  (it is not firn-corrected), but much more besides.
- Trend decomposition: sim +0.00036 = residual +0.00043 + SMB -0.00007. The residual (dynamic
  loss) goes -0.183 (1996) -> -0.172 (2000) -> -0.167 (2019): almost all the change is in the
  first ~4 years.

Hypotheses from the conversation, assessed (24 autocorrelated points — no causal claims):
- **Start-up transient — supported.** The sim's upward trend is entirely dynamic, concentrated
  in 1996-1999; excluding those years the fixed-coverage sim trend is flat (4s). With static
  ocean forcing and trendless SMB, a changing dynamic term can only be internal adjustment.
- **SMB-driven variability — supported (for the sim).** r=0.97 in first differences.
- **Missing time-varying ocean forcing — consistent, not tested.** After the transient the model
  is flat while CPOM trends negative, and the model has no mechanism for strengthening dynamic
  thinning (static TF climatology). The basin-9/8/10 zero correlations point the same way.
  A test needs per-basin trends, or a historical run with time-varying TF.
- **CPOM not firn-corrected — untested.** Obs-vs-SMB r=0.44 shows altimetry carries SMB/firn
  signal; whether firn trends bias the obs trend is not assessed here.

## 4u. HO historical run complete (all 4 chunks)

Chunk 3 (job 179637270, 2007->2013): exit 0, 14h56m of 45h. Chunk 4 (job 179716149,
2013->2019, final): exit 0, 16h08m of 45h, 2324 SU. The 45h walltime / 40h waitonlock fix (4p)
held for both. Full 1995-2019 HO run is in `AIS3_historical_1995_2019.nc`; `ais_0.1.py` `steps`
reverted to `['ho_friction_inv']`.

Production diagnostic: grounded dH/dt mismatch RMSE vs dhdt_cpom = 1.768 m/yr (SSA: 1.402).
Same caveat as 4r: per-vertex spatial RMSE of the 24-year mean rate against only dhdt_cpom's
2019 slice, dominated by local noise; not a temporal validation and cannot see a domain-wide
bias. The area-weighted/fixed-coverage/per-basin time-series checks done for SSA (4r-4t) have
not been done for HO: the chunked run only keeps the last chunk's per-step history, so yearly
dH/dt would come from chunk-boundary snapshots (1995/2001/2007/2013/2019) plus chunk 4's
steps, unless the chunks are rerun with per-year snapshots saved.

## 4v. HO per-year history mostly lost; chunk-1 rerun + partial HO analysis

Correction to 4u: the chunk-boundary thicknesses for 2001 and 2007 do NOT survive. Every chunk
saved to the same checkpoint (`AIS3_historical_1995_2019.nc`) and ran in the same execution
directory, so each overwrote the last. What survives: H(1995) (`AIS3_ho_relaxed.nc`), H(2013)
(final checkpoint's `md.geometry`, chunk 4's initial condition) and chunk 4's 60 steps
(2013.1-2019.0). Lesson for any future chunked run: keep a per-chunk copy of the history.

Two follow-ups submitted:
- `check_ho_historical_timeseries.py` (job 179851729, hugemem): HO vs SSA vs CPOM under
  identical definitions — 1995-2013 mean rate and yearly 2014-2019, fixed-coverage grounded
  mask, area-weighted, whole domain and per ISMIP7 basin. HO has 15x SSA's vertex count
  (23,833,290 = 15 x 1,588,886), i.e. apparently the SSA mesh extruded to 15 layers; the
  script checks the base layer matches the SSA mesh exactly before comparing.
- Chunk-1 rerun (job 179851673, `launch_historical_chunk1_rerun.pbs`): 1995-2001 again, into
  `AIS3_historical_1995_2019_chunk1rerun.nc` with its full 0.1 yr history, to look for the HO
  start-up transient (SSA's was in 1996-1999, see 4t). The launcher refuses to run if that file
  already exists, so it cannot continue into chunk 2.

`ais_0.1.py` change: `steps` can now be overridden per job via `qsub -v AIS3_STEPS=...`, and
`AIS3_HIST_TAG` suffixes the historical checkpoint and execution name. This avoids holding an
edit to the shared `steps` through a multi-day hugemem queue wait (the race hazard that the
qcat-confirmation protocol existed for). Default behaviour with neither variable set is unchanged.

## 4w. HO vs SSA vs CPOM with the surviving HO output (`check_ho_historical_timeseries.py`)

Job 179851729. Checks before comparing: the HO base layer is identical to the SSA mesh
(23,833,290 = 15 x 1,588,886 vertices, same coordinates and order), so both models use the same
vertices, areas and fixed-coverage mask (599,285 vertices, 7.563 Mkm^2, as in 4s). The final
HO checkpoint's `md.geometry.thickness` is the 2013 state: H(2013.1) minus it is +0.021 m mean
(p99 0.65 m), i.e. ~0.1 yr of SMB. Grounded masks (SSA final vs HO 2013) agree on 96.35% of
vertices. Quantities: 1995-2013 mean rate (H2013-H1995)/18 vs mean of CPOM's 1996-2013 slices;
yearly 2014-2019. Area-weighted, rho_ice 917.

Whole domain (Gt/yr over the fixed area):

| period | HO | SSA | CPOM |
|---|---|---|---|
| 1995-2013 mean | -150.6 | -64.1 | -5.3 |
| 2014-2019 mean | -111.7 | -47.9 | -56.9 |

- Both models thin LESS in the late period than the early one, while CPOM thins MORE (-5 ->
  -57 Gt/yr). HO has the same qualitative behaviour as SSA (4s/4t).
- HO is systematically more negative than SSA: by -0.0125 m/yr over 1995-2013 and by an almost
  constant -0.0088 to -0.0096 m/yr in every year 2014-2019. HO and SSA year-to-year variability
  is otherwise the same (same RACMO forcing, variability SMB-driven per 4t); the difference is
  a steady, slowly shrinking offset, i.e. extra dynamic thinning in HO.

Per ISMIP7 basin (Gt/yr; Mouginot equivalents from the basin mapping):
- Mass-loss basins 8/9/10 (= Mouginot 10/11/12): basin 9 early HO -98.9, SSA -88.8, obs -54.3;
  late HO -82.5, SSA -73.2, obs -97.1. Basin 8 early -24.6/-22.5/-9.0, late -18.0/-15.9/-17.4.
  Both models over-thin early and under-thin late: they relax while CPOM accelerates. HO is
  ~10% more negative than SSA here. Basin 10: late HO +0.5 vs SSA -4.3 vs obs -4.7 (HO wrong
  sign late; 2016-2017 HO spike +0.11 m/yr).
- Opposite-sign basins from 4s (0/6/11 = Mouginot 1/7/13): basin 0 obs thickens (+8.9 early,
  +20.5 late), HO -6.4/-7.7 and SSA -2.4/-3.6 both thin; basin 11 early obs +2.7, HO -5.3,
  SSA -1.7; basin 6 late all positive but models low (HO +1.4, SSA +1.9, obs +6.8).
- Where SSA matched CPOM, HO often does not: basin 14 (= Mouginot 16+17) early HO +4.4 vs SSA
  +28.6 vs obs +26.4; basin 15 early HO -5.4 vs SSA +3.3 vs obs +6.7 (wrong sign); basin 2 early
  HO -7.0 vs SSA +3.9 vs obs +2.2 (wrong sign). HO's extra thinning is spread across East
  Antarctica too, not only West Antarctica.

Candidate reasons for HO's extra dynamic thinning (NOT tested): HO had only ~1 yr of relaxation
(ho_relaxation) vs SSA's 20 yr, and HO's friction inversion stopped at chunk 3 (grounded velocity
RMSE ~94-99 vs SSA ~60 m/yr), either of which could leave HO further from balance. The chunk-1
rerun (job 179851673) will show whether HO has a larger 1996-2001 start-up transient than SSA.
The 6 yearly 2014-2019 points are too few for correlations; levels only.

## 4x. HO chunk-1 rerun: reproduces the original; HO start-up transient (1996-2001)

The chunk-1 rerun (job 179851673, `launch_historical_chunk1_rerun.pbs`, AIS3_HIST_TAG=_chunk1rerun)
succeeded: exit 0, 13h17m, 1913 SU, `AIS3_historical_1995_2019_chunk1rerun.nc` written, the main
`AIS3_historical_1995_2019.nc` untouched (mtime still 2026-09-25 03:14). Analysis:
`check_ho_chunk1_transient.py` (job 180009965).

- **Reproduces the original chunk 1 exactly**: every dH percentile identical (p0 -1544.68 ... p100
  1012.11 m), mean|dH| 16.71 m, newly floating 130,380, newly grounded 484,350.
- HO and SSA start the historical run from the same thickness: base-layer 1995 difference is
  exactly 0.000 m. First 0.1 yr step in HO changes thickness by -0.74 m mean (p99 14.6 m, full 3D)
  vs +0.021 m for chunk 4's first step -- a large immediate adjustment (see 4y).
- Whole domain, fixed coverage, area-weighted, 1996->2001 (m/yr): HO -0.048 -> -0.021, SSA -0.024
  -> -0.007, CPOM -0.005 -> +0.010. Mean 1996-2001: HO -222, SSA -104, CPOM +33 Gt/yr.
- Dynamic residual (dH/dt - SMB): HO -0.208 -> -0.187 (change +0.021), SSA -0.185 -> -0.173
  (+0.012). Both have a start-up transient; HO's is ~1.7x larger and HO stays ~0.014 m/yr more
  negative at 2001. Since both start from identical geometry, the HO-SSA gap is NOT relaxation
  length (the 4w hypothesis is withdrawn, see 4y) -- it comes from the models themselves
  (HO stress balance, HO friction field from the unconverged chunk-3 inversion, ...), untested.
- Transient is concentrated in West Antarctica: residual change 1996->2001 HO/SSA basin 8 +0.11/
  +0.09, basin 9 +0.07/+0.04, basin 10 +0.15/+0.09 m/yr. Basin 9 dH/dt 1996 HO -0.43, SSA -0.35,
  CPOM -0.09. Basin 14 (= Mouginot 16+17): HO -2.9, SSA +26.7, CPOM +16.1 Gt/yr (HO residual
  -0.14 vs SSA -0.107 -- HO's extra dynamic loss also in East Antarctica). Basin 13 (tiny):
  HO residual change +0.23.

## 4y. BUG (HO only): relaxation end states never propagated to downstream steps

> **Correction (2026-09-28):** for SSA, starting downstream steps from the inverted rather than
> the relaxed geometry is by design (user). The SSA consequences listed below are withdrawn;
> only the HO part is a bug. See the correction note after §4z.

Neither relaxation step writes its relaxed state into the saved geometry. `ho_relaxation`
(`ais_0.1.py`) and `ssa_relaxation_budd` (`ais_0.1_SSA.py`) call `solve()`, which fills
`md.results` only; `md.geometry`/`md.mask` stay the PRE-relaxation input and `save_model` writes
that. (The relaxation diagnostics themselves compute `TransientSolution[-1].Thickness -
md.geometry.thickness`, i.e. treat geometry as the start state.)

Confirmed directly for SSA (`check_relaxed_geometry_propagation.py`, job 180016250):
`md.geometry.thickness` of AIS3_SSA_relaxed, AIS3_melt_final_ssa and AIS3_historical_1995_2019_SSA
are all bit-identical to AIS3_SSA_inverted (max|d| 0.0000 m) and differ from the 20-yr
relaxation's end state by mean|d| 26.19 m, max 1522.78 m. For HO: code path is identical, and HO's
historical 1995 geometry equals SSA's (4x, diff 0.000 m), so HO's 1-yr relaxation was not used
either.

Consequences:
- The start-up transients in both historical runs (4t, 4x) are at least partly the model relaxing
  at the start of the historical run from an unrelaxed state.
- 4w's "HO 1 yr vs SSA 20 yr relaxation" explanation is moot: neither relaxation reached the run.
- SSA melt calibration (melt_gamma_tuning_ssa, deltaT refit, finalize -> gamma_0=300) was done on
  unrelaxed geometry; the relaxed geometry differs by up to ~1.5 km in places, so ice-shelf drafts
  and hence basin melt could change. HO's melt calibration was on pre-relaxation geometry by
  design (melt tuning precedes relaxation on HO), so that part is unaffected.

Proposed fix (NOT applied, awaiting decision): after `solve()` in both relaxation steps, copy
`TransientSolution[-1]` Thickness and MaskOceanLevelset into `md.geometry.thickness` /
`md.mask.ocean_levelset`, recompute base/surface with the hydrostatic/bed formula (as the chunk-
resume fix in 4o does), and ideally set `md.initialization` velocities from the last step, before
`save_model`. The existing relaxation outputs already contain their end states, so the relaxations
need not be rerun: a patch script can rewrite AIS3_SSA_relaxed.nc / AIS3_ho_relaxed.nc geometry.
Rerun plan: SSA -- melt_gamma_tuning_ssa, deltaT refit + recalibrate, finalize, historical
(historical alone ~6 h / ~530 SU; the melt chain is many short solves). HO -- historical 4 chunks
(~13-17 h and ~2,000-2,500 SU each, ~9-10k SU plus queue), saving a per-chunk copy this time.
Cheaper partial alternative: discard the first N years in comparisons (spin-up); that tidies the
dH/dt comparison but does not fix the SSA melt calibration.

**4y addendum -- HO direct check** (`check_ho_relaxed_geometry_propagation.py`, job 180019992):
confirmed for HO. `md.geometry.thickness` (full 3D) of AIS3_ho_relaxed.nc and of the historical
run's 1995 start (AIS3_historical_1995_2019_chunk1rerun.nc) are bit-identical to the relaxation
input AIS3_melt_gamma_tuning.nc (max|d| 0.0000 m), and differ from the relaxation's own end state
(1 yr, 50 saved steps) by mean|d| 6.623 m, max 1020.34 m. Grounded/floating mask: 100% equal to
the input, 98.911% equal to the relaxation end (~1.1% of vertices changed state during the
relaxation and that change was also lost). HO's 1-yr relaxation never reached the historical run.

## 4z. Relaxation fix: rerun the HO relaxation with real forcing and propagate the end state

> **Correction (2026-09-28):** this was first set up for both tracks; the SSA part was reverted
> before running (job cancelled while queued, files and code restored). Only the HO rerun went
> ahead. See the correction note and result below.

Decision (user, 2026-09-28): full fix, both tracks, no old files deleted.

Found while preparing the patch: the old relaxations ran with unrealistic forcing. SSA (20 yr):
`smb.mass_balance = 0` AND floating/grounded melt = 0, i.e. grounded ice lost mass with no
accumulation and ice shelves grew with no melt for 20 years. HO (1 yr): SMB = 0 (inherited from
the melt-tuning model), calibrated ocean melt on. Propagating those end states as-is would have
replaced "unrelaxed start" with "start shaped by zero forcing", so the user chose to rerun both
relaxations with real forcing instead.

Code changes:
- New helpers (ais_0.1.py; copies in ais_0.1_SSA.py): `racmo_mean_smb(md, y0, y1)` (RACMO
  smbgl time-mean, same unit handling as the historical runs), `adopt_final_state(md)` (copies
  TransientSolution[-1] Thickness + MaskOceanLevelset into md.geometry/md.mask, recomputes
  base/surface hydrostatically, sets md.initialization velocities when saved) and, HO only,
  `sync_mesh_z(md)` (recomputes the extruded mesh's vertex z from base + sigma*H, keeping each
  column's layer spacing; prints the prior z-vs-geometry mismatch).
- `ho_relaxation`: SMB = RACMO 1979-1994 mean; `sync_mesh_z` before solve; `adopt_final_state` +
  `sync_mesh_z` after solve, before save; waitonlock 40 h; launcher walltime 25 -> 45 h.
- `ssa_relaxation_budd`: SMB = RACMO 1979-1994 mean; ocean melt = ismip6 config from
  AIS3_melt_final_ssa.nc (gamma_0=300 + refit deltaT, i.e. the current calibration, itself fitted
  on unrelaxed geometry -- the melt chain is recalibrated afterwards); `adopt_final_state` after
  solve; waitonlock 40 h; launcher walltime 25 -> 45 h.
- `historical_dhdt_tuning`: `sync_mesh_z` at fresh start and after chunk resume (z was never
  updated when the resume changed geometry), and a per-chunk base-layer history file
  `AIS3_historical_1995_2019{tag}_history_{t0}-{t1}.npz` (times, thickness, ocean_levelset,
  initial_thickness) so per-step history survives the checkpoint being overwritten (4v).
- Possible second latent issue: `restore_floored_thickness.py` changed HO thickness/base/surface
  without updating `mesh.z`; `sync_mesh_z` prints the mismatch at the start of the HO relaxation,
  which will show whether earlier HO runs started with stale z for those vertices.

Archived (renamed with `mv -n`, not deleted): models/AIS3_SSA_relaxed.nc ->
AIS3_SSA_relaxed_prerelaxfix.nc, execution_SSA/AIS3_SSA_relaxed -> *_prerelaxfix,
models/AIS3_ho_relaxed.nc -> AIS3_ho_relaxed_prerelaxfix.nc, execution/AIS3_ho_relaxed ->
*_prerelaxfix. Later steps will archive their own outputs the same way before overwriting.

Submitted: SSA relaxation job 180027936 (ais_0.1_SSA.py steps=['ssa_relaxation_budd']); HO
relaxation job 180027937 (`qsub -v AIS3_STEPS=ho_relaxation launch_ho_relaxation.pbs`).
Then: SSA melt chain (gamma sweep, +1C sensitivity, deltaT refit rounds, recalibrate, finalize)
and SSA historical; HO historical with AIS3_HIST_TAG=_relaxfix (new files; the finished run
stays untouched).

**4y/4z correction (user, 2026-09-28): SSA not using its relaxed state is BY DESIGN.** The
unrelaxed-start finding is only a problem for HO. For SSA, the melt calibration (gamma_0=300)
and the historical run starting from the inverted geometry are intended and stand as they are;
the "SSA melt calibration on unrelaxed geometry" consequence in 4y is withdrawn. (The user's
earlier "the SSA stuff was fine right just the HO one" meant this; I misread it as a question.)
SSA side reverted: SSA relaxation job 180027936 cancelled while still queued (no compute used);
AIS3_SSA_relaxed.nc and execution_SSA/AIS3_SSA_relaxed renamed back from _prerelaxfix;
ais_0.1_SSA.py relaxation code, helpers, waitonlock, steps toggle and launch_ais_0.1_SSA.pbs
walltime restored to their pre-4z state. The HO relaxation rerun (job 180027937) continues.

## 5a. Why HO thins more than SSA: diagnosis from existing output

**Stale HO vertical mesh (found by the HO relaxation rerun's `sync_mesh_z`, job 180027937):**
"mesh.z vs geometry before sync: base max 80.987 m (123,133 columns off by >1 cm), surface max
90.000 m (198,014 columns)". 198,014 columns ~ the ~198k vertices floored at 100 m and later
restored by `restore_floored_thickness.py`, which edited thickness/base/surface but never
`mesh.z`; max 90 m = 100 m floor minus ~10 m restored. Every HO run since that restoration (old
relaxation, all 4 historical chunks) started with stale vertical coordinates in ~12.5% of
columns (thin coastal ice). Likely part of HO's large first-step change (-0.74 m mean in 0.1 yr);
it is confined to the margins, so it cannot explain the interior basins below. Fixed going
forward by `sync_mesh_z` (4z).

**Speed / rheology / first-year dynamic loss** (`check_ho_vs_ssa_dynamics.py`, job 180030381;
grounded ice with observed speed, 12.02 Mkm^2; speeds at t=1995.1; HO = chunk-1 rerun, SSA =
historical run; identical 1995 thickness, max|d| 0.0000 m):
- Area-weighted mean speed: observed 22.5, SSA 21.8, HO depth-avg 24.6, HO surface 25.0, HO base
  22.5 m/yr. Median vertex ratios: HO surface/observed 1.11, SSA/observed 0.99, HO depth-avg/SSA
  1.15 (1.09-1.64 in every basin). SSA matches observed speed; HO flows ~10-15% too fast almost
  everywhere.
- First-year dynamic loss (dH/dt - SMB, integrated): HO -2300 vs SSA -2035 Gt/yr (+13%),
  proportional to the speed excess. HO's extra thinning is HO flowing too fast.
- HO flow is mostly sliding at the start: median HO base/surface speed 0.93.
- Rheology: HO depth-averaged rheology_B is lower than SSA's (median ratio 0.939 by vertex, but
  the map shows ~0.7-0.8 across the East Antarctic interior and much of West Antarctica; basin
  medians b7 0.79, b14 0.84, b2 0.86, b15 0.91). Basins where B matches (b4 1.00, b5 1.00) have
  HO ~= SSA dynamic loss (-243 vs -253, -129 vs -128 Gt/yr); the softest basins carry the largest
  HO excess (b14 -284 vs -218, b2 -93 vs -64). Strong spatial association.
- Map (models/ho_vs_ssa_dynamics_maps.png): per-vertex HO-SSA dynamic difference is largest at
  the margins (mixed sign, where the stale-z columns are); HO/SSA speed ratio >1 broadly across
  West Antarctica, the Peninsula and coastal East Antarctica; HO/SSA B ratio ~0.7-0.8 over the
  interior, ~1 near the margins.

Assessment (not yet tested causally):
- Supported: HO's excess thinning comes from HO flowing ~10-15% faster than observed; HO's
  depth-averaged ice is substantially softer than SSA's, and the softness lines up with where
  HO's excess is.
- Entangled: the HO friction inversion was run with this softer B and should have compensated
  with stronger friction; it stopped unconverged (chunk 3, RMSE ~94-99 m/yr) with surface speed
  ~11% too fast. Softer rheology and the unconverged friction inversion are two sides of the same
  outcome; this output cannot split them.
- Ruled out as the main cause: stale mesh.z (margins only).
- Not checked: where SSA's grounded rheology_B comes from vs HO's (HO's is from its own 3D
  thermal steady state).
- Tests that would settle it: (1) one HO diagnostic stress-balance solve with SSA's B replicated
  vertically -> if HO speed drops to ~observed, rheology is the driver; (2) continue the HO
  friction inversion (chunk 4, paused) to convergence with the current B -> if speed then matches
  observations, the start is fixed but the softer ice may still evolve faster.
- Recommendation: settle this before the ~9-10k SU HO historical rerun, which would otherwise
  inherit the same ~10-15% too-fast flow.

## 5b. dH/dt plots updated with the early HO years (`check_ho_historical_timeseries_v2.py`)

Job 180030939. HO now: yearly 1996-2001 (chunk-1 rerun), 2002-2013 mean (H2013 - H2001; valid
since the rerun reproduces chunk 1 exactly), yearly 2014-2019. SSA and CPOM yearly. Fixed
coverage (7.563 Mkm^2), area-weighted. New files `models/ho_vs_ssa_historical_dhdt_domain_v2.png`
and `..._basins_v2.png`; earlier plots untouched. All HO numbers are from the UNRELAXED-start,
stale-mesh-z, too-fast (5a) HO run.

Whole domain (Gt/yr): 1996-2001 HO -222, SSA -104, CPOM +33; 2002-2013 HO -115, SSA -44, CPOM
-25; 2014-2019 HO -112, SSA -48, CPOM -57. HO yearly m/yr: -0.048 (1996) -> -0.021 (2001), then
-0.0165 mean 2002-2013 and -0.005..-0.025 in 2014-2019. HO's start-up transient is about twice
SSA's, and after it HO stays ~0.009-0.010 m/yr below SSA in every year with data (2014-2019
offset -0.0088 to -0.0096) -- a persistent excess matching the too-fast flow in 5a, not only a
start-up effect. SSA, once past its own ~6-yr transient, sits near CPOM in the domain mean
(2002-2013 -0.0064 vs -0.0035 m/yr).

Per basin (early / 2002-2013 / late, Gt/yr, HO | SSA | CPOM):
- 9 (Amundsen, = Mouginot 11): -107/-95/-83 | -89/-89/-73 | -32/-65/-97. Models steady or
  weakening; CPOM thinning triples. 8: -27/-24/-18 | -23/-22/-16 | -0.5/-13/-17. 10: -7/-1/+0.5
  | -8/-6/-4 | +1/-0.3/-5.
- 0 (= Mouginot 1): CPOM thickens (+8/+9/+20); HO -13/-3/-8 and SSA -9/+1/-4 thin -- wrong sign
  throughout. 6: late HO +1.4, SSA +1.9, CPOM +6.8. 11: early HO -5.9, SSA +0.3, CPOM +12.1.
- 14 (= Mouginot 16+17): SSA +27/+30/+31 vs CPOM +16/+32/+29 (good); HO -3/+8/+16 (far low).
  15: 2002-2013 HO -3.0, SSA +5.1, CPOM +8.2. 2: HO -14/-4/-11 vs CPOM +6/+0.5/-3.

**4z result -- HO relaxation rerun succeeded** (outer 180027937 exit 0, 8h46m, 1262 SU; inner
180031263). Forcing: RACMO 1979-1994 mean SMB (mesh-mean 0.549 m ice eq/yr, full coverage) and
the calibrated ismip6 melt (gamma_0=300). `sync_mesh_z` before the solve corrected the stale z
(198,014 columns, max 90 m, see 5a). Over the 1 yr: mean|dH| 6.60 m, max 868.37 m;
grounded/floating changed at 1.113% of vertices; initialization velocities (Vx, Vy, Vz, Vel)
taken from the last step. Old zero-SMB relaxation for comparison: mean|dH| 6.623 m, max 1020.34 m,
1.089% -- nearly the same mean, as expected for a 1-yr run where SMB is a small term. The
post-adoption `sync_mesh_z` moved z in 1,577,063 columns (base max 778 m, surface max 616 m):
that is the mesh following the adopted relaxed geometry, expected rather than an error.
New `models/AIS3_ho_relaxed.nc` (139 GB, 2026-09-29 00:36) now holds the relaxed state in its
geometry; `AIS3_ho_relaxed_prerelaxfix.nc` is kept. HO historical rerun NOT started: awaiting the
decision on HO's too-fast flow (5a/5b).

## 5c. HO stress-balance tests: rheology vs friction (submitted)

User chose to settle HO's too-fast flow (5a/5b) before the HO historical rerun, via a rheology
test. While setting it up, found from code that `finalize_melt_calibration.py` builds
AIS3_melt_gamma_tuning.nc from AIS3_ho_friction_inv.nc and uses its `md.friction.C` as is -- it
never copies `results.StressbalanceSolution.FrictionC` (the inversion's actual output) into the
model. Given the chunk-resume logic, the saved `md.friction.C` is the field chunk 3 STARTED from
(chunk 2's output), so every downstream HO step (melt calibration, relaxation, historical) may be
running one inversion chunk behind. To be confirmed with data.

`ho_stressbalance_tests.py` (`qsub -v PHASE=submit|analyze launch_ho_stressbalance_tests.pbs`,
job 180093951 submit phase): prints the friction provenance (relaxed model's friction.C vs chunk-2
output, chunk-3 output unsmoothed, chunk-3 smoothed), then submits single HO stress-balance
solves from the new relaxed model: A as is; B with SSA's rheology_B copied to all 15 layers;
C with the unsmoothed chunk-3 FrictionC (only if the friction in use differs from it). Analyze
phase compares surface/depth-avg speeds with observed and SSA speeds, domain and per basin.
New execution names AIS3_ho_sbtest_*; nothing existing modified.

**5c friction provenance -- CONFIRMED** (job 180093951, submit phase): the relaxed HO model's
`friction.C` is bit-identical to the chunk-2 output (chunk 3's starting field; max|d| 0). So every
downstream HO step -- melt calibration, relaxation (old and new), all historical chunks -- ran with
friction one inversion chunk behind. The size of the lag is small over most of the ice: chunk-3 vs
chunk-2 C differs by mean|d| 1.406 (max 3929); on grounded base vertices the chunk-3/in-use ratio
has median 1.001, p10 0.987, p90 1.031. So chunk 3 barely changed C in most places, and the stale
field is unlikely to explain a ~10-15% speed excess on its own; case C of the test measures it.
The smoothed chunk-3 field (used only by the failed chunk-4 attempt 5) differs more (mean|d|
1.711, max 7973). HO rheology_n = 3.

**5c solves submitted** (driver 180093951 exit 0; SSA rheology_n = 3, same as HO): A
`AIS3_ho_sbtest_A_asis` (job 180096981), B `AIS3_ho_sbtest_B_ssaB` (180097005), C
`AIS3_ho_sbtest_C_chunk3C` (180097089). Case A crashed 9 min in, during its first velocity solve,
with SEGV on many MPI ranks (exit 59). B and C, identical apart from the one changed field, ran past
that point, so this looks like a node/MPI fault rather than an input problem (a hugemem SIGBUS
also hit chunk-4 attempt 5, §4j). Resubmitted A's prepared `.queue` script unchanged (job
180100939); crash log kept as `AIS3_ho_sbtest_A_asis.outlog_segv1`.

## 5d. `melt_gamma_tuning` step replaced with the validated calibration (2026-09-29)

The `melt_gamma_tuning` step in `ais_0.1.py` still ran the per-vertex RMSE `gamma_0` sweep that
§4b showed is invalid. Its save branch wrote to `AIS3_melt_gamma_tuning.nc`, `ho_relaxation`'s
input, so running the step would have replaced the validated file with the invalid pick. The
production file was actually built outside `ais_0.1.py`, by `finalize_melt_calibration.py`
followed by `restore_floored_thickness.py`.

The step now builds that file directly, with no solve:
- loads `AIS3_ho_friction_inv.nc`
- restores the 100 m floored thickness from `AIS3_param.nc` (same method as
  `restore_floored_thickness.py`), then runs `sync_mesh_z`
- sets ismip6 with IMBIE2 basins, Zhou TF, `gamma_0=300` and the 16 refit `deltaT` values
  hard-coded from `deltaT_refit_state.json` candidate 1 (round 5, converged)
- keeps `md.friction.C` as loaded, i.e. the chunk-2 field, unchanged from what the current HO
  runs used (§5c)

It refuses to overwrite an existing output. `AIS3_MELT_TAG` writes a tagged copy instead.
The calibration search itself stays in the standalone scripts (§4a–4c).

Expected differences from the current file: no `TransientSolution` from finalize's 0.01-yr
salvage run (nothing downstream reads it), and `mesh.z` synced (`ho_relaxation` syncs it
anyway). Not yet verified by a rebuild-and-compare run.

**SSA, same change.** `melt_gamma_tuning_ssa` in `ais_0.1_SSA.py` ran the J1-only sweep with
the published `deltaT`, and saved its superseded pick (5537.7) to
`AIS3_melt_gamma_tuning_ssa.nc`. It now builds `AIS3_melt_final_ssa.nc` directly, which is the
file `historical_dhdt_tuning_ssa` and the projections read. The step:
- loads `AIS3_SSA_relaxed.nc`, as the calibration and `finalize_ssa_melt_calibration.py` did
- sets ismip6 with IMBIE2 basins, Zhou TF, `gamma_0=300` and the SSA refit `deltaT`
  hard-coded from `deltaT_refit_state_ssa.json` candidate 1 (round 5)
- does no thickness restore (already applied to `AIS3_SSA_inverted.nc`, §4k) and no solve
- has the same overwrite refusal and `AIS3_MELT_TAG` as HO

The old step's submit phase was also the first stage of the SSA calibration chain:
`ssa_melt_deltaT_basin_refit.py` (round 0) and `melt_deltaT_sensitivity_test_ssa.py` read their
starting melt from its runs, `execution_SSA/AIS3_melt_gamma_tuning_ssa_g{0..8}` (still on disk).
The sweep has moved to a standalone script, `melt_gamma_sweep_ssa.py`
(`qsub -v PHASE=submit|analyze launch_melt_gamma_sweep_ssa.pbs`). It uses the same model setup
and the same run names, and refuses to submit if those dirs exist. Its analyze phase only prints
basin totals and J1; nothing is saved. Neither the step nor the script has been run yet (user:
a rebuild-and-compare is not needed for now).


**5c: all three solves crashed identically -- not a node fault.** A (resubmitted, 180100939),
B (180097005) and C (180097089) all exited 59 with SEGV on 40-45 of 96 ranks at the same point:
the horizontal velocity solve completed (A after 9 min, since it starts from the relaxation's
final velocities; C 43 min; B 60 min), then the run died in "computing basal mass balance ->
ISMIP 6 Floating melting rate module -> ... computing vertical velocities". My earlier "node/MPI
fault" reading of A's first crash was wrong. This is the first stress-balance-only solve with
ismip6 basalforcings in this pipeline (every earlier ismip6 run was a Transient, which worked;
the friction-inversion stress-balance solves had default basalforcings). Likely the ismip6
time-dependent forcing is only fully set up inside a Transient -- not verified. ~580 SU lost.
Workaround (melt only sets the basal vertical-velocity condition, not the horizontal speed being
compared): `ho_stressbalance_tests.py` now uses the default basalforcings class with zero melt.
Failed run dirs kept as execution/AIS3_ho_sbtest_*_segv, submit logs as
launch_ho_stressbalance_tests_submit1.pbs.out/.err, state as ho_stressbalance_tests_state_segv1.json.
Submit phase resubmitted: job 180107158.
