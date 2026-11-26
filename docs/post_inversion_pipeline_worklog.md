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
| 5. Future projections | `projection_ssp_ssa` | **implemented 2026-09-29** (§6): anomaly forcing, chunked to 2300, CTRL/SSP/attribution/melt variants. Smoke test then CTRL first; SSPs wait for the CTRL drift check. J3 shows gamma_0=300 gives ~4% of ocean models' warming response (§6.1). No SMB elevation feedback | `AIS3_proj_ssa_<exp>_<year>.nc`, `..._history.npz` |

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

## 6. Projections (SSA track): setup and melt-sensitivity check

Design and experiment matrix: [`projection_experiments.md`](projection_experiments.md). Defaults
(chosen, open to change): SSA only; start 2019.0 from the historical end state; end 2300 (2100 for
ssp370, the only length ISMIP7 provides); anomaly forcing on the calibrated baseline, TF = Zhou TF0
+ [TF_gcm(t) - TF_gcm(1995-2014)], SMB = RACMO 1995-2014 mean + [SMB_gcm(t) - SMB_gcm(1995-2014)];
GCM SMB product dEBM2-8000m (the only one for all 7 GCMs). ISMIP7 forcing on disk: CESM2-WACCM and
MRI-ESM2-0 have ctrl/ssp126/ssp534-over/ssp585 to 2300 (+ssp370 to 2100); five more GCMs have
ssp370 to 2100 only.

### 6.1 J3 for the SSA refit candidates (`ssa_melt_j3_refit.py`, job 180110126)

J3 had never been computed with the per-basin refit deltaT (4c/4k picked gamma_0=300 on J1+J2).
Computed here against ISMIP7's term-3 targets (`melt_{cold,warm}_target_term3_v2.nc`: basin-mean
melt, kg/m2/a, 7 ocean models: Mathiot NEMO, Timmermann FESOM, Naughten FESOM ACCESS/MMM,
Jourdain-Naughten, Naughten MITgcm, Haid FESOM), using each model's own cold/warm TF.

Method: ISSM's ismip6 local melt reimplemented in Python from the ISSM source (Element.cpp
Ismip6FloatingiceMeltingRate, FloatingiceMeltingRatex.cpp): gamma_0*(1028*cp/(rho_i*Lf))^2 *
max(TF(draft)+deltaT_basin, 0)^2 on all-floating elements, TF linear in depth and clamped, P1DG
averaged to vertices. Checked against three existing ISSM refit solves (g1_r5, g4_r3, g8_r5):
max vertex difference 3e-14 m/yr, basin totals identical to 1e-15 relative -- so no cluster solves
were needed. The target rows equal each ocean model's own basin-mean melt exactly (8 km grid,
IMBIE2 basins, Bedmap3 floating mask), which also fixes the naughten_ais_1/2 = FESOM ACCESS/MMM
mapping.

| gamma_0 | J1 | J2 | J3 (kg/m2/a) | J3 on warm-cold change | param/target warm-cold, median |
|---|---|---|---|---|---|
| 100 (unconverged) | 6.38 | 115.4 | 3172 | 4125 | 0.019 |
| **300** | 0.85 | **12.6** | 3062 | 4034 | **0.038** |
| 1000 | 1.22 | 45.5 | 2943 | 3812 | 0.081 |
| 3000 | 0.90 | 102.9 | **2776** | **3498** | 0.155 |
| 5537.7 | 1.43 | 151.4 | 2780 | 3522 | 0.197 |
| 11075.5 (prior) | 1.10 | 240.2 | 2871 | 3731 | 0.281 |
| 15000 | 0.67 | 268.7 | 2975 | 3938 | 0.329 |
| 25000 | 1.32 | 345.3 | 3309 | 4596 | 0.439 |

- With deltaT refit to present-day melt, the warming response grows ~sqrt(gamma_0), as expected.
  At gamma_0=300 the parameterisation gives ~4% of the ocean models' cold->warm melt increase;
  even 25000 gives ~44%. Every candidate under-responds; 300 is the least responsive converged one.
- Monte Carlo over weights (10,000, terms normalised by their medians), 5th/50th/95th gamma_0:
  J1+J2+J3 300/300/15000 (300 wins 91%); J2+J3 300/300/3000; J1+J2 300/300/300. J3's spread
  across candidates is small (2776-3309) relative to J2's (12.6-345), so after median
  normalisation J2 decides -- the ISMIP7-style objective keeps 300 even though it is the
  weakest warming response.
- Consequence for projections: with the reference melt (gamma_0=300) the ocean-forced mass loss
  will be small by construction. A "high melt" member is needed to bracket it; see §6.2. A
  "low melt" member below 300 does not exist among converged candidates.
- **Decision (user, 2026-09-29):** keep gamma_0=300 as the reference melt; add gamma_0=15000
  (candidate g7, the J1+J2+J3 95th percentile, ~33% of the ocean models' response) as the single
  high-melt member (`AIS3_PROJ_MELT=high`); no low-melt member.

### 6.2 Projection code (`projection_ssp_ssa`, `launch_projection_ssa.pbs`)

Replaced the SSA scaffold. Per job (`qsub -v`): scenario, GCM, forcing (both/ocean/smb for
attribution), melt (`ref` or refit candidate `g<i>`), end year, smoke flag. First chunk builds
`models/AIS3_proj_ssa_init.nc` (historical 2019 end state adopted into geometry, results stripped)
and `AIS3_proj_ssa_baseline.npz` (TF0, SMB0); GCM 1995-2014 TF/SMB climatologies are cached on the
mesh per GCM. Chunks 2019->2050->2100->...->2300, each writing a new checkpoint
`AIS3_proj_ssa_<exp>_<year>.nc` and an annual history npz (thickness, ocean levelset, speed, melt,
SMB; VAF, grounded/floating area, total SMB/BMB, GL and ice-front flux), then qsubbing the next
chunk unless `models/AIS3_proj_ssa_<exp>.STOP` exists. Memory note: ISSM reads TF one depth at a
time (MasstransportAnalysis.cpp), so annual TF for a 50-yr chunk (~20 GB total) is fine on 48 ranks.
Analysis: `check_projection_ssa.py` (SLE from VAF, per basin, CTRL-relative).

**5c second attempt** (driver 180107158, exit 0): inner solves A 180111475, B 180111501, C 180111530.
A completed cleanly (exit 0, 10 min, .outbin written) through "computing basal mass balance" and
"computing vertical velocities" -- confirms the first-round crash was specific to the ismip6 melt
module in a stress-balance-only solve; the default-basalforcings workaround avoids it. (Worth
noting for any future stress-balance-only HO diagnostic: don't use ismip6 basalforcings there.)

### 6.3 Smoke test and CTRL launch

Smoke test (ssp585/CESM2-WACCM, 2019.0->2019.2; outer 180111657, inner 180112657). The first run
built `AIS3_proj_ssa_init.nc` (2019 end state adopted: mean|dH| 26.98 m vs the 1995 geometry,
1,265,464 grounded vertices), `AIS3_proj_ssa_baseline.npz` (SMB0 mesh-mean 0.5441 m ice eq/yr;
2019 RACMO 0.5379) and the CESM2-WACCM 1995-2014 TF/SMB climatologies. First-year anomalies:
TF +0.11 C (2019) and +0.19 C (2020) mean over floating vertices, all depths; SMB +0.014 and
+0.005 m ice eq/yr mesh-mean. The ISSM solve ran cleanly (2 steps, 57 s) with every requested
output present; first-step scalars: VAF 2.2458e16 m^3, grounded area 1.2102e13 m^2, TotalSmb
2393, TotalFloatingBmb 816, GroundinglineMassFlux 1618, IcefrontMassFlux 1730 (Gt/yr).

The driver then crashed in post-processing (TypeError): `md.results.TransientSolution` is a
pyissm `solution` whose `.steps` holds the steps; attribute access on the solution returns all
steps stacked. Fixed (iterate `ts.steps`) and checked offline against the smoke outbin: history
npz and scalar series written correctly, last-step extraction for the checkpoint matches. Not
rerun on the cluster.

Launcher resized to 32 cpu / 128 GB (smoke used 80 GB building the initial state): 190 GB needs a
whole free `normal` node and sat queued for hours.

CTRL launched: job 180114544 (`AIS3_PROJ_SCENARIO=ctrl`), chunk 2019->2050, self-chaining to 2300.
SSP runs wait for the CTRL drift check (`check_projection_ssa.py ctrl`).

**Allocation note (2026-09-29):** CTRL (180114544) first sat with "Could not reserve allocation
from project": au88 has 253.7 KSU left in 2026.q3 but **no 2026.q4 grant yet (0 SU)**, and a 46 h
walltime crossed the 30 Sep quarter boundary. Walltime cut to 24 h with qalter; the projection
step now sizes the inner solve to the chunk (2 x 12.6 min per model year: 13 h for 2019-2050, 21 h
for 50-yr chunks) and sets waitonlock to the driver's own walltime minus 90 min. Chunks that would
run into Q4 will wait in the queue until the Q4 grant is loaded.

**5c RESULT** (analysis job 180117730; solves A 180111475, B 180111501, C 180111530, all exit 0;
relaxed HO geometry, zero basal melt; grounded ice with observed speed; observed = SSA model's
vel_obs). Area-weighted mean speed (m/yr) and median vertex ratio to observed:

| | observed | SSA | A as is | B SSA's B | C chunk-3 friction |
|---|---|---|---|---|---|
| surface speed | 22.9 | 22.1 | 24.8 | 23.1 | 24.8 |
| median ratio to observed | -- | 0.99 | 1.09 | 1.02 | 1.09 |

Median B/A 0.963; median C/A 1.000.
- **Rheology is the main driver.** With SSA's rheology_B, HO's speed drops to about observed
  (median 1.09 -> 1.02). The East Antarctic basins where HO over-thinned are essentially fixed:
  b14 1.13 -> 1.00, b2 1.16 -> 1.03, b3 1.14 -> 1.04, b1 1.09 -> 1.01, b0 1.08 -> 1.03, b4 1.10 -> 1.04.
- **Part remains in the fast West Antarctic basins:** b8 1.12 -> 1.07, b9 1.06 -> 1.03, b10 1.08
  -> 1.05, b11 1.07 -> 1.04; also b15 1.17 -> 1.09, b5 1.14 -> 1.07. There the incomplete friction
  inversion (or HO physics) contributes too.
- **The one-chunk friction lag is irrelevant to speed** (C identical to A). Fixing the downstream
  copy of FrictionC is bookkeeping only.
- Caveat: HO friction was inverted with HO's soft B, so "SSA's B + HO's friction" is not a
  self-consistent pair; the result shows where the excess comes from, not a ready configuration.
  SSA's B is depth-uniform, which throws away the vertical temperature structure HO exists for.
- Map (models/ho_stressbalance_tests.png): A shows orange (too fast) broadly across the grounded
  ice, strongest along the East Antarctic coast and West Antarctica; B is closer to white
  (= observed) along the East Antarctic coast but shows purple (slower than observed) patches in
  the East Antarctic and Siple-side interior -- SSA's B overshoots there, i.e. is too stiff for HO
  where friction was tuned to HO's softer ice -- and still orange in West Antarctica, the
  Peninsula and fast outlets; C is indistinguishable from A.

**5c case D -- rescaled HO rheology (submitted).** User approved testing a usable HO fix: scale
each column of HO's rheology_B so its depth-average (trapezoid in sigma from the synced mesh.z)
equals SSA's B at that column, keeping HO's vertical profile. Same relaxed geometry, HO friction
as in use, zero basal melt. `ho_stressbalance_tests.py` now supports `CASES=` (submit a subset)
and merges the state json; `OUT_TAG=` suffixes the analysis plot. Submitted with
`qsub -v PHASE=submit,CASES=D launch_ho_stressbalance_tests.pbs` (job 180122920); execution name
AIS3_ho_sbtest_D_rescaledB. First analysis logs kept as launch_ho_stressbalance_tests_analyze1.pbs.out/.err.

**5c case D solved** (driver 180122920 exit 0; inner 180124339 exit 0, 60 min). Column rescale
factor SSA B / HO depth-avg B: grounded median 1.064 (p10 0.971, p90 1.341) -- HO's grounded
column-mean B is softer, strongly so in the p90 tail; **floating median 0.757** -- HO's ice-shelf B
is ~1.32x STIFFER than SSA's inverted shelf B. This supports the reading of the code that
`ho_thermal_steadystate` recomputed B = cuffey(T) everywhere, overwriting the inverted shelf
rheology on the shelves (not yet confirmed by direct comparison with load_shelf_rheology_B()).
Stiffer shelves mean more buttressing, so that alone would slow HO's grounded outflow, not speed
it up; case D restores the shelf B to SSA's. Analysis of A-D submitted: `qsub -v
PHASE=analyze,OUT_TAG=_v2 launch_ho_stressbalance_tests.pbs` (job 180181532). Submit-phase logs
kept as launch_ho_stressbalance_tests_submit3.pbs.out/.err.

**5c case D RESULT** (analysis job 180181532; plot models/ho_stressbalance_tests_v2.png, first plot
kept). Median surface-speed ratio to observed (area-weighted mean speed m/yr in brackets):
A 1.09 (24.8), B 1.02 (23.1), C 1.09 (24.8), **D 1.04 (23.7)**; observed 22.9, SSA 0.99 (22.1).
Median D/A 0.985 (B/A 0.963).
- D removes about half of HO's excess. Basins where it nearly matches B: 14 1.13 -> 1.02 (B 1.00),
  0 1.08 -> 1.04, 1 1.09 -> 1.04, 2 1.16 -> 1.06. Basins where D does little but B helped:
  3 1.14 -> 1.10 (B 1.04), 4 1.10 -> 1.10 (B 1.04), 5 1.14 -> 1.14 (B 1.07), 15 1.17 -> 1.12 (B 1.09).
  West Antarctica: 8 1.12 -> 1.10, 9 1.06 -> 1.04, 10 1.08 -> 1.08, 11 1.07 -> 1.06.
- B and D differ only in the vertical profile of grounded B (B: SSA's value at every depth; D: HO's
  profile scaled to SSA's column mean; both use SSA's shelf B). Where HO's column-mean B already
  matched SSA's (basins 3/4/5, column ratio ~0.96-1.00), D changes nothing but B still speeds
  things up less -- so there the excess comes from HO's SOFT BASAL LAYERS (warm steady-state basal
  temperature), not the column mean. An arithmetic column mean underweights the basal layers,
  which dominate deformation.
- Map: D shows less of B's purple (too slow) interior patching -- the overshoot is largely avoided
  -- but keeps orange (too fast) along the margins and in West Antarctica, like A in weaker form.
- The remaining West Antarctic excess (also left by B) points at the friction inversion / sliding.

**5c case D2 -- deformation-weighted rescale (submitted).** User approved. D matched HO's plain
column-mean B to SSA's; D2 instead matches the flow-effective column B for shear deformation on
grounded columns: B_eff = (sum w B^-n / sum w)^(-1/n) with w = (1 - sigma)^(n+1) (sigma 0 at the
bed, 1 at the surface), i.e. layer contribution to shear flux ~ A(z) (s - z)^(n+1). Floating
columns keep D's plain-mean match (plug flow). HO's vertical profile is kept. Submitted with
`qsub -v PHASE=submit,CASES=D2 launch_ho_stressbalance_tests.pbs` (job 180187628); execution name
AIS3_ho_sbtest_D2_defweightedB. Previous analysis logs kept as
launch_ho_stressbalance_tests_analyze2.pbs.out/.err.

### 6.4 CTRL 2019-2050 and stage-1 SSP launch (2026-09-30)

CTRL chunk 1 (driver 180114544, solve 180120303): exit 0; 31 yr in 5 h 25 min (10.5 min per
model year, faster than the historical run's 12.6), 76 GB, ~930 SU. Checkpoint
`AIS3_proj_ssa_ctrl_2050.nc` and history written; chunk 2050->2100 queued itself (180135512).

Drift (`check_projection_ssa.py ctrl`, `models/proj_ssa_summary.png`):
- VAF gain, i.e. sea-level fall: -0.546 mm/yr (2019-2030), -0.522 (2030-2040), -0.507
  (2040-2050); -16.2 mm by 2050 from ISSM's own VAF (about +190 Gt/yr above flotation). Smooth,
  nearly linear, decaying ~7% over 30 yr.
- Grounded area +11,810 km^2 (<0.1%); GL flux 1618 -> 1642 Gt/yr; area-mean grounded speed
  22.51 -> 22.90 m/yr; floating BMB 760 -> 754 Gt/yr; TotalSmb 2401 Gt/yr (constant SMB0).
- Per basin (mm SLE by 2050): only Amundsen loses (b9 +6.93, b8 +1.54, matching the historical
  over-thinning in 4s); gains elsewhere, largest b12 -5.66, b7 -3.75, b1 -3.46, b14 -2.91.
- Assessment: no runaway or GL instability; the drift is large (~-40 mm by 2100 if it holds) but
  smooth, so forced runs are reported as forced minus CTRL. The model sits well above observed
  present-day mass balance (IMBIE roughly -100 to -150 Gt/yr); not diagnosed further here.

Stage-1 reference runs queued (CESM2-WACCM, dEBM2-8000m, ref melt): ssp126 180191333, ssp370
180191335 (to 2100), ssp585 180191337. All four chains (with CTRL's next chunk) wait on the
2026.q4 grant (0 SU as of 30 Sep 13:00), since each 46 h chunk crosses the quarter boundary.

### 6.5 HO projections: scope and code (2026-09-30)

Scope agreed: 2-3 HO runs to 2100 (ctrl, ssp585 ref, optional ssp585 high melt) as a check on the
SSA ensemble, not an HO ensemble. Cost from the HO historical run's logs: solve 6 yr in 8.1-8.3 h
on 96 hugemem cpus, 2343-2409 SU (~395 SU per model year); driver 48-cpu hugemem 1913-2324 SU per
6-yr chunk (500-700 GB peak, from loading every 0.1-yr 3D step). About half of HO's cost was the
driver, so:
- `historical_dhdt_tuning`: output every 10th step (annual); the relaxed model's own 50-step
  history is dropped after loading; history times taken from the results; the final dH/dt
  diagnostic takes H(1995) from chunk 1's history file instead of reloading the 139 GB relaxed
  model. `launch_historical_dhdt_tuning.pbs` driver 48 cpu/1450 GB -> 24 cpu/720 GB (hugemem
  accepts 12/16/24-cpu requests with 45 h walltime, probed by qsub).
- `projection_ssp` (HO) replaces the scaffold: see projection_experiments.md, "HO runs".
  Estimate ~38k SU per run to 2100 (was ~61k with the old driver).
- `check_projection_ssa.py` reads HO tracks too (`ho_relaxfix/<exp>`, measured from the HO
  initial state, compared with HO CTRL).

HO smoke test (job 180193125): ssp585/CESM2-WACCM, 0.2 yr, started from the old unrelaxed
historical run (`AIS3_PROJ_HIST_TAG=''`, files `AIS3_proj_ho_*`, test only) because the
`_relaxfix` historical run does not exist yet. Full 48-cpu/1450 GB driver for this test only: that
old checkpoint still holds every 0.1-yr step.

**5c case D2 submitted** (driver 180187628 exit 0; inner solve 180191459, queued). On grounded
columns HO's deformation-weighted B is only 0.762 of its plain column mean (median) -- the soft
basal layers confirmed quantitatively. D2's grounded rescale factor (SSA B / HO B_eff): median
1.431, p10 1.064, p90 2.261, i.e. D2 stiffens HO's grounded ice by ~43% at the median column and
more than 2x in the p90 tail (D's plain-mean factor was median 1.064). D2 and B share the same
deformation-weighted column B (= SSA's); they differ in profile -- D2 keeps HO's soft base / stiff
top -- so D2's upper layers are much stiffer than SSA's, which may resist stretching in fast
streams more than B does.

**5c case D2 blocked by allocation (2026-10-01).** D2's solve (job 180191459) never started: it sat
queued through the end of 2026.q3, then was put on hold (state H, hold type "o") with comment
"Project au88 does not have sufficient allocation to run job (3.46 KSU required)". nci_account:
2026.q4 grant for au88 is 0 SU (used 0, avail 0); 2026.q3 granted 1.85 MSU, used 1.71 MSU, 138 KSU
unused and lost at quarter end. No hugemem work can run until a q4 grant exists. The job, its
.bin and queue script are intact; nothing was deleted. Once the q4 allocation arrives, release
it with `qrls 180191459` (or resubmit execution/AIS3_ho_sbtest_D2_defweightedB/*.queue), then
run the analysis as planned (PHASE=analyze, OUT_TAG=_v3).

HO smoke result: driver ran (initial state built, TF anomaly 2020 +0.191 C, SMB +0.0048 m/yr on
the base layer, mesh.z already in sync, 24 GB input .bin), but the 96-rank solve (180198606) failed
at its first output: `IcefrontMassFlux error message: not implemented` (Element.h:313) -- ISSM
implements it for 2D elements only. Removed from the HO outputs (analysis script tolerates its
absence). Rerun of the smoke test waits on the Q4 grant (all au88 jobs held since 1 Oct:
"insufficient allocation"). Cost of the failed test: 1040 SU driver + inner.

### 6.6 LARMIP setup (2026-10-01)

Request: 5+1 experiments (control + uniform extra sub-shelf melt per region), 200-yr yearly
sea-level series, model description, BibTeX, authors, by 30 Oct; optional linear-ramp set.
Implemented on the SSA track (HO not ready; HO uses Schoof, SSA uses Budd p=q=1):
- `projection_ssp_ssa` LARMIP option (`AIS3_PROJ_LARMIP_REGION/_MODE/_AMP`, `AIS3_LARMIP_MASK`):
  sets `md.basalforcings.melt_anomaly` (step: static field; ramp: time series 0/8/8 m/yr at
  2019/2119/2219); CTRL forcing otherwise; ends 2219. ISSM adds melt_anomaly inside
  Ismip6FloatingiceMeltingRate, i.e. only on fully floating elements. Mask loader checked offline
  on the SSA mesh with a provisional Amundsen-only mask (147,122 vertices, 23,146 floating).
- Existing settings already match what was agreed: `ismovingfront=0`, `melt_interpolation=
  NoMeltOnPartiallyFloating`, `SubelementMigration`, `SubelementFriction1`, static SMB0.
- `export_larmip.py`: yearly SLE (m) years 0-200, ISMIP-convention VAF; ISSM's own
  IceVolumeAboveFloatation differs by ~6% on CTRL 2020-2050 (-15.73 vs -16.75 mm) because it
  counts partially floating elements whole and does not clip negative height above flotation.
- Drafts: `docs/larmip/model_description.tex` (+ .md) and `larmip_refs.bib`; three references
  and the CTRL drift number still to be confirmed.
- Smoke test queued (held on the Q4 grant): region 3 step, 3 yr, provisional mask, job 180251370,
  to check the 0.1-yr timestep under 8 m/yr.

LARMIP-3 mask (2026-10-01): downloaded from https://www.pik-potsdam.de/~anders/larmip3/ to
`assets/larmip3/` (Infos text = the protocol as received; sector masks at 1/2/4 km, variable `z`,
values 0-5). Numbering identified on the AIS3 mesh: 1 Weddell, 2 East Antarctica, 3 Ross,
4 Amundsen, 5 Peninsula (floating areas 537k/205k/459k/91k/92k km^2; ~1,100 km^2 floating outside
all sectors). The 1 km file is now the default mask. The provisional smoke test (180251370, region
3 = Amundsen on my provisional mask) was deleted while held; resubmitted on the real mask as
region 4 (180251895), 3 yr, 8 m/yr step. Core-run amplitude (8 m/yr) is not stated in the protocol
text beyond the ramp's final value -- to confirm.

### 6.7 Q4 grant in; progress, slow-node incident, LARMIP launched (2026-10-02)

The 2026.q4 grant (2.00 MSU) arrived on 1 Oct and the held jobs ran.
- CTRL: 2050 -> 2100 -> 2150 done; 2150-2200 running (step 352/500 after 2.9 h). Chunk solve times
  vary with the node: 2019-2050 5h25 (31 yr), 2050-2100 10h17, 2100-2150 6h46 (50 yr each).
- ssp370 (CESM2, ref): complete to 2100 (2019-2050 5h26, 2050-2100 10h18).
- ssp126/ssp585 (CESM2, ref): 2019-2050 done (5h16/5h19). Their 2050-2100 solves (180285501 on
  gadi-cpu-clx-1475, 180285537 on clx-1491) ran ~7 min per 0.1-yr step, ~5x slower than every
  other chunk, with all 48 cpus busy, no solver warnings, inputs identical in size to ssp370's
  (20.8 GB .bin) and the same 2050 state as the other scenarios; ssp370's identical chunk ran
  normally alongside. Taken to be slow nodes. At step 119/130 of 500 after 15 h they could not
  finish inside the 21 h limit, so both drivers and solves were deleted, the partial run
  directories archived (`execution_SSA/..._2050-2100_killed_slownode_20261002`), and both chunks
  resubmitted from their 2050 checkpoints (180326867, 180326868). ~2.9k SU lost.
- LARMIP smoke (180251895, region 4 Amundsen, 8 m/yr step, 3 yr): stable at the 0.1-yr step, 16 min;
  floating BMB 760 -> 1387 Gt/yr (+627 vs 8 m/yr x 91,000 km^2 = 668 Gt/yr; extra melt only on
  fully floating elements).
- LARMIP core runs launched: larmip_r1..r5_step, 180326869-73 (control = the CTRL chain).
- Checked: the ~115k floating vertices at the 1 m minimum thickness are ocean cells outside the
  ice mask (114,315 at 2019); inside the ice mask 390 at 2019, 2,798 by 2150 in CTRL.

**5c case D2 solved** after the 2026.q4 grant arrived (2.00 MSU): job 180191459 was released and ran
on 2026-10-01, exit 0, 61 min, .outbin written. Five-case analysis submitted
(`qsub -v PHASE=analyze,OUT_TAG=_v3 launch_ho_stressbalance_tests.pbs`); D2 submit-phase logs kept
as launch_ho_stressbalance_tests_submit4.pbs.out/.err.

**Later 2026-10-02 -- first forced result and stage-2 launch.** ssp370/CESM2 (ref melt) minus CTRL:
+0.02 mm (2020), -8.6 mm (2050), -57.5 mm (2100): under ssp370 the model *lowers* sea level relative
to the control. Expected direction given the inputs -- SMB increases with warming and the reference
melt (gamma_0=300) responds to ocean warming only ~4% as strongly as the ocean models (6.1) -- but at
the low end of ISMIP6-type ranges. CTRL drift over 2019-2150 is -0.463 mm/yr; 2150 basin pattern as
before (only b8/b9 lose). The attribution and high-melt runs below test this directly.

Launched (budget ~200k SU for this campaign, user): LARMIP ramps r1-r5 (180326922-26); stage 2,
SSA: ssp585/ssp370 CESM2 high melt (180327202/03), ssp585 CESM2 ocean-only/SMB-only (180327205/06),
MRI-ESM2-0 ssp126/ssp585/ssp370 (180327207-09), ssp370 x ACCESS-CM2, CanESM5, GFDL-ESM4,
IPSL-CM6A-LR, MPI-ESM1-2-HR to 2100 (180327210-14). ~60k SU for stage 2, ~130k SU committed in
total. Climatology cache temp files now carry hostname+pid (several runs of one GCM start together).

**5c case D2 RESULT** (analysis job 180326968; plot models/ho_stressbalance_tests_v3.png).
Median grounded surface-speed ratio to observed: A 1.09, B 1.02, C 1.09, D 1.04, **D2 1.00**
(area-mean 21.8 vs observed 22.9 m/yr, SSA 22.1). Median D2/A 0.921.
- D2 brings every basin to within ~6% of observed and removes the excess D left: b3 0.97 (D 1.10),
  b4 1.00 (1.10), b5 0.99 (1.14), b15 1.03 (1.12), and West Antarctica b8 1.00 (1.10), b9 0.98
  (1.04), b10 1.00 (1.08), b11 1.00 (1.06). It does NOT over-stiffen the fast streams in the
  median; the remaining WA excess after B was also a basal-softness effect.
- Slight undershoot in places: b14 0.94, b1 0.96, b2 0.98, b9 0.98; the area-weighted mean
  (21.8) sits ~5% below observed and ~1% below SSA, while the median is 1.00 -- i.e. some of the
  larger/faster areas are a little slow. Map: D2 is the closest to white overall, with scattered
  purple (too slow) patches in the East Antarctic interior and around parts of the coast, similar
  in extent to B's and more than D's; margins still show orange/purple speckle from mesh-scale noise.
- Ranking for a pragmatic fix: D2 (median exactly 1.00, best in WA and the soft-base basins,
  mild slow bias in a few EA basins) > B (1.02, discards HO's vertical profile) > D (1.04).
- These are diagnostic solves (zero basal melt, HO friction inverted with the OLD soft B, relaxed
  geometry from the old-B relaxation); the pairing of D2's B with that friction is not
  self-consistent, but the residual is now within the inversion's own error level.

## 5d. D2 rheology adopted in the HO chain (code change, not yet run)

User decision (2026-10-02): adopt D2. `config/ais_0.1.py` (backup `ais_0.1.py.pre_d2_backup`):
- New helper `rescale_rheology_B_d2(md, B_ref_2d)`: per-column rescale keeping the vertical
  profile; grounded columns match the deformation-weighted B (w = (1-sigma)^(n+1)), floating
  columns the plain mean; idempotent.
- `ho_thermal_steadystate`: B = cuffey(3D T) is now rescaled (D2) to the input model's pre-thermal
  B, which also restores the inverted shelf rheology the old code overwrote. Only affects future
  reruns of the thermal step.
- `ho_relaxation`: after SMB/mesh-z setup, applies D2 against AIS3_inverted.nc's rheology_B (the
  current melt-tuned file still carries the old unscaled B). No-op once the chain is rebuilt.
Reference note: AIS3_inverted.nc (HO track) and AIS3_SSA_inverted.nc (SSA track, used in the 5c
tests) have identical grounded B but different inverted shelf B (14% of vertices differ, ratio
0.63-1.43 at p1-p99). The HO chain uses its own AIS3_inverted.nc, so shelf B (and buttressing)
differs slightly from the tested D2 case; grounded speed should be close to the tested result.
Not changed: the HO friction field (still inverted with the old soft B), the one-chunk friction
bookkeeping lag, and the melt calibration (a mass-transport-only transient, independent of B).
Next: rerun ho_relaxation with D2 B, then the relaxed-start HO historical rerun.

**5d: HO relaxation rerun with D2 submitted** (job 180330595, `qsub -v AIS3_STEPS=ho_relaxation
launch_ho_relaxation.pbs`). Archived first (mv -n, nothing deleted): models/AIS3_ho_relaxed.nc ->
AIS3_ho_relaxed_oldB.nc (the 2026-09-29 relaxed-with-old-B run), execution/AIS3_ho_relaxed ->
AIS3_ho_relaxed_oldB, launcher logs -> launch_ho_relaxation_oldB.pbs.out/.err. Earlier
zero-SMB relaxation still kept as AIS3_ho_relaxed_prerelaxfix.nc.

**5d relaxation startup** (driver 180330595, inner compute job 180332266): D2 applied as tested --
grounded scale median 1.430 (p10 1.064, p90 2.261), same as the test; floating median 0.758 (test
D, against SSA-track shelf B: 0.757), so despite the shelf-B differences between the two tracks
the median shelf correction is the same. mesh.z resync before the solve showed the same stale-z
signature as the old-B relaxation (198,014 columns, max 90 m), as expected since both start from
AIS3_melt_gamma_tuning.nc.

**2026-10-02 afternoon -- progress check; ssp585 slow again; 25-yr chunks.** CTRL reached 2200
(15:26). 25 running solves at 60-165 steps/h (LARMIP core and ramps, stage 2, ssp126 rerun) except
ssp585/CESM2 ref 2050-2100 rerun (180327838, gadi-cpu-clx-2103): 48/500 after 2.25 h, then 2 steps
in 7 min (~3.5 min/step, ~5x normal), while ssp126's rerun of the same chunk is normal (210/500).
Second slow attempt for this chunk on a different node, so possibly specific to this run (cause not
identified: the CESM2 ssp585 TF files for 2045-2100 are clean, max 10-12 C, no fill values; the
2050 state matches the other scenarios). Deleted (driver 180326868 + solve), run directory archived
(`..._2050-2100_killed_slow2_20261002`), resubmitted (180334483).
Code: chunk edges now 2019, 2050, then every 25 yr (2075 ... 2300); inner solve time request 3x the
12.6 min/yr estimate (16 h per 25-yr chunk), so a solve at ~1/3 normal speed still finishes.
Running chunks keep their old 50-yr length; chains switch at their next chunk. If ssp585 2050-2075
is slow a third time, run a short diagnostic with stress-balance convergence output on.

**5d relaxation with D2 RESULT** (driver 180330595, exit 0, 2026-10-02 22:56, 8h29m, 1221 SU).
Over the 1 yr: mean|dH| 6.07 m (old-B 6.60), max 1155.17 m (old-B 868.37), grounded/floating
changed at 1.077% of vertices (old-B 1.113%); initialization velocities set from the last step.
Mean adjustment down ~8% with the stiffer grounded ice; the larger max is a local extreme (likely
margins/shelves, where D2 softened the shelf B to the inverted value, median factor 0.758) --
not yet located. New models/AIS3_ho_relaxed.nc (2026-10-02 22:56); AIS3_ho_relaxed_oldB.nc and
AIS3_ho_relaxed_prerelaxfix.nc kept. HO historical rerun not started: awaiting user go-ahead.

### 6.8 Status 2026-10-06; MRI calendar bug; lost MPI driver

Complete: CTRL (2300); all 10 LARMIP runs (step + ramp, to 2219); ssp126/CESM2 (2300); ssp585/CESM2
ocean-only and SMB-only (2300); ssp370 x CESM2 ref + high, MRI, ACCESS-CM2, CanESM5, GFDL-ESM4,
IPSL-CM6A-LR (2100). Running: ssp585/CESM2 ref (2275-2300), ssp585/CESM2 high (2250-2275).
au88 Q4 use 177k SU of 2.0 MSU (whole project).

- **ssp585 slow chunk was the node after all**: the third attempt at 2050-2075 took 3h25 (later
  ssp585 chunks 2.6-3.5 h). Chunk-time growth is real for the high-melt run, though:
  ssp585 high 2100-2125 7h51, 2225-2250 11h56 (ref ~3.5 h) -- stronger melt and retreat make each
  solve harder. The 3x time request covers it (2250-2275 projected ~9.4 h of 16 h).
- **MRI-ESM2-0 ssp126/ssp585 drivers failed at the 2250->2275 chunk**: `open_mfdataset` "Cannot
  combine along dimension 'time' with mixed types (Timestamp, DatetimeProlepticGregorian)" -- MRI's
  proleptic-Gregorian dates decode as pandas timestamps before ~2262 and cftime after. CESM2 (noleap)
  never mixes. Fixed in both projection steps (decode TF times as cftime throughout; tested on the
  MRI 2241-2290 files). Both resubmitted from 2250 (180576566/67).
- **ssp370/MPI-ESM1-2-HR stuck at 2050**: its 2050-2075 solve finished normally (3h41, exit 0,
  3 Oct 00:43) but the driver left no checkpoint, no next chunk and no PBS log; the job is no longer in
  PBS history, so the cause is unknown (the only one of ~120 chunk drivers). Run directory archived
  (`..._2050-2075_driverlost_20261006`), rerun from 2050 (180576568).

### 6.9 LARMIP series exported (2026-10-06)

`export_larmip.py --ramp` (batch job 180580366, 4 min after fixing the reads: `dict(np.load(f))`;
indexing an NpzFile re-reads the whole array every time, which made the login-node run time out;
same fix in check_projection_ssa.py) -> `docs/larmip/AIS3_ssa_larmip_sle.csv`, 11 series x 201 years,
complete. Response at year 200, experiment minus control (mm SLE, ISMIP-convention VAF):

| sector | step | ramp |
|---|---|---|
| 1 Weddell | 51.5 | 35.4 |
| 2 East Antarctica | 23.1 | 17.0 |
| 3 Ross | 27.5 | 20.8 |
| 4 Amundsen | 11.9 | 9.6 |
| 5 Peninsula | 3.5 | 3.0 |

Control: -88.7 mm over 200 yr, rate -0.56 (first 30 yr) -> -0.39 mm/yr (last 50 yr); the model
description now states this.
- VAF method matters: with ISSM's own IceVolumeAboveFloatation the year-1->200 differences are
  Weddell 58.4, East Antarctica 26.0, Ross 31.7, Amundsen 13.8, Peninsula 4.6 mm (8-30% higher).
- The Amundsen response is small for the sector usually most sensitive in marine-ice-sheet terms;
  not investigated yet (it has the smallest floating area of the large sectors, 91,000 km^2, and the
  setup uses linear Budd sliding, fixed calving front, no melt on partially floating elements).

### 6.10 BUG: walltime-killed solves were adopted as complete chunks (time skipped) -- fixed

Found from straight segments in `docs/figures/projections_ssa.png`. Three inner solves hit their 16 h
limit (slow nodes: 63, 121 and 215 of 250 steps): ssp126/CESM2 2225-2250 (stopped at 2231.3),
ssp585/CESM2 ref 2225-2250 (2237.1), ssp585 SMB-only 2125-2150 (2146.5). pyissm loaded the partial
outbin without error; the driver adopted the last step and saved it as the chunk-end checkpoint, then
chained on. Net effect: those runs skipped 19, 13 and 4 model years, and everything after is invalid.
A scan of all 188 projection solve directories found no other walltime kills (LARMIP unaffected; its
export had all 201 years for all 11 series).
- Fix (SSA and HO steps): after loading results, stop with SystemExit unless the last output time
  equals t1 -- no history, no checkpoint, no next chunk.
- Archived (renamed `*_INVALID_timeskip`, nothing deleted): ssp126 and ssp585 ref checkpoints after
  2225, SMB-only after 2125, with their histories and run directories (24 files, 13 dirs); the
  ssp585 ref 2275-2300 chunk then running on the bad state was deleted (180566473/180567047).
- Restarted from the last valid checkpoints: ssp126 and ssp585 ref from 2225, SMB-only from 2125
  (180581952-54).
- Drivers already running when the guard went in (ssp585 high 2250-2275; MRI ssp126/ssp585 from 2250;
  MPI ssp370 from 2050) do not have it: check their chunk end times by hand.

**5d: where the D2 relaxation's max |dH| is** (`check_ho_relaxed_d2_extremes.py`, job 180575405;
plot models/ho_relaxed_d2_extremes.png). dH = relaxed end state minus relaxation input.
- Overall D2 has FEWER large changes than the old-B relaxation: |dH| > 100 m 6,445 vs 7,157;
  > 300 m 354 vs 378; > 500 m 47 vs 52. By basin (>300 m) D2 is lower in b3 (44 vs 71) and b6
  (18 vs 24), higher in b7 (21 vs 16), b12 (19 vs 15), b13 (4 vs 1), b14 (22 vs 14).
- The 1155 m maximum is at x=389.5 km, y=-1186.3 km (78.55 S, 161.82 E; basin 7), the western
  margin of the Ross Ice Shelf by the Transantarctic Mountains. Floating at start and end, 1938 m
  thick at ~flotation over a -1737 m bed (a deep, narrow inlet). It is a coherent ~20-40 km strip
  along the grounding line (window +-150 km: 76 vertices > 100 m in D2 vs 59 in old B), not one
  vertex. The old-B run already had the same spot as its maximum (-868 m). Modelled speed there is
  wildly too fast in BOTH runs: observed 4 m/yr, old B 529, D2 867; neighbours obs 33-46 vs D2
  316-680 / old 190-432. D2's floating-column rescale softened these columns (D2/old column-mean
  B 0.38-0.83), consistent with D2 amplifying a pre-existing local problem (likely the inverted
  shelf B and/or poorly resolved geometry in the inlet) rather than creating a new one.
- About half of the top 25 are vertices with zero modelled and observed speed whose thickness
  collapsed to the 1 m floor IDENTICALLY in both runs (e.g. b8, b9, b4, b5, b6 shelf-edge
  points): isolated/detached ice outside the active flow, unrelated to D2.
- The rest are grounding-zone G->F transitions with similar dH in both runs (b3, b1, b14).

## 5e. Relaxed-start HO historical rerun with D2 rheology (started 2026-10-06)

User approved, accepting the western-Ross-margin inlet artifact as a known local issue (5d).
New launcher `config/launch_historical_relaxfix.pbs` (sets AIS3_STEPS=historical_dhdt_tuning,
AIS3_HIST_TAG=_relaxfix; own log names, so the finished run's launcher logs are untouched).
Outputs: models/AIS3_historical_1995_2019_relaxfix.nc (checkpoint), per-chunk
..._relaxfix_history_<t0>-<t1>.npz, execution/AIS3_historical_1995_2019_relaxfix. Starts from the
D2-relaxed models/AIS3_ho_relaxed.nc. Chunk 1 (1995-2001): job 180584759. Resubmit the same
launcher for chunks 2-4, archiving its .out/.err with a chunk suffix (mv -n) each time.

### 6.11 Campaign results (2026-10-07; SMB-only rerun at 2250, finishing)

All other runs complete; no gaps in any series; the four pre-guard chunks reached their end years.
Sea level minus CTRL (mm, + = rise), ISMIP-convention VAF (`models/proj_ssa_sle_series.csv`,
`docs/figures/projections_ssa.png`):

| run | 2100 | 2150 | 2200 | 2250 | 2300 |
|---|---|---|---|---|---|
| ssp126 CESM2 | -43 | -82 | -121 | -167 | -213 |
| ssp585 CESM2 ref | -73 | -143 | -148 | -19 | +211 |
| ssp585 CESM2 high melt | -59 | -97 | -62 | +86 | +313 |
| ssp585 CESM2 ocean only | +1 | +4 | +10 | +19 | +31 |
| ssp585 CESM2 SMB only | -74 | -147 | -156 | -33 | (running) |
| ssp126 MRI | -18 | -37 | -62 | -91 | -124 |
| ssp585 MRI | -42 | -80 | -80 | -60 | -39 |

ssp370 in 2100 across seven GCMs: -34 (MPI) to -64 (ACCESS-CM2); CESM2 -57, high melt -44.
- The CESM2 ssp585 reversal (minimum ~-155 mm near 2185, +211 mm by 2300) is SMB-driven: the
  SMB-only run follows the combined run through the turn (-33 vs -19 mm at 2250) and ocean-only +
  SMB-only (-14) is close to the combined run, i.e. near-additive. MRI ssp585 turns only weakly
  (-80 -> -39), so the size of the reversal is GCM-specific.
- Ocean forcing alone gives +31 mm by 2300 with the reference melt; high melt (gamma_0=15000) adds
  ~100 mm by 2250-2300 at ssp585 but only +13 mm by 2100 at ssp370.

**5e chunk 1 done** (driver 180584759, inner 180607207; exit 0, 2026-10-07 00:30, driver walltime
10h26m, solve 6h32m, 1501 SU). Fresh start from the D2-relaxed model confirmed; mesh.z mismatch
at start 0 m. Outputs: models/AIS3_historical_1995_2019_relaxfix.nc (73 GB) and
..._relaxfix_history_1995-2001.npz (7 states: t=1995.1, 1996, ..., 2001). The finished run
(AIS3_historical_1995_2019.nc, 2026-09-25 03:14) is untouched. Only 7 states because
`ais_0.1.py` now sets `md.settings.output_frequency = 10` in historical_dhdt_tuning (edited
outside this session on 2026-09-30: "annual output ... loading every 0.1-yr 3D step made the
driver job need 500-700 GB"), and the history writer now reads the real solution times; this also
explains the smaller checkpoint. Annual states are what the dH/dt analyses use. Logs kept as
config/launch_historical_relaxfix_chunk1.pbs.out/.err. Chunk 2 (2001-2007): job 180708117.

**5e chunk 2 done** (driver 180708117, inner 180730490; exit 0, 2026-10-08 03:10, driver walltime
11h33m, 1663 SU). Resumed correctly from t=2001.0; the pre-sync mesh.z mismatch on resume (base
max 697 m, 1.47M columns) is expected -- z still held chunk 1's start geometry until sync_mesh_z.
Wrote ..._relaxfix_history_2001-2007.npz; 1995-2001 history kept; finished run untouched. Logs
kept as launch_historical_relaxfix_chunk2.pbs.out/.err. Chunk 3 (2007-2013): job 180779544.

**2026-10-08.** SMB-only rerun: its 2250-2275 solve (180698031, gadi-cpu-clx-0007) was walltime-
killed at 178/250 steps (11 steps/h; this run's previous chunk managed 59/h, and the combined
ssp585 run did 2250-2275 in 5h20 under the same SMB, so a slow node, not the forcing). The guard did
not get to act: a killed solve never writes the lock file, so the driver (180697637) sat out its
wait budget (24 h of 44.5 h used, ~64 SU/h idle) -- deleted by hand. Run directory archived
(`..._2250-2275_killed_walltime_20261008`); resubmitted from 2250. No bad output was written.
Still open: the driver cannot tell a killed solve from a queued one, so a kill costs up to the
whole wait budget in idle driver time.
Noted: CESM2 ssp585 dEBM2 SMB anomaly reaches -3.1 (2250) and -3.8 (2275) m ice eq/yr as a mesh
mean (SMB0 0.54) -- the SMB-driven reversal in 6.11.

**5e chunk 3 done** (driver 180779544, inner 180796316; exit 0, 2026-10-09 00:37, driver walltime
10h30m, 1513 SU). Wrote ..._relaxfix_history_2007-2013.npz; earlier histories kept; finished run
untouched. Logs kept as launch_historical_relaxfix_chunk3.pbs.out/.err. Chunk 4 (2013-2019, final)
submitted (job 180846839).

**5e chunk 4 done -- run complete** (driver 180846839, inner 180861817; exit 0, 2026-10-09 20:50,
driver walltime 10h26m, 1503 SU). "Final chunk complete -- full 1995-2019 run finished".
- Wrote ..._relaxfix_history_2013-2019.npz; all four history files are present. The checkpoint
  models/AIS3_historical_1995_2019_relaxfix.nc is 73 GB.
- Logs kept as launch_historical_relaxfix_chunk4.pbs.out/.err. The finished run is untouched.
- Grounded dH/dt RMSE vs dhdt_cpom: **1.593 m/yr**. Old HO run 1.768; SSA historical 1.402.
- Same caveat as before: this is a spatial RMSE of the run's dH/dt field against a single CPOM
  field, not the area-weighted, fixed-coverage, per-basin time-series comparison of 4s-4u.
  That comparison (relaxfix vs old HO vs SSA) has not been done yet.
- Total cost of the rerun (4 chunks): ~6180 SU.

### 6.12 ISMIP7 ice-shelf collapse (fracture path C) added (2026-10-09)

Option `AIS3_PROJ_COLLAPSE=1` (+ `AIS3_PROJ_COLLAPSE_VER`) in `projection_ssp_ssa`. It is off by
default, so existing experiments are unchanged. Collapse runs get a `_collapse` suffix, so
nothing is overwritten. The HO `projection_ssp` refuses the option (NotImplementedError) until
the method is ported.
- Mask: `fracture/<ver>/ice_shelf_collapse_mask_*` for the run's GCM/scenario; newest version by
  default (CESM2 v2.1, MRI v2 or v1). Annual int8 0/1 on the TF 8 km grid (checked at run
  time), years 1950-2299 (ssp370: to 2100). None for ssp245, ctrl or the ssp370-only GCMs.
- **Attempt 1, extra melt (dropped).** 1e4 m/yr `melt_anomaly` on masked vertices, which ISSM
  applies only under fully floating elements.
  - Smoke 180848028 (2019 mask, 0.2 yr): nothing to test, because all 81 masked mesh vertices
    in 2019 are already ice-free. These are Peninsula shelves gone before our start.
  - Masked floating-ice vertices: 1,399 by 2050 and 33,830 by 2100.
  - Smoke 180848732 (2100 mask via `AIS3_PROJ_COLLAPSE_SMOKE_SHIFT=81`, 0.5 yr, 38 SU): 97%
    of masked floating vertices reached 1 m. But the solve blew up: max speed 4.3e9 m/yr,
    thickness spikes to 20 km, IcefrontMassFlux 1.5e3 -> 2.1e6, -5.1 mm SLE in 0.5 yr.
  - Cause: thick shelf remnants embedded in 1 m ice have almost nothing holding them, and the
    fixed 0.1-yr step cannot follow.
  - History archived as ..._collapse_2019-2020_smoke_meltanomaly_history.npz.
- **Attempt 2, levelset removal between sub-solves (current).**
  - The chunk is split at each 1 January on which the mask changes on the mesh.
  - Before each sub-solve, masked vertices holding floating ice become ice-free (ice_levelset
    +1, H = 1 m, floating). So do floating pieces that this newly cuts off from grounded ice:
    connected components over fully-ice elements; the 64 already-detached vertices in the
    init state are left alone.
  - With ismovingfront=0 the new edge is an ordinary front.
  - Each sub-solve gets only the forcing columns around its own years (TF is 30 depths x 1.59M
    x 27 columns, ~10 GB per chunk).
  - The walltime guard runs per sub-solve. History gets `collapsed_at`.
  - Cost: one inner job and queue wait per mask change, i.e. roughly annual after ~2030 under
    ssp585.
- Smoke 180862834 (2100 mask, 1.5 yr): NaN at step 4; 100 nonlinear iterations every step.
  - The failed solve wrote no lock file, so the driver sat waiting (6.10); killed by hand.
  - `check_collapse_smoke.py`: 1e15-1e17 m/yr at small grounded patches left by the cut
    (65.8S 99.8E, 69.2S 31E). These have friction coefficient exactly 0 (40,851 of 1.26M
    grounded ice vertices have fc = 0), and/or join the sheet only at single vertices or
    through partly grounded elements.
  - Fix: a piece is held only if it is joined across shared edges to an element that is fully
    grounded with fc > 0 at all three vertices. After a cut, pieces that are not held are
    removed if newly cut off, inside the mask, or touching the cut.
  - On the 2100 mask: 33,830 cut + ~1,230 such pieces removed; pieces far from the collapse
    are left as in the baseline.
- Smoke 180869085: rejected by the consistency check ("base = bed on grounded ice"). Removed
  grounded pieces keep ocean_levelset > 0 but float at 1 m. Fix: removed vertices take the
  flotation state of 1 m ice (base hydrostatic, ocean_levelset -1 where it floats).
- Smoke 180869837: no NaN, no singular modes.
  - Max speed 1.4e5 m/yr (t=2019.1) and 1.9e5 (2019.2) at newly exposed thin grounded fronts
    (67.5S 59.5E; 66.1S 99.8E), which thin fast (574 -> 144 m). Then 2.0e4 (2019.3) and
    1.7e4 (2019.4), at the 2,240 m Amery-area front left at 73.0S 67.5E.
  - Ice faster than 20 km/yr: 546 vertices -> 0 by t=2019.4.
  - Every step hit the 100-iteration cap (~12 min/step vs ~1.3 normally). The 1-h inner job
    was walltime-killed at t=2019.5; the guard stopped the chain correctly.
  - Read: the mechanism works, and the cost is slow steps after a big collapse. This test
    removes 300k km² at once on 2019 geometry; real masks arrive yearly (1,399 floating ice
    vertices by 2050).
  - Collapse sub-solves now get an inner time of max(4 h, 10x the normal rate).

**2026-10-09 -- campaign complete.** All 26 SSA experiments reached their end years (CTRL and SSPs
to 2300 or 2100, LARMIP to 2219); the SMB-only rerun finished 9 Oct 01:50 (last two chunks 5h23,
6h00). Final series `models/proj_ssa_sle_series.csv` (no missing years) and
`docs/figures/projections_ssa.png`. SSP5-8.5 CESM2 attribution at 2300 (minus CTRL): SMB only +193,
ocean only +31, sum +224, combined +211 mm -- near-additive, SMB-dominated.

**6.12 launch (2026-10-13).** `ssp585_cesm2_ref_ref_collapse` started from 2019 with the real
yearly v2.1 masks (driver 181109462; logs/proj_ssa_ssp585_cesm2_ref_ref_collapse_2019.out).
Before release, the driver got an early chunk end: a collapse chunk has ~30 sub-solves (one
per mask change). If driver time left < the next sub-solve's inner time + 1 h, the chunk ends
at that mask-change year, writes its checkpoint and queues the next chunk. Otherwise a long run
of queue waits would hit the wait budget and lose the whole chunk. Non-collapse runs are
unchanged: one sub-solve, so the check never triggers.

### 6.13 Figures: relaxed-HO historical comparison and continental maps (2026-10-13)

**Historical dH/dt, relaxed-start HO (D2) vs old HO vs SSA vs CPOM**
(`config/plot_historical_relaxfix_comparison.py`; `docs/figures/historical_dhdt_relaxfix_{domain,basins}.png`).
Same fixed-coverage grounded mask as 4s/4w/4x/5b (599,285 vertices, 7.563 Mkm^2); SSA and CPOM
recomputed and identical to the earlier output (max difference 0.0000 m/yr), old HO taken from that
output. Whole domain, Gt/yr (1996-2001 | 2002-2013 | 2014-2019):

| | 1996-2001 | 2002-2013 | 2014-2019 | RMSE vs CPOM (m/yr) | corr |
|---|---|---|---|---|---|
| CPOM | +33.0 | -24.5 | -56.9 | | |
| SSA | -103.5 | -44.4 | -47.9 | 0.0110 | +0.14 |
| HO, old (unrelaxed) | -222.3 | -114.4 | -111.7 | | |
| **HO, relaxed + D2** | **-80.3** | **-29.8** | **-29.1** | **0.0096** | +0.13 |

- The relaxed, D2 HO run removes most of the old HO's excess thinning (~-110 -> ~-30 Gt/yr after
  the start-up years) and is now at least as close to CPOM as SSA in the mean; start-up transient
  smaller than SSA's (-80 vs -104 Gt/yr in 1996-2001). Neither has interannual skill (corr ~0.13).
- It under-thins 2014-2019 (-29 vs CPOM -57). Amundsen (basin 9) unchanged in character: -87/-91/-74
  vs CPOM -32/-65/-97 -- over-thins early, misses the acceleration, as SSA does.
- Basin 10 improves (RMSE 0.134 SSA -> 0.086 HO), basin 8 too (0.129 -> 0.089).

**Continental maps** (`config/plot_continental_maps.py`; symmetric-log diverging scale, +-5 m linear):
- `docs/figures/maps_larmip_dH.png`: step runs minus control at 2219. Thinning stays within each
  sector's shelves and the grounded ice just upstream; Weddell reaches furthest inland; Amundsen is
  confined to its narrow coastal shelves.
- `docs/figures/maps_projections_dH.png`: CESM2 runs minus control at 2100/2300, and control drift.
  Interior thickens under every SSP (SMB); margins/shelves thin; by 2300 under ssp585 the whole
  coastal margin thins (the SMB-driven reversal). Control drift shows metre-scale interior speckle.
  Checked a suspected 180-degree seam over the Ross sector: SMB anomaly, melt and thickness differences
  are smooth across x = 0 -- the bands are a smooth field crossing zero plus along-flow streaks.

**5e area-weighted comparison** (`check_ho_relaxfix_historical_timeseries.py`, job 181111485).
Same fixed-coverage grounded mask as 4w/4x (599,285 vertices, 7.563 Mkm², identical to v2).
Old-HO numbers are parsed from the v2 log. Outputs: models/ho_relaxfix_historical_dhdt_
{domain,basins}.png and ..._series.npz.

Whole domain, Gt/yr:

| run | 1996-2001 | 2002-2013 | 2014-2019 | 1996-2019 |
|---|---|---|---|---|
| HO relaxed start | -80 | -30 | -29 | -42 |
| old HO | -222 | -114 | -112 | -- |
| SSA | -104 | -44 | -48 | -60 |
| CPOM | +33 | -25 | -57 | -18 |

- Yearly RMSE vs CPOM, 1996-2019: HO relax 0.0096 m/yr, SSA 0.0110.
- On the 12 years old HO has: old HO 0.0273, HO relax 0.0131, SSA 0.0149. Correlation with
  CPOM stays low (+0.13 HO relax, +0.14 SSA).
- Reading: the relaxed start removes most of HO's excess thinning. 2002-2013 is now close to
  CPOM (-30 vs -25 Gt/yr), and HO relax beats SSA in every period. The start-up transient is
  much smaller but not gone (-80 vs +33 in 1996-2001). After 2014 it loses too little (-29 vs
  -57).
- Per basin, HO relax has the lower yearly RMSE in 8 of 16, mostly the small or noisy basins
  (8, 10, 11, 13) plus 0, 2, 6, 9. SSA is closer in 1, 3, 4, 5, 7, 12, 14, 15.
- Basin 9 (largest loss): both runs ~-85 vs CPOM -65 Gt/yr; neither captures the observed
  acceleration (-32 -> -97).
- Basin 14: HO relax +16 vs SSA +29 vs CPOM +27 Gt/yr, i.e. HO under-thickens there.

### 6.13 HO projections started (2026-10-13)

User approved, after the 5e comparison: ctrl + ssp585 ref to 2100, from the relaxed-start 2019
state (`AIS3_PROJ_HIST_TAG=_relaxfix`, the default).
- Step 1: smoke test ssp585/CESM2-WACCM, 0.2 yr (driver 181116872,
  logs/proj_ho_relaxfix_smoke_ssp585_cesm2.out). It also builds
  models/AIS3_proj_ho_relaxfix_init.nc and ..._baseline.npz once.
- The full runs wait for that build, so that two drivers do not write the same init file. The
  old-run smoke files (AIS3_proj_ho_init.nc, AIS3_proj_ho_baseline.npz, tag '') are untouched.
- Allocation 2026.q4: 1.75 MSU available. Estimate ~38k SU per HO run to 2100.

### 6.14 LARMIP response vs LARMIP-2 (2026-10-13)

LARMIP-2 8 m/yr step results, 17 models (Levermann et al. 2020, ESD 11:35; data from
pik-potsdam.de/~anders/larmip/Experiments/*_BM08_R[1-5].dat). Region order assumed R1 EAIS, R2
Ross, R3 Amundsen, R4 Weddell, R5 Peninsula: the panel order of the paper's Figs 4/10; the text
never defines the numbering. Ours: experiment minus CTRL, ISMIP-convention VAF, mm SLE.

| region | ours y100 | L2 median y100 (p05-p95) | ours y200 | L2 median y200 (min-max) | our rank of 18 at y200 |
|---|---|---|---|---|---|
| East Antarctica | 9.7 | 77.5 (25-250) | 23.1 | 162 (30-506) | 1 |
| Ross | 12.2 | 76.2 (43-147) | 27.5 | 189 (89-468) | 1 |
| Amundsen | 6.1 | 42.0 (13-107) | 11.9 | 76 (-2.5-208) | 2 |
| Weddell | 18.7 | 123 (62-306) | 51.5 | 313 (129-677) | 1 |
| Peninsula | 2.4 | 7.4 (0.3-41) | 3.5 | 15 (2.7-113) | 2 |

- Rank 1 means the smallest response; ours is ~4-8x below the LARMIP-2 median everywhere.
- ISSM JPL / UCI at y200: EAIS 162/359, Ross 110/167, Amundsen 70/174, Weddell 244/333,
  Peninsula 21/48 mm.
- Forcing checked: extra TotalFloatingBmb vs CTRL is +3830 (r1 Weddell), +1440 (r2 EAIS) and
  +3280 (r3 Ross) Gt/yr from the first output, steady over 2019-2050, i.e. 8 m/yr on each
  sector's floating area. So the melt is applied, and the weak response is the ice sheet's.
- Not yet diagnosed. Candidates:
  - linear Budd sliding (p=q=1) with the inverted friction;
  - fixed front;
  - no melt on partially floating elements;
  - how far the shelves actually thin under 8 m/yr with this rheology.
- Caveats:
  - LARMIP-3 sectors are not the LARMIP-2 regions;
  - initial states differ (ours: 2019 end of the historical run);
  - LARMIP-2 files are assumed control-free/control-subtracted (one model, MALI DOE, is noted
    in the paper as drift-corrected).

**6.14 diagnosis** (`check_larmip_response.py`, job 181123202; models/larmip_response_check.png).
Weddell (r1) and Amundsen (r4) step runs vs CTRL; VAF here is ISSM's own, as in 6.9.
- Shelves thin almost completely, i.e. buttressing is mostly removed:
  - Weddell shelf mean H 576 -> 20 m by y200 (92% of the initial shelf area at <= 2 m);
  - Amundsen 318 -> 49 m (76%).
- The grounded ice barely responds:
  - Weddell: grounded speed exp/ctrl x1.14-1.20 within 150 km of the GL, x1.03 beyond;
    100k km² of grounded area lost by y200; GL flux +102 Gt/yr.
  - Amundsen: speed x1.05 at every distance; 21k km² lost; GL flux +17 Gt/yr at y50, falling
    to +0 by y200 (re-equilibrated).
  - Still-grounded ice within 20 km of the GL thins only 8-17 m (median) vs CTRL.
- So the low LARMIP response is not weak forcing or weak shelf thinning. Our grounded ice is
  insensitive to losing its shelves.
- Candidates, in order:
  1. Friction `coupling=3`: effective pressure prescribed and fixed in time. Drag does not fall
     as ice thins towards flotation, so the usual Budd-law GL feedback is off. Its direct
     effect so far looks modest (only 14-29% of near-GL grounded vertices would have flotation
     N < 0.5x the prescribed N), because the ice hardly thins. That is circular, though: with N
     coupled, thinning would itself lower drag.
  2. Linear (viscous) sliding, p = q = 1: speed scales only linearly with the stress change.
  3. Inverted friction is high right at the GL (median C 111-178 within 20 km), which may carry
     the force balance that buttressing carries in other models.
- Test before the LARMIP submission: a 50-yr Amundsen step run (+ CTRL) with N coupled to
  flotation (coupling 2) and C rescaled so that the initial basal drag is unchanged
  (C_new = C_old sqrt(N_old/N_new)); optionally a nonlinear-sliding variant with the same
  drag-preserving rescale.

**6.13 HO smoke passed** (driver 181116872, inner 181121303; exit 0, 97 SU, 1h21).
- Built models/AIS3_proj_ho_relaxfix_init.nc (58 GB) + _baseline.npz. mesh.z in sync;
  2 outputs, t=2019.1-2019.2.
- GroundinglineMassFlux is reported as -1589 Gt/yr on the HO mesh (SSA: +1617), i.e. the
  opposite sign convention in 3D; analysis scripts must not compare its sign across tracks.
- Launched: ho ctrl (181124031) and ho ssp585_cesm2_ref_ref (181124033), to 2100.

**6.12 collapse run progress.** Chunk 2019-2050 done in ~3 h: yearly sub-solves converge
normally (0 iteration-cap hits through 2026). From 2044, tens to ~140 masked floating vertices
are removed per year. Checkpoint AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2050.nc; chunk
2050-2075 running (181122542).

**6.14 coupled-N test launched (2026-10-13).** New option `AIS3_PROJ_FRICTION=coupledN` in
projection_ssp_ssa. Off by default; runs get a `_coupledN` suffix.
- friction.coupling 3 -> 2, i.e. N = max(rho_i g H - rho_w g max(0,-base), 0.07 rho_i g H).
- C is rescaled once at the experiment start (C_new = C_old sqrt(N_old/N_new)) so the initial
  drag is unchanged. Checkpoints carry it on; a resumed chunk asserts coupling 2.
- Runs, 2019-2069: `ctrl_coupledN` (181124133) and `larmip_r4_step_coupledN` (181124134).
- Checks:
  - the first output (t=2019.1) speeds should match the standard CTRL's. This verifies the
    drag-preserving rescale, which assumes ISSM applies the same 0.07 floor under coupling 2;
  - then the Amundsen response at y50 against 4.3 mm (standard) and LARMIP-2's y50 range.

**6.14 coupled-N result (2026-10-14).** Both runs completed to 2069, exit 0; both resumed at 2050
with coupling 2 carried over.
- Rescale at start: grounded N_new/N_old median 2.09 (p5 0.39, p95 10.3); C factor median 0.69.
- Rescale check: first-output (t=2019.1) speed coupledN/standard CTRL median 1.000 (p5 0.957,
  p95 1.007); 94.8% of vertices with vel > 10 m/yr are within 5%. So the initial state is
  preserved.
- CTRL drift is barely changed: -26.4 vs -25.1 mm SLE over 50 yr.

Amundsen response (exp - ctrl, ISSM VAF, mm SLE):

| | y10 | y20 | y30 | y40 | y50 |
|---|---|---|---|---|---|
| standard | 0.68 | 1.60 | 2.60 | 3.51 | 4.30 |
| coupledN | 0.73 | 1.89 | 3.27 | 4.82 | 6.46 |
| ratio | 1.09 | 1.18 | 1.26 | 1.37 | 1.50 |

- LARMIP-2 Amundsen at y50: median 18.9 mm (p17 11.2, p83 37.2, range 1.3-63.4). Only 2 of 17
  models are below 6.5 mm.
- Reading: coupling N to flotation raises the response, and the gain grows with time, i.e. the
  feedback builds. But at y50 it closes only ~15% of the gap to the median. Fixed N is a
  contributor, not the main cause.
- Next suspect: linear (viscous) sliding, p = q = 1.

**6.14 nonlinear-sliding test launched (2026-10-14).** User: go ahead; submit both LARMIP sets
(standard + variant) if the variant is ready in time.
- New option `AIS3_PROJ_FRICTION=coupledN_p3` = coupledN + Budd p=3, q=1, i.e. drag
  C^2 N^(1/3) u^(1/3).
- C is rescaled once so the drag is unchanged at the initial speed:
  C_new = C_old sqrt(N_old u0^(2/3) / N_new^(1/3)), u0 in m/s with a 1 m/yr floor.
- Jobs:
  - smoke ctrl, 0.2 yr (181192078), to check that the initial speeds are preserved and that
    the solve converges;
  - `ctrl_coupledN_p3` (181192079) and `larmip_r4_step_coupledN_p3` (181192080), to 2069.
- Plan: if the smoke passes, launch the full 11-run coupledN_p3 LARMIP set straight away rather
  than waiting for the 50-yr result. ~50k SU; 11 x 200 yr at ~42 h of solve each fits before
  30 Oct only if started now.

**6.14 p3 smoke passed; full coupledN_p3 LARMIP set launched (2026-10-14).**
- Smoke 181192078: exit 0; 0 iteration-cap hits; C factor median 1.37.
- First-output speed vs standard CTRL: median 1.007 (p5 1.000, p95 1.090); 89.5% of vertices
  within 5%. Slightly looser than coupledN, from the 1 m/yr floor and the nonlinearity; accepted.
- The 50-yr pair (END=2069) was cancelled after ~10 min and replaced by full-length runs, since
  their 2019-2050 chunk is identical. Partial execution dirs archived as
  *_cancelled_end2069_20261014.
- Launched, all `AIS3_PROJ_FRICTION=coupledN_p3`:
  - `ctrl_coupledN_p3` to 2219 (181192739);
  - `larmip_r{1..5}_step_coupledN_p3` and `larmip_r{1..5}_ramp_coupledN_p3` (181192740-49).
  Logs: logs/proj_ssa_*_coupledN_p3_2019_full.out.
- The Amundsen y50 comparison will come from these runs' 2050-2075 chunk. If the change does not
  help, stop the set with models/AIS3_proj_ssa_<exp>.STOP and submit the standard set only.

**6.12 collapse run: NaN at 2059, driver idle 23 h (2026-10-14).**
- Chunk 2050-2075 (driver 181122542): sub-solves 2050-2058 ran normally; 2058 removed 264
  vertices + 17 detached.
- At 2059.0, 368 masked floating vertices + 6 detached were removed, and the sub-solve's inner
  job (181130482) hit "Solver residue is NaN" on step 1 and exited 1 after 74 s.
- No lock file, so the driver waited ~23 h (6.10 issue again) until killed by hand. The chunk
  checkpoints only at 2075, so 2050-2058 was lost.
- Fix, collapse runs only (the LARMIP/SSP code path is unchanged):
  - `_solve_watched` submits with waitonlock=0, then polls for the lock file and for the PBS
    usage footer in the inner outlog. Ended-without-lock means failed.
  - Stale lock/outlog/errlog/outbin in a reused run dir are renamed *.prev<epoch> before
    submitting.
  - On failure the chunk ends at s0: it writes history so far and a checkpoint holding the state
    the failed solve started from (collapse at s0 applied), then stops the chain (exit non-zero).
- Failed dir archived as ..._sub2059.0_failed_nan_20261013. Chunk rerun from the 2050 checkpoint:
  driver 181231013. It should save that state at 2059 for diagnosis.

**6.14 coupledN_p3 progress.** All 11 inner solves running; r4 step outbin at 53% of the
standard chunk's size after 4h06, i.e. ~15 min per model year (standard 12.6). 200 yr ~50 h +
queues, so done ~17-18 Oct.

**6.14 coupledN_p3 set status and first results (2026-10-16).**
- 8 of 11 runs complete to 2219. Three lost chunks to slow nodes, not to convergence (at most
  1 iteration-cap hit each):
  - r3 ramp: 2019-2050 killed at its 20 h walltime at 2041.6 (11 steps/h; r3 step did the same
    chunk in 15 h);
  - r4 step: 2075-2100 killed at 16 h, at 2097;
  - r4 ramp: 2100-2125 killed at 16 h, at 2105.4 (3.4 steps/h).
- Both r4 drivers then sat waiting on lock files (6.10 issue) and were killed by hand. Killed
  dirs archived as *_killed_walltime_20261016.
- New `AIS3_PROJ_TIME_FACTOR` (default 3 = unchanged) sets the inner walltime; it is passed down
  the chain. Relaunched with factor 5:
  - r3 ramp from 2019 (181381287);
  - r4 step from 2075 (181381288);
  - r4 ramp from 2100 (181381289).

Response (exp - ctrl, ISSM VAF, mm SLE), standard -> coupledN_p3:

| sector | y100 | y200 | LARMIP-2 step median y100 / y200 |
|---|---|---|---|
| Weddell step | 23.0 -> 38.5 | 58.4 -> (running) | 123 / 313 |
| Weddell ramp | -- | 40.9 -> 81.8 | -- |
| EAIS step | 12.1 -> 23.1 | 26.0 -> 56.9 | 78 / 162 |
| Ross step | 15.3 -> 23.6 | 31.8 -> 58.6 | 76 / 189 |
| Amundsen step | y50: 4.3 -> 7.0 (coupledN alone 6.5) | -- | -- |
| Peninsula step | 3.4 -> 5.7 | 4.6 -> 8.4 | 7 / 15 |

- coupledN_p3 roughly doubles the 200-yr response, but it stays at about 1/3 of the LARMIP-2
  median: low end of the ensemble rather than below it. EAIS and Peninsula are now inside the
  LARMIP-2 range; Ross (58.6 vs min 89) is still below.

**6.12 watched-solve fix.** The collapse rerun 181231013 "failed" on its first sub-solve
although that solve ran fine (exit 0, 10/10 steps). ISSM writes the lock file only if
md.settings.waitonlock > 0 is marshalled, and _solve_watched had set it to 0.
- Fix: keep waitonlock = budget and call solve(..., batch=True), which returns right after qsub;
  then poll as before. The false-failure dir is archived as ..._sub2050.0_nolock_20261014.
- Collapse chunk 2050-2075 resubmitted: 181381388.

**6.12 collapse run, 2026-10-18.**
- Rerun 181381388 ran 2050 -> 2073. 2059 passed this time with the identical removal (368 + 6
  vertices) that went NaN on the first try, so that NaN was not reproducible.
- Yearly removals then grew: 800 (2071), 696 (2072) and 1130 (2073) masked floating vertices
  plus 17-37 detached. Sub-solves took 6-9 min each with 0 iteration-cap hits.
- 2073 went NaN at step 5, after 4 normal steps. The watched solve caught it: chunk ended at 2073,
  history ..._2050-2073_history.npz, checkpoint ..._collapse_2073.nc, chain stopped. Working as
  designed.
- Added one automatic retry of a failed collapse sub-solve, if the driver has time left, before
  ending the chunk.
- Failed dir archived as ..._sub2073.0_failed_nan_20261017. Resumed from 2073: 181480181.

**2026-10-19 status.** coupledN_p3 LARMIP: 9 of 10 runs + ctrl complete to 2219. r3 ramp's 2100-2125
solve was walltime-killed again (27 h at time factor 5; reached 2122.3, ~8 steps/h, 0 iteration-cap
hits; third slow node for this run). The end-year guard stopped the chain with no checkpoint. Run
directory archived (`..._2100-2125_killed_walltime_20261019`); resubmitted from 2100 with
AIS3_PROJ_TIME_FACTOR=7 (37 h per 25-yr chunk). HO projections complete to 2100 (both). The collapse
resume from 2073 (181480181) failed on `checkpoint final_time 2074.0 != 2073.0`: the
failure-path checkpoint stores the failed sub-solve's end time rather than its start (s0); not fixed
here.

### 6.15 Status 2026-10-19

**HO projections complete to 2100** (ctrl, ssp585_cesm2_ref_ref; 8 chunks each, from the
relaxed-start 2019 state). Sea level from 2019, ISSM VAF, mm SLE (negative = sea-level fall):

| year | HO ctrl | HO ssp585 | HO ssp585-ctrl | SSA ssp585-ctrl |
|---|---|---|---|---|
| 2050 | -23.1 | -34.6 | -11.4 | -11.5 |
| 2070 | -38.2 | -66.4 | -28.1 | -28.6 |
| 2100 | -60.9 | -132.1 | -71.2 | -75.1 |

- The forced response (ssp585 - ctrl) of HO and SSA agrees within 5% at every year: SMB-dominated
  mass gain, as in 6.11.
- HO's control drifts faster: -61 vs -39 mm by 2100.

**Collapse run.** Resume 181480181 died at start-up: "checkpoint final_time 2074.0 != 2073.0".
- Cause: the failure path saved the checkpoint with the failed sub-solve's timestepping
  (2073 -> 2074) instead of t1 = 2073. Fixed: the failure path now sets start/final time = t1.
- The 2073 checkpoint was corrected in place (timestepping final_time attr -> 2073.0). The original
  is kept as ..._collapse_2073.nc.orig_finaltime2074.
- Resumed: 181547044.

**coupledN_p3 LARMIP.** 10 of 11 complete to 2219. r3 ramp: 2100-2125 walltime-killed again at
2122 (factor 5 = 26 h; gadi-cpu-clx-0004). The user resubmitted it with AIS3_PROJ_TIME_FACTOR=7
(181546272, logs ..._2100-rerun2.out).

**6.15 collapse run, 2026-10-20.**
- Resume 181547044 ran 2073 -> 2089. Removals kept growing: 1552 (2087), 1827 (2088) and 1898
  (2089) masked floating vertices + 40-214 detached per year. Checkpoint at 2075.
- The 2089 sub-solve failed on both the first try and the automatic retry (deterministic): step 1
  linear solve residue 1.87e-3 > solver_residue_threshold 1e-3, and the recovery solver failed
  too. Not a NaN.
- The chunk ended correctly: checkpoint ..._collapse_2089.nc, with start = final = 2089 (time
  stamp fix verified).
- `check_collapse_pieces.py` (2075 vs 2089 checkpoints): no weakly anchored floating pieces;
  unanchored fragments 16-18 elements, loose vertices 91-100 and sliver elements 835-917 are
  similar in both. So no structural cause was found; the system is just ill-conditioned after
  large removals.
- Collapse runs now use solver_residue_threshold 1e-2 (`AIS3_PROJ_RESIDUE` override, passed down
  the chain). Failed dir archived as ..._sub2089.0_failed_residue_20261020. Resumed: 181685092.

**coupledN_p3 r3 ramp:** 2100-2125 (inner 181546757, factor 7) at ~85% of the expected outbin
after 26 h, i.e. slow but inside its 37 h limit. This run is consistently ~3x slower than the
other sectors on every chunk, so it is the run itself, not one bad node. 4 chunks remain after
this one; at ~30 h each it finishes ~25-26 Oct.

### 6.15 LARMIP analysis: both sets vs LARMIP-2, and linear-response test (2026-10-20)

`config/analyse_larmip.py`; LARMIP-2 8 m/yr files (17 models) now kept in `assets/larmip2/`;
coupledN_p3 series exported to `docs/larmip/AIS3_ssa_larmip_sle_coupledN_p3.csv` (r3 ramp only to
year 81, still running). Figures: `docs/figures/larmip_vs_larmip2.png`, `larmip_lrt_test.png`.
Region mapping assumed as in 6.14. Responses: experiment minus control, ISMIP-convention VAF, mm.

| sector, step | y100 std / p3 | y100 L2 median [p5-p95] | y200 std / p3 | y200 L2 median [p5-p95] | p3 rank of 18 at y200 |
|---|---|---|---|---|---|
| Weddell | 18.7 / 34.8 | 121 [61-302] | 51.5 / **132.4** | 313 [161-628] | 2 |
| East Antarctica | 9.7 / 20.9 | 77 [25-247] | 23.1 / **54.3** | 161 [49-443] | 3 |
| Ross | 12.2 / 20.9 | 75 [42-144] | 27.5 / **54.7** | 189 [96-349] | 1 |
| Amundsen | 6.1 / 13.4 | 42 [13-106] | 11.9 / **29.4** | 76 [23-201] | 3 |
| Peninsula | 2.4 / 4.7 | 7 [0.2-39] | 3.5 / **7.3** | 15 [4.3-63] | 4 |

- coupledN_p3 raises the 200-yr response 2.1-2.6x (sum of sectors 117 -> 278 mm; LARMIP-2 median of
  per-model sums 815 mm). It is inside the LARMIP-2 5-95% range for East Antarctica, Amundsen and
  Peninsula, still below it for Weddell (132 vs p5 161) and Ross (55 vs p5 96). Low end of the
  ensemble, not an outlier any more except Ross.
- Both sets accelerate through 200 yr like the LARMIP-2 median; no saturation.
- Control drift differs: standard -89 mm, coupledN_p3 -56 mm over 200 yr (both subtracted).
- **Linear response holds.** Simulated ramp vs LRT prediction from the step run: ratios 0.96-1.13,
  mostly 1.00-1.07 (standard and coupledN_p3 alike), e.g. Weddell p3 y200 77.0 vs 78.8, East
  Antarctica p3 38.6 vs 37.3, Amundsen p3 22.9 vs 21.7. The ramps run slightly ahead of the linear
  prediction (weak superlinearity), most in Ross/Amundsen standard (1.07-1.12 at y100).

### 6.16 Ross ramp (coupledN_p3) slowdown: shorter chunks (2026-10-20)
- `larmip_r3_ramp_coupledN_p3` 2100-2125 (inner 181546757, factor 7 = 36.75 h): the step rate fell
  to ~3.7 steps/h (~2.7 h per model year) as the ramp reached full strength (2119). Step 239/250 at
  31.7 h, so this chunk should just finish. The next 25-yr chunk would need ~68 h, more than the
  48 h maximum.
- Added `AIS3_PROJ_CHUNK_YEARS` to `projection_ssp_ssa`: it caps each chunk at N years past the
  latest checkpoint (checkpoint names stay integer years), and it is passed on when chaining.
- Created `models/AIS3_proj_ssa_larmip_r3_ramp_coupledN_p3.STOP`, so the driver saves the 2125
  checkpoint without auto-chaining.
- Resubmit: PBS job 181722448 (`config/resub_larmip_r3_ramp_p3.pbs`, `depend=afterany:181546272`).
  A first watcher, run as a shell background task, died with the Claude session, so this is a PBS job
  instead. If the 2125 checkpoint exists, it moves the STOP file to `.STOP.released_<date>` and
  resubmits with `AIS3_PROJ_CHUNK_YEARS=8, AIS3_PROJ_TIME_FACTOR=22`. That gives a 37 h inner
  limit, against ~27 h at 3 steps/h (rate at 21:44 was 3.3 h per model year). If the checkpoint is
  missing, it does nothing.
- Remaining: 2125 -> 2219, 12 chunks of ~1.1 days each plus queue waits (~2 weeks). The
  coupledN_p3 Ross ramp column in `docs/larmip/AIS3_ssa_larmip_sle_coupledN_p3.csv` stops at year
  81 until then. If the rate falls below ~2.5 steps/h, drop to `CHUNK_YEARS=5`.
- Update 2026-10-21 14:40: the resubmit job ran at 00:38. The 8-yr chunks run at full step count
  (80 x 0.1 yr) and are now much faster: inner solve 2125-2133 took 4 h 58 min (linear solver
  16546 s), 2166-2174 took 42 min (linear solver 2161 s). The earlier ~3 steps/h is not
  reproduced, so it is probably node- or iteration-count-related. VAF changes per chunk are smooth
  (+0.29 -> +0.73 mm per 8 yr). Checkpoints up to 2175; 2175-2183 running. The extra 1-yr chunks
  (2149-2150, 2174-2175) come from the 25-yr CHUNK_EDGES still applying with CHUNK_YEARS; harmless.
  Expected completion ~2026-10-22, well before the 30 Oct LARMIP deadline.

**6.15 collapse run, 2026-10-21.** The 2089 resume (181685092, residue threshold 1e-2) failed
on both tries, now with "Solver residue is NaN" on step 1 instead of the residue rejection. So
the 2089 starting state breaks the first SSA stress balance. Nothing was lost: "first sub-solve
failed", and the 2089 checkpoint is unchanged.
- Test job 181804306 (`check_collapse_2089_sb.py`): stress-balance-only solves on the 2089 state:
  - V1 as saved;
  - V2 Picard start velocity zeroed;
  - V3 every unanchored piece and loose ice vertex removed;
  - V4 V2 + V3.
- r3 ramp coupledN_p3 at 2175. The user added `AIS3_PROJ_CHUNK_YEARS` (8-yr chunks, 6.16).

**6.15 2089 cause found (2026-10-21).**
- `check_collapse_2089_sb.py`: stress-balance-only solves on the 2089 state.
  - V1 (as saved) and V2 (zero Picard start) end without error but with max speed 1.2e15 and
    1.7e14 m/yr: rigid-body modes.
  - V3/V4: removing all 113 unanchored ice vertices (pieces with no anchor element, plus ice
    vertices in no fully-ice element) gives max 10 km/yr, median 81.5 m/yr (unchanged).
- So the transient failures (residue 1.9e-3, NaN) came from unanchored pieces that the collapse
  step left in place. They were outside the mask and not touching that year's cut, presumably
  formed by grounding-line migration between years. At the 2089 resume nothing was cut, so no
  clean-up ran at all.
- Fix: apply_collapse now removes every unanchored piece each year, whether or not anything is
  cut. In collapse runs this also removes the ~150 init-detached vertices at the first collapse
  year (negligible mass).
- (`check_collapse_2089_tr.py`, the 1-step transient variant test, failed at marshalling: tf
  loaded as a list. Not needed after V3.)
- Resumed from 2089 with the fix (see qstat). Failed dir archived as
  ..._2089-2100_sub2089.0_failed_nan_20261021.

### 6.17 Melt at the grounding line: LARMIP sensitivity test (2026-10-21)

User approved testing melt on partially floating elements, the remaining suspect after
coupledN/p3 (6.14).
- ISSM's Element::Ismip6FloatingiceMeltingRate returns early unless the element is all
  floating (checked in the ISSM source). So with the ismip6 class, melt_interpolation alone
  cannot put melt into grounding-line elements.
- New test options in projection_ssp_ssa:
  - `AIS3_PROJ_STATICMELT=1` replaces ismip6 by the default class with a fixed floating melt
    field: CTRL's first output (t=2019.1), extended by nearest neighbour across the grounding
    line, + the LARMIP step anomaly;
  - `AIS3_PROJ_MELTINTERP` sets groundingline.melt_interpolation.
  - One 2019-2050 chunk from the initial state only; set_static_forcing now skips tf for
    non-ismip6 classes.
- Standard friction. Pairs (exp names):
  - A `ctrl_staticmelt_nmp` / `larmip_r4_step_staticmelt_nmp` (181844493/95);
  - B `ctrl_staticmelt_sem1` / `larmip_r4_step_staticmelt_sem1` (181844497/98);
  - smoke 181844491.
- Compare Amundsen exp-ctrl at y30: B/A gives the effect of melting the floating part of GL
  elements. A vs the standard run (2.60 mm at y30) checks the fixed-melt approximation.
- First submission (181844491-98) failed at once: the sub-solve set-up read md.basalforcings.tf,
  which the default class lacks. Fixed (tf windowing skipped when there is no tf). Logs archived
  *_tfbug. Resubmitted: smoke 181845516; A 181845517/18; B 181845519/20.

**6.15 collapse 2089 fix confirmed (2026-10-21).**
- Resume 181817133 (remove-all-unanchored fix): 2089 sub-solve ran 10/10 steps, exit 0, no
  iteration-cap hits. It had failed 4x before.
- 2090 sub-solve (181822217) wrote its first output at 15:59, then nothing for 2.5 h (normal ~7
  min per year), so it was killed by hand. The driver detected the end and queued its automatic
  retry (181854438).
- The re-staged run dir lost the killed job's outlog (pyissm cleans the dir on staging), so the
  cause of the stall is unknown.
- Melt test (6.17): smoke 181845516 exit 0; the four 2019-2050 runs are solving.
- coupledN_p3 r3 ramp at 2216.
- Update 2026-10-21 18:40: checkpoints to 2216 (2208-2216 +1.05 mm). Final chunk 2216-2219 running
  (driver 181853335, inner 181853797). Queued `config/launch_larmip_p3_final.pbs` (181855251,
  afterok:181853335, 64 GB because the last export used all 32 GB). It needs the 2219 checkpoint.
  It archives the partial CSV and the two LARMIP figures as `*.partial_r3ramp_y81_20261021.*`, then
  reruns `export_larmip.py --ramp --variant coupledN_p3` and `analyse_larmip.py`.
- Update 2026-10-23: 2219 checkpoint written 2026-10-21 18:45. `launch_larmip_p3_final.pbs` ran (exit 0,
  40 GB): coupledN_p3 CSV now complete for all 10 experiments; analysis rerun. Ross p3 ramp vs LRT
  1.09 / 1.08 / 1.05 at y100 / 150 / 200; overall LRT ratios unchanged at 0.96-1.13. Model description
  corrected: p3/standard step ratio at y200 is 1.99-2.57 (Ross 1.99), so "2.0-2.6", not "2.1-2.6".

### 6.17 LARMIP-3 submitted (2026-10-23)
- Sent to anders.levermann@pik-potsdam.de. Attachments: main set `AIS3_ssa_larmip_sle_coupledN_p3.csv`,
  sensitivity set `AIS3_ssa_larmip_sle.csv`, `model_description.tex/.md`, `larmip_refs.bib`
  (TODOs filled: ehrenfeucht2025, vandalum2025, zhou2024, plus reese2026 for the ISMIP7 protocol),
  `authors.md`, `README.txt`. Model name: ACCESS-AIS3.
- Open questions put to Anders: sector numbering (1 Weddell, 2 EAIS, 3 Ross, 4 Amundsen,
  5 Peninsula), and whether the forced area should follow the grounding line (ours does) or stay at
  the initial shelf extent.

### 6.18 Melt-at-the-grounding-line test: results (2026-10-23)

Test design in the first 6.17 section ("Melt at the grounding line"). All 4 runs 2019-2050
completed, exit 0. Amundsen 8 m/yr step, exp - ctrl, ISSM VAF, mm SLE:

| run | y10 | y20 | y30 | grounded area lost by 2049 |
|---|---|---|---|---|
| standard (ismip6, no melt on partly floating el.) | 0.68 | 1.60 | 2.60 | 7,053 km² |
| A: static melt, no melt on partly floating el. | 0.67 | 1.60 | 2.62 | 7,555 km² |
| B: static melt, SubelementMelt1 | 0.94 | 2.14 | 3.50 | 14,401 km² |
| coupledN (6.14) | 0.73 | 1.89 | 3.27 | -- |
| coupledN_p3 (6.14) | 0.83 | 2.16 | 3.74 | -- |
| LARMIP-2 median (y10 / y30) | 1.1 | -- | 9.4 | -- |

- A reproduces the standard run, so the fixed-melt approximation is valid and B/A isolates the
  melt-interpolation effect.
- Melting the floating part of grounding-line elements raises the response x1.34-1.39 (steady
  over y10-y30, unlike the friction feedback, which grows) and doubles the grounded area lost.
- Size: comparable to coupledN_p3 (x1.44 at y30). If the two multiply, ~x1.9, i.e. ~5 mm at y30,
  still ~half the LARMIP-2 median (9.4 mm).
- Accounted for so far: fixed N, linear sliding and no melt on partly floating elements, together
  roughly x2 of the ~x3.6 gap at y30. The rest is not diagnosed; candidates are high inverted
  basal drag near the GL (inversion compensating), resolution, and the SSA approximation.
- The production runs cannot use this as is: the ismip6 class sets melt only on fully floating
  elements (Element::Ismip6FloatingiceMeltingRate). Using SEM1 with ismip6 melt needs either an
  ISSM source change (drop the IsAllFloating test in that routine, then rebuild) or a fixed melt
  field as in this test.
- LARMIP-3 was already submitted with coupledN_p3 (second 6.17 section). If Anders' answer on the
  forced area requires reruns, SEM1 could be folded in then.

**6.12 collapse run, 2026-10-23.**
- Chunks 2089-2100, 2100-2125 and 2125-2150 complete; the automatic retry rescued 2090 and 2147.
  Chunk 2150-2175 running (driver 182042838), at 2162.
- Removals reach ~3,650 masked floating vertices per year by 2149.
- Sub-solve cost is now very uneven:
  - 7-30 min in easy years;
  - 1-3.7 h when 5-9 of the 10 steps hit the 100-iteration cap (2148: 3h42 of its 4 h limit;
    2152, 2157, 2158: 2.3-2.7 h).
- The collapse inner walltime floor is raised 240 -> 480 min (`AIS3_PROJ_COLLAPSE_INNER_MIN`,
  passed down the chain); the driver's per-sub-solve time need follows it. Applies from the next
  chunk (2175).
- Still on for collapse runs: solver_residue_threshold 1e-2 (6.15).

**6.12 collapse run, 2026-10-24.**
- Chunks 2150-2175 and 2175-2200 complete with no failures. Chunk 2200-2225 running (driver
  182168491), at 2205; sub-solves are fast again (5-24 min, 0 iteration-cap hits in 2200-2204).
- Collapse effect, ISSM VAF, mm SLE (+ = rise), CESM2-WACCM ssp585 ref:

| year | ref - ctrl | collapse - ctrl | collapse - ref | floating area ref / collapse (10^3 km²) |
|---|---|---|---|---|
| 2050 | -11.5 | -11.5 | +0.0 | 1403 / 1395 |
| 2100 | -75.1 | -64.4 | +10.7 | 1417 / 1264 |
| 2150 | -148.1 | -98.3 | +49.8 | 1506 / 1085 |
| 2175 | -160.2 | -81.1 | +79.1 | 1566 / 961 |
| 2200 | -152.0 | -27.6 | +124.5 | 1626 / 817 |

- Collapse adds +11 mm by 2100 and +125 mm by 2200. By 2200 it has removed half the floating
  area (817 vs 1626 thousand km²) and nearly cancels the SMB-driven sea-level fall.

**6.12 collapse run complete to 2300 (2026-10-27).** `ssp585_cesm2_ref_ref_collapse` finished
("complete through 2300"). The last chunk needed one automatic retry (2285). No time gaps
> 1 yr in the history.

| year | ref - ctrl | collapse - ctrl | collapse - ref | floating area ref / collapse (10^3 km²) |
|---|---|---|---|---|
| 2100 | -75.1 | -64.4 | +10.7 | 1417 / 1264 |
| 2150 | -148.1 | -98.3 | +49.8 | 1506 / 1085 |
| 2200 | -152.0 | -27.6 | +124.5 | 1626 / 817 |
| 2250 | -21.3 | +200.4 | +221.7 | 1747 / 438 |
| 2300 | +211.8 | +521.7 | +309.8 | 1863 / 412 |

- Units: ISSM VAF, mm SLE (+ = rise); a final ISMIP-convention export is still to do.
- Collapse adds +11 mm by 2100, +125 mm by 2200 and +310 mm by 2300. By 2300 the collapse run
  has 412 vs 1863 thousand km² of floating ice.
- Total ssp585 + collapse: +522 mm vs CTRL by 2300 (no collapse: +212).

**6.12 collapse: final export and figures (2026-10-27).**
- Collapse history outputs: 559 time points = one per integer year + one at Y.1 per 1-yr
  sub-solve (each sub-solve also saves its first step). No duplicates; every export uses
  integer years only.
- `plot_projections.py --proj` rerun (182327674; the 2026-10-09 CSV, figure and log are kept as
  *_20261009). ISMIP-convention VAF, collapse - ref: +11.0 (2100), +126.8 (2200), +314.6 mm
  (2300). Totals vs CTRL by 2300: collapse +526, no collapse +211.
- New `config/plot_collapse_and_ho.py` -> docs/figures/projections_collapse.png and
  projections_ho_vs_ssa.png.
- HO ssp585 - ctrl (ISMIP VAF, own 2019 state): -11.3 (2050), -69.5 (2100); SSA -11.3, -73.0.
  Control drift by 2100: HO -65, SSA -41.
- **Ross Ice Shelf removed in one year, 2214, mostly by the unanchored-piece rule.**
  - 2357 masked + 9911 unanchored vertices, ~209k km² of floating ice; other years remove 9-65
    unanchored vertices.
  - So that year's mask band evidently separated the rest of the Ross shelf from grounded ice
    (no edge connection to an anchoring element), and the remnant was removed as an iceberg.
  - Physically defensible: a detached shelf drifts away, and it cannot stay in the solve (rigid
    body). But it is a choice of ours, not the mask's, and it should be stated wherever the
    collapse results are used.
  - Not yet verified from the 2213 geometry (no ice_levelset in the history files).

**6.12 collapse: shelves go from the grounding line outward (2026-11-19).** Seen in the animation
(`config/animate_ssa_projection.py` -> docs/figures/anim_ssa_ssp585_cesm2_ref_ref_collapse.mp4).
- The CESM2 ssp585 v2.1 mask flags inner shelves first. Ross: in 2150 it covers the inner shelf off
  the Siple Coast but not the outer shelf, and the run opens a hole there at the grounding line
  while the outer shelf stays, held by lateral contact. That outer remnant is the piece dropped as
  unanchored in 2214.
- apply_collapse removes all masked floating ice up to the grounding line. Ice that ungrounds
  inside the mask goes at the next mask change, so in masked areas the ice front follows the
  retreating grounding line inland. By 2300 there is open ocean upstream of the 2019 grounding line
  on the Siple Coast.
- `config/check_anim_ice_mask.py` (job 184581313) confirms the animation shows the model's ice
  extent. Of 523,156 vertices ever removed, at most ~1,000 hold grounded ice > 2 m in any decade;
  these are ice-free vertices of front elements, where ISSM keeps a thickness. Removed ice >2 m
  (mostly floating), the same front-element effect: 37 vertices (2030) to 20k (2190), 14k (2300).
- The reverse also holds: vertices still inside the ice domain at min_thickness (1 m) grow from
  ~800 (2020) to 100,727 (2300), 86,476 of them grounded. Presumably surface-melt stripping after
  ~2200, not checked. The animation still draws these as ice.
- So in collapse results the holes at the grounding line are imposed by the mask and the
  remove-all-masked-floating-ice rule; they are not a dynamic grounding-line failure. State this
  wherever the collapse run is shown.

### 6.19 HO projections extended to 2300 (2026-10-28)

User decision. ho ctrl and ho ssp585_cesm2_ref_ref resume from their 2100 checkpoints with
AIS3_PROJ_END=2300 (drivers 182423946 / 182423947; logs proj_ho_relaxfix_*_2100-to2300.out),
20 more 10-yr chunks each.
- Recent chunks used 11.4-11.8 h of the 23.2 h inner limit, a 2x margin.
- Estimate ~95k SU per run, ~3-4 weeks wall clock with hugemem queues.
- Allocation 2026.q4: 1.35 MSU available.
- The HO step does not use the collapse runs' watched solve: a failed inner solve still leaves
  the driver waiting out its budget (6.10). Check the chains for stalls.

### 6.20 HO extension: SIGBUS stalls, watched solve, pinned-velocity runaway; chains stopped (2026-11-05)
- Two ctrl inner solves died with SIGBUS (2120-2130 at 2127.6, node clx-0023, 2026-10-30;
  2130-2140 at 2130.4, clx-0015, 2026-10-31). No NaN or residue warnings before; both chunks
  passed on rerun. ssp585 solved through the second crash window, so not a /g/data outage.
  ISSM install unchanged since May. Cause not found. Failed dirs kept as
  `execution/AIS3_proj_ho_relaxfix_ctrl_21{20-2130,30-2140}.failed_sigbus_2026103{0,1}`.
- An ssp585 driver failed once loading `AIS3_proj_ho_relaxfix_ssp585_cesm2_ref_ref_2130.nc`
  (HDF error, 2026-10-30 10:52). The rerun loaded the same file fine: a transient read error.
- Each SIGBUS left the ctrl driver idle until its lock wait timed out (18 h and 43 h).
  `projection_ssp` in ais_0.1.py now uses the SSA script's watched solve (poll for lock file
  or PBS footer), with one retry if driver time allows. In use from 2130 (ctrl) and 2180
  (ssp585): 7 chunks ran clean.
- **Runaway thickness in HO.** max|dH| per chunk grew ~1.2x per chunk from 2 km (2040-2050) to
  41 km (ssp585 2210-2220). 57 vertices > 5 km at 2220. Worst at (-2459, 1284) km, northern
  Peninsula: 875 m (2019) -> 226 km (2220). Another cluster near (-2329, 1052) km.
  Ice > 5 km = 4.6 mm SLE at 2220 (0.2 mm at 2100), similar in ctrl and ssp585. SSA max H
  stays < 4.9 km to 2300.
- Cause: the HO model carries Dirichlet velocities (spcvx/spcvy = observed velocity) on 175,734
  columns. They come from the three Schoof-subdomain pins
  (thin ice, C at lower bound, ice_levelset >= 0; ais_0.1.py ~1596-1653) and the
  domain-boundary pins (~568). The SSA init has none. At the spike vertices vel is constant
  for 150 years (106.6, 16.6 m/yr; vel_base = vel). Where the pinned field converges, H grows
  exponentially because the flow cannot respond.
- The pins hold 1.01 M km2 of grounded ice (8.4% of grounded area) and 36k km2 of shelf at
  observed speed for the whole run. Only 95k km2 of it is in the Peninsula. This affects
  every HO transient (historical relaxfix, projections, HO vs SSA comparison), not only the
  spikes.
- Chains stopped after their current chunks: `models/AIS3_proj_ho_relaxfix_{ctrl,ssp585_cesm2_ref_ref}.STOP`
  (ctrl will stop at 2180, ssp585 at 2230). Remove the STOP file and qsub the launcher to resume.
- Next: map pinned columns against the forced basins; test the HO forward solve with the
  pins released (all, or only on ice-free / thin-ice vertices) before any HO restart.

### 6.21 HO pins: where they are, and whether HO runs without them (2026-11-06)
- Chains stopped as planned: ctrl at 2180, ssp585 at 2230 (STOP files left in place).
- `check_ho_pins_map.py` -> `models/ho_pins_by_basin.csv`, `models/ho_pins_map.png`. Of 175,734 pinned
  columns, 163,897 are on ice (median H 342 m) and 11,837 are ice-free. None are on the domain
  boundary (the ocean buffer leaves no ice there). Share of the obs-speed x H flux within 50 km of
  the GL that sits on pinned columns, by LARMIP-3 sector:
  Ross (3) 64%, Weddell (1) 20%, EAIS (2) 7%, Peninsula (5) 5%, Amundsen (4) 2%.
  ISMIP basin 7 is 62%; basins 12 and 13 have 40-50% of their grounded area pinned.
- `check_ho_unpin.py`: stress-balance-only solves on the HO 2019 state, all converged (exit 0, no NaN).
  - U0 (as is): max 4.2 km/yr; never-pinned grounded speed/obs median 0.95.
  - U1 (on-ice pins released): max 7.0e5 m/yr, 265 vertices > 10 km/yr. Released grounded ice:
    22% > 2x obs, 5% > 10x obs. Never-pinned ice: 26% of vertices change by > 20% vs U0.
  - U2 (all pins released): max 6.7e8 m/yr, 303 vertices > 10 km/yr.
  - Fastest vertices are in the Peninsula, (-2328, 1051), (-2392, 1157), (-2271, 944) km, the same
    area as the thickness spikes.
  - Releasing the pins brings back the extreme speeds they were added for. The HO friction field
    in the pinned regions cannot be used without them.
- So HO's Ross (and partly Weddell) response is largely prescribed; Amundsen is not. For the paper:
  SSA is the projection set; HO is a comparison to 2100 (spikes < 0.2 mm SLE then), strongest
  for Amundsen. HO beyond 2100 needs either a pin-free HO friction inversion or a fix for the
  runaway (e.g. spcthickness on pinned columns) and an explicit caveat.

### 6.22 HO extension resumed; first watched-solve retry; Gadi maintenance (2026-11-09)
- Chains resumed 2026-11-06 (STOP files kept as `*.STOP.used_20261105`). Reached ctrl 2210 and
  ssp585 2270 by 2026-11-08 with no other failures.
- ctrl 2190-2200: the first inner solve (183389044) ended without a lock file; the watched solve
  retried (183399631), which completed and wrote the 2200 checkpoint. The cause is unknown: the failed
  attempt's outlog was lost, because pyissm's launch command runs `rm -rf` on the run directory and
  the stale-file rename put the files inside it. Fixed in both `ais_0.1.py` and `ais_0.1_SSA.py`:
  the lock/outlog/errlog/outbin of an earlier attempt now move to a sibling `<run dir>_attempts/`.
- Next drivers (ctrl 2210-2220, ssp585 2270-2280) queued since 2026-11-08 with "Job would cross
  dedicated time boundary": Gadi maintenance from 04:00 2026-11-10. They start after it.
  Remaining: ctrl 9 chunks, ssp585 3.
- max|dH| per chunk still growing: 29 km in ctrl 2190-2200 (6.20).
- 2026-11-11: after maintenance the ctrl 2210-2220 driver's qsub failed ("qsub: cannot connect to
  server", errno 15010, 08:39); no inner job existed and the watched solve would have waited out its
  43 h budget. Both watched solves now also check `qselect -N <name>.queue`: no job and no outlog for
  three polls = failed submission, and the one retry waits 10 min first. ctrl driver killed and
  resubmitted from the 2210 checkpoint. ssp585 2270-2280 submitted normally and is solving.
- 2026-11-12 16:16: third SIGBUS. ssp585 2280-2290 died at 2288.5 (rank 55, node hmem-clx-0046,
  10.2 h into the solve). The watched solve caught it at once but did not retry: the driver had
  queued 9 h for the inner job, so 19 h of its 45 h were gone and a 23.2 h retry did not fit.
  Failed dir archived as `..._2280-2290.failed_sigbus_20261112` (12 core dumps, 157 GB, kept).
  Resubmitted from the 2280 checkpoint 2026-11-13 (183817808, logs `..._2280-resub.{out,err}`).
- First usable core dump (the October ones are 0 bytes). Crashing rank: SIGBUS, si_code 2
  (BUS_ADRERR) inside MUMPS numeric factorisation (`dmumps_fac_arrow_arrange` /
  `dist_fill_send_buffer`, OpenMP region, called from PETSc `MatLUFactorNumeric` via ISSM
  `SolverxPetsc`). PETSc's signal handler then took a second SIGBUS while lazily resolving `sleep`,
  and that fault address lies in the file-backed mapping of `libISSMCore.so` on /g/data/vk83.
  All 12 cores are from the same node, all in MUMPS arrowhead distribution, two at
  `__atomic_store@plt`. A read-only library page that cannot be paged in gives exactly this
  signal, so the likeliest cause is the node losing access to /g/data/vk83 for a moment, not a
  bug in ISSM or MUMPS (library files unchanged since 2026-05-28). Three different nodes so far
  (clx-0026, clx-0015, clx-0046), so not one bad node.
- Auto-resubmit (2026-11-13): when a failed inner solve cannot be retried inside the driver,
  `projection_ssp` now qsubs a fresh driver for the same chunk (same env plus
  `AIS3_PROJ_RESUB=1`, log `proj_ho_<exp>_<start>-resub1_<stamp>`), which restarts from the same
  checkpoint. At most one per chunk: the flag is never passed to next-chunk drivers. Not done for
  smoke runs or when the STOP file exists. The watched solve also moves `core.*` from a crashed
  attempt into `<run dir>_attempts/` before pyissm's rm -rf of the run dir. Takes effect from the
  next driver start (running drivers keep the old code).
- Configurable paths (2026-11-14). Site settings moved to `config/site.env` (project,
  storage, Python env, ISSM build, ISMIP forcing dir; env vars override), read by
  `config/ais3_site.py`. Repo root is derived from the code location, user from `$USER`.
  Converted: `ais_0.1.py`, `ais_0.1_SSA.py`, `melt_gamma_sweep_ssa.py`, all 88 launchers
  (`cd` + Python + log paths; `#PBS -o/-e` now relative to `config/`), the notebook
  launchers, and the notebooks (regenerated, which also picked up the watched-solve and
  auto-resubmit changes). New `config/submit` = qsub with `-P`/`-l storage` from site.env;
  chunk resubmits pass the same flags. Defaults reproduce the old values exactly (checked by
  running both drivers' preambles from another folder); HO inner jobs now also mount av17.
  Launchers accept submission from `config/` or the repo root (drivers chdir to the root
  before their chain qsub). Checks: bash -n on all launchers, a held qsub for relative log
  paths (deleted), notebook static test 23/23, setup test on both projection notebooks.
  Old copies: session scratchpad (`*.before_site`, `pbs_before_site/`, `notebooks_before_site/`).
- 2026-11-16: HO ctrl complete through 2300 (2300 checkpoint 17:0x; last three chunks ran on the
  site-settings code). HO ssp585 - ctrl (ISMIP VAF, mm, + = rise): 2100 -71, 2150 -124,
  2200 -102, 2250 +57, 2300 +319. SSA at the same years: -73, -143, -148, -19, +211. HO ctrl
  drift 2250-2300: -10 mm/decade (gaining). HO turns positive about 2235, SSA about 2255; HO
  ends 108 mm higher.
- Shelf rheology B differs between the tracks. AIS3_inverted.nc (so the whole HO track) has the
  shelf B of the June models/AIS3_ssa_rheology_floating_inv_lcurve/run_004_1_10_1e-17 (100% of
  226,372 shelf vertices identical). The friction inversions and the SSA track
  (ssa_inverted_solve_budd) use execution_newB_rheology/run_001_1_10_1e-17 via
  load_shelf_rheology_B. newB / run_004: median 0.99, 5-95% 0.78-1.37, max |rel diff| ~2.
  So HO forward runs use a different shelf B from the one their friction field was fitted
  with, and HO vs SSA differences include this as well as the flow approximation.
- Reproducibility check: today's ssa_rheology_floating_inv_lcurve step builds the newB input
  exactly (358 marshalled fields identical; only friction law Budd vs Schoof differs, which does
  not act on a floating-only domain). A from-scratch run of that step reproduces newB.
- Figure docs/figures/projections_ho_vs_ssa_2300.png (plot_collapse_and_ho.py, now plots to the
  last common year; the 2100 version is kept). It computes HO SLE from the base-layer thickness
  history: ssp585 - ctrl 2100/2150/2200/2250/2300 = -69.5/-119.8/-95.9/+62.1/+323.6 mm (the
  ISSM IceVolumeAboveFloatation scalar gives -71/-124/-102/+57/+319). Controls by 2300: HO -254,
  SSA -119 mm. The overview deck's HO vs SSA slide now shows this figure.
- Shelf rheology now from models/ (option 1, 2026-11-16). The June run_004 directory was moved
  (mv -n) to run_004_1_10_1e-17.jun30_before_geometry_fixes; finalize_newB_rheology.py saved the
  newB run as models/AIS3_ssa_rheology_floating_inv_lcurve/run_004_1_10_1e-17/. load_shelf_rheology_B
  in both drivers now reads that file (rheology_lcurve_run); checked bit-identical to the old
  direct load of execution_newB_rheology (226,372 vertices, B identical). ssa_inverted_solve and
  the forward checks read the same file, so all steps now use one shelf B. No existing model file
  changed: AIS3_inverted.nc and the HO track keep the June field (record it as such).
- Schoof friction lineage (HO track only; the SSA track uses Budd p=q=1), 2026-11-17. The
  production field (models/AIS3_ssa_friction_inv_reg_lcurve/schoof_m1qn3_tightrestol_cmax2.0,
  run 2026-08-16) was made by a scratchpad script that no longer exists (no copy anywhere on
  gdata, home or /scratch). Recovered from its .bin:
  - Start C is not C_init=5000 (inversion_worklog.md and ais_0.1_param.py say so; wrong). It is
    the Schoof C giving the same drag, at observed speed, as a Budd p=q=1 field
    (tau = C_b^2 N u), with Cmax 2.0 and m 1/3. Budd parent: execution_constrain_ab_off/
    run_001_10_100_0.0001 (2026-08-12, outbin record 1; the use_constrain_regions A/B "off"
    run, never saved to models/). Exact (5e-15) on 97.9% of grounded vertices; on the other
    2.1% (26,145, continent-wide) the drag was lowered by an unrecorded rule (median x0.8),
    including all 248 vertices where the Budd drag exceeds Cmax*N. The Aug 12-15 Schoof tests
    (budd_corrected_v6 etc.) used the same target drag with their own Cmax.
  - Shelf B: AIS3_param.nc's own (no inverted shelf B loaded; 100% of values match).
  - Pins: constrain_Schoof.exp (2,184 extra vertices). cf101 10, 15 inside PIG_Outline.exp
    (91,166 vertices); cf103 23; cf501 7e-8; no masks. Bounds 50..8000 grounded, 0 floating.
    m1qn3 maxiter 170, maxsteps 800, dxmin 1e-5, dfmin_frac 0.1, gttol 1e-4; restol 1e-3,
    reltol 1e-2, maxiter 300. 48 cores, 36.6 h.
  - Geometry: the grounded surface/base differ from today's driver (and from both Budd
    inversions) at 252,226 vertices, by -1308..+984 m; the values are in no saved model file.
    So the production C was fitted on a different surface from the one ssa_inverted_solve uses
    it with.
  - New step ssa_friction_inv_schoof (ais_0.1.py; launcher launch_ssa_friction_inv_schoof.pbs)
    rebuilds this from pipeline inputs: today's geometry, the Budd field
    run_001_10_100_0.0001 (the SSA track's), the formula above with the drag capped at 0.95
    Cmax*N, and all the settings above. Dry run: 358 of 362 marshalled fields identical to
    production; the rest are C, surface, base and name. Start C from the SSA Budd field: 5.4%
    exact, corr 0.981 vs production (from constrain_ab_off: 97.9% exact). Submitted
    2026-11-17 as schoof_from_budd_cmax2.0 in execution_ssa_friction_inv_schoof/; to be
    judged on RMSE after ssa_inverted_solve (production 70.43) and correlation with the
    production field.
  - Progress check 2026-11-17: by iteration 54 the rebuild's m1qn3 cost was 1.17x production's at
    the same iteration (1.94e5 vs 1.66e5; production ends at 1.40e5). Scored on today's geometry
    (one m1qn3 iteration in ssa_friction_inv_schoof's setup, execution_ssa_friction_inv_schoof_diag/),
    production's final C costs 2.36e5 (velocity term 103: 2.06e5). So the gap is the different
    surface production was fitted on, not the start: on the geometry the HO track uses, the
    rebuild already fits better than the production field (103: 1.58e5 vs 2.06e5).
  - Rebuild finished 2026-11-17 (job 184210576): m1qn3 stopped on dxmin at iteration 136 (genuine
    convergence; production stopped on its function-call budget). 3 h 56 min, 377 SU, 150 GB.
    Cost flat at 1.8326e5 from iteration ~79. Saved to
    models/AIS3_ssa_friction_inv_reg_lcurve/schoof_from_budd_cmax2.0/.
    Grounded RMSE on the inversion subdomain (Vel vs vel_obs, ocean_levelset >= 0, vel_obs > 0):
    production on its own geometry 94.3 (fast >100 m/yr 137.0, slow 53.3); production C on
    today's geometry 124.4 (138.2, 115.1); rebuild 126.1 (142.0, 115.2). So on the same geometry
    the rebuild matches the production field to ~1% (slightly worse in fast ice; its lower total
    cost is the log term). Both today-geometry fields have runaway vertices (max ~16,700 m/yr)
    and double the slow-ice error: a property of today's surface on this subdomain, which
    ssa_inverted_solve's pins handle. Next: ssa_inverted_solve with this field, written to a new
    file, against production's 70.43.
  - ssa_inverted_solve comparison 2026-11-17 (new options AIS3_INVERTED_FRICTION_RUN and
    AIS3_INVERTED_TAG; outputs models/AIS3_inverted_<tag>.nc, AIS3_inverted.nc untouched). Both
    on today's code (newB shelf B), 48 cores, ~3 min each. Grounded RMSE (grounded, ice,
    vel_obs > 0): production field 71.98 (AIS3_inverted_prodC_today.nc); rebuilt field 75.67
    (AIS3_inverted_schoof_from_budd.nc). All-vertex RMSE 83.10 vs 83.59; max speed 12,350 vs
    12,273 m/yr. The shipped AIS3_inverted.nc (June shelf B) has 70.43, so the shelf-B change costs
    the production field ~1.5 m/yr. Both runs pin ~84.7k thin-ice and ~82-83k lower-bound-clamped
    vertices to observed velocity. Verdict: the step rebuilds the HO Schoof field to within ~5%
    (3.7 m/yr) of the production fit on identical code.

### 6.23 HO initialization: drift vs pins, Budd friction without pins, origin of the thickness spikes (2026-11-17/18)
- Control drift 2019-2030 (grounded mass change, Gt/yr), split by HO's pinned columns:
  HO +290, SSA +211. Pinned columns (1.06 Mkm^2): HO +171, SSA +141 -- the same regions drift in
  SSA, which has no pins. Within 20 km of pins: HO +1, SSA +18. Elsewhere, ice < 10 m/yr
  (6.21 Mkm^2): HO +123, SSA +69 (most of HO's extra +79); faster ice: HO -5, SSA -17.
  Away from pins HO basal/surface speed median 0.97; HO/SSA surface speed 1.22 (1-10 m/yr),
  0.99 (10-50), 0.89 (50-200), 0.85 (> 200).
- `check_ho_budd_unpin.py` (job 184316875; inner solves 437 + 447 SU, driver 441 SU): HO 2019 state
  with the SSA track's Budd p=q=1 field and N (models/AIS3_proj_ssa_init.nc), all pins released.
  B2 = SSA field as is (46,556 HO-grounded vertices have C = 0); B2f = those filled from the nearest
  grounded C > 0. Output models/ho_budd_unpin_test.npz.
  - B2 max 3.8e5 m/yr (Schoof U1 7.0e5), 207 vertices > 10 km/yr (U1 265). Same fastest vertices.
  - Never-pinned grounded ice is fine: speed/obs median 0.97, HO-Budd/SSA median 1.12 (1-10 m/yr),
    1.08, 1.02, 0.97 (> 200). Formerly pinned grounded: speed/SSA median 1.02.
  - 2,373 grounded vertices > 3x SSA and > 1 km/yr, in 108 clusters, 80% on pinned columns.
    Grounded RMSE vs obs excluding them: never pinned 124 (SSA 84, HO production U0 83),
    formerly pinned 152 (SSA 89).
  - B2f is worse (never-pinned RMSE 198; fast ice slowed, C = 0 spots median 0.35x obs). The C = 0
    spots are not a problem in B2 (median 1.06x obs). Drop the fill.
- The blow-ups are geometry, not friction. At every large cluster the HO 2019 surface is
  400-2,800 m above SSA's: e.g. Peninsula (-2328, 1051) km H 3,230 m vs 470 m (surface 4,082 vs
  1,322 m); (-608, 304) H 3,670 vs 2,264. Grounded |dS(HO - SSA)| > 300 m: 1,115 vertices, ~1,000 km^2,
  97% pinned; > 1000 m: 33 vertices, all pinned.
- The spikes grow during the HO historical run, not in the inversion or relaxation: (-2328, 1051)
  inverted 100 m (floor), relaxed 97, 1995 105, 2001 1,087, 2007 1,901, 2013 2,583, 2019 3,230
  (~130 m/yr). |dH| > 300 m: 342 vertices in the relaxation, 3,853 in 1995-2019. This is the 6.20
  pinned-column runaway, already under way before 2019: a pinned column cannot speed up as it
  thickens. The pins added for the inversion's thin-ice / lower-bound speeds build the spikes that
  then need the pins.
- So the HO 2019 state cannot be unpinned with any friction field; the test has to be made before
  the spikes form (the 1995 relaxed state), with both the Schoof and the Budd field.

### 6.24 Melt sensitivity: how much do the shelves buttress the SSA initial state? (2026-11-18)
Question: LARMIP shelves thin almost completely but grounded ice barely speeds up (6.14). Ruled
out first: the friction inversion used the same inverted shelf B as the projections; GL mesh
median 1.33 km (p10 0.66, p90 2.66); no friction spike at the GL (C median 128 within 5 km,
118-147 inland). Prescribed N within 5 km of the GL is 1.23x flotation N (does not go to 0).
Initial fast ice within 20 km of the GL runs 8-15% below obs (fits the positive control drift).
Inverted shelf B / temperature-based B (AIS3_param.nc) on floating ice: median 0.66, p10 0.37.

`check_shelf_removal.py` (job 184450907, 13 SU + six 48-cpu solves of ~3 min). Stress balance only
on AIS3_proj_ssa_init.nc: S0 as is; T0 temperature shelf B; X1 all floating ice removed (front at
the GL, unanchored pieces removed as in apply_collapse); P0 / PT0 / P1 the same with coupledN_p3
friction. Output models/shelf_removal_test.npz. Grounded RMSE vs obs: S0 85.2, T0 91.7, P0 85.3,
PT0 94.6 m/yr.
- 3,122 grounded vertices (median 0.9 km from the GL) run > 10x faster with the front at the GL
  (max 3.8e8 m/yr): the cut GL elements. Excluded below (ratios capped at 10 / medians).
- Speed-up from removing every shelf, grounded ice > 100 m/yr, median (flux-weighted, capped):

| sector | 0-5 km X1/S0 | 5-20 km | 20-50 km | 0-5 km X1/T0 | 0-5 km P1/P0 |
|---|---|---|---|---|---|
| Weddell | 1.84 (4.04) | 1.30 (1.84) | 1.03 (1.30) | 2.13 | 2.48 |
| East Antarctica | 1.04 (2.68) | 1.01 (1.13) | 1.00 (1.01) | 1.04 | 1.10 |
| Ross | 2.32 (3.53) | 1.72 (2.24) | 1.29 (1.64) | 3.02 | 2.82 |
| Amundsen | 1.09 (1.49) | 1.02 (1.10) | 1.00 (1.01) | 1.11 | 1.20 |
| Peninsula | 1.11 (1.49) | 1.00 (1.04) | 1.00 (1.00) | 1.13 | 1.23 |

- Temperature-based shelf B adds only 10-30% to the speed-up (most in Ross). Soft inverted
  shelves are a minor factor, not the main one.
- Basal drag / driving stress (tau_b = C^2 N u vs rho g H |grad s|), S0, fast trunks
  (> 1000 m/yr) within 20 km of the GL: Amundsen 0.32 (n 8,096; speed-up 1.08, p3 1.17), EAIS 0.05
  (1.12), Ross 0.00 (n 61; 4.0), Peninsula 0.02 (1.00).
- So Amundsen's trunks are not drag-held (drag carries a third of the driving stress) and still
  barely respond to losing their shelves, stiff or soft. The rest of the resistance must be carried
  inside the grounded ice, most plausibly by the shear margins: grounded B is the temperature-based
  B (inverted/param = 1.00 on 97% of grounded vertices), so the margins are not softened by damage,
  shear heating or fabric, and the friction inversion lowers trunk drag to compensate. A
  margin-held trunk is insensitive to buttressing loss and to thinning near the GL.
- Next test (not run): invert grounded B together with friction in one sector (Amundsen), then
  repeat this shelf-removal test there.

### 6.25 Amundsen joint friction + grounded-B inversion: B does not move (2026-11-18)
`check_amundsen_joint_inv.py` (6.24 follow-up), Amundsen subdomain (LARMIP-3 sector 4) of the SSA
2019 state, 281,654 vertices; sector cuts Dirichlet obs, floating margin Neumann, grounded margin
Dirichlet obs. Run dirs execution_SSA/AIS3_amujoint_*.
- J0 (friction only, production settings, warm start): 39 iterations, stopped on dxmin; cost
  79,173 -> 66,124 (184455778, 4 min, 6.7 SU). Usable reference.
- J1 (C + grounded B, control_scaling_factors [1, 1e6], B free on all grounded ice): stopped on
  dxmin after 3 iterations, cost rising 79,287 -> 79,292. Gradient norm 4.43e6 vs C's 3. 79% of
  |G_B|^2 is in ice < 50 m/yr. Per vertex the raw B gradient is ~4e5 (fast ice) to ~2e6 (slow ice)
  times the C gradient, i.e. B*dJ/dB ~ 1e6 x C*dJ/dC. On floating ice, in the shelf inversion,
  the B gradient norm is 2.45e-7 (scaling 1).
- J1b (scaling [1, 1], B free only where obs > 50 m/yr, 181,886 vertices; 184463909, 3.5 min,
  6 SU): 36 iterations, stopped on dxmin at cost 70,405 (worse than J0). The cf502 term stays at
  70.16 throughout; max |dB|/B0 1.3e-5, median 9e-9. B does not move.
- For one control, m1qn3's first step is independent of the scaling, so this is not a scaling
  choice. Taken at face value, the grounded-ice B gradient says a change of a few Pa s^(1/3)
  (1e-8 relative) would remove the whole cost, and the tiny steps J1 took raised the cost. So
  ISSM's MaterialsRheologyBbar gradient on grounded SSA ice looks wrong in this build (size, and
  possibly sign); it behaves on floating ice. Not confirmed with a finite-difference check.
- The driver (184463045) was stopped by hand just after J1b ended; no forward solves ran. Logs of
  the first driver kept as launch_check_amundsen_joint_inv.pbs.{out,err}.failed_load_20261118.

### 6.26 Softened shear margins do not make Amundsen sensitive (2026-11-18)
`check_amundsen_soft_margins.py` (job 184465808, 3.6 SU driver + 2 inversions and 6 forward solves of
a few minutes; output models/amundsen_soft_margins_test.npz). Same Amundsen subdomain and J0
settings as 6.25. Margins: grounded, obs > 20 m/yr, flow-frame shear strain rate of the observed
velocity > 0.01/yr and > the along-flow strain rate (45,609 vertices, 27,495 km^2, 4.0% of the grounded
area, obs median 294 m/yr). Plain effective strain rate was not usable: the trunks stretch along
flow at a median 0.07/yr. B x 0.7 (M70) or x 0.5 (M50) there, friction re-inverted as J0.

| case | final cost | grounded RMSE | margins | obs > 1000 m/yr | trunk drag/driving | shelf removal: 0-5 km median (flux-wt) | trunks |
|---|---|---|---|---|---|---|---|
| J0 | 66,124 | 99.4 | 172.3 | 333.0 | 0.35 | 1.11 (1.41) | 1.07 |
| M70 | 66,058 | 125.1 | 224.3 | 450.0 | 0.42 | 1.12 (1.51) | 1.07 |
| M50 | 68,403 | 195.9 | 376.9 | 723.3 | 0.35 | 1.13 (1.68) | 1.09 |

- Softening the margins 2.9x-8x barely changes the response to losing every shelf (trunks 1.07 ->
  1.09; 5-20 km 1.02 in all three) and worsens the fit sharply; re-inverted friction does not
  compensate. Stiff margins are not why Amundsen is insensitive.
- With 6.24: the immediate Amundsen response to shelf loss is small whatever the shelf B, margin
  B or friction exponent (p3 1.17). The LARMIP gap there has to come from the transient: thinning
  and grounding-line retreat, where coupledN_p3 (x2.0-2.6) and melt on partly floating elements
  (SEM1, x1.35) are the factors found so far.

### 6.27 HO without pins on the 1995 state: Budd works, Schoof does not (2026-11-18)
`check_ho_unpin_1995.py` (driver 184442665, 253 SU, + two 96-cpu hugemem solves). State:
models/AIS3_ho_relaxed.nc (the historical run's start), loaded as historical_dhdt_tuning does.
HO 1995 surface vs SSA: |dS| > 300 m at 291 vertices, max 601 m (no km-scale spikes yet, cf. 6.23).
All pins released. First attempt (184412401) failed on my own assertion that Schoof C is constant up
the columns: layers 1-14 hold a different C on ~1.46M columns (only the base layer is used).
- B95, SSA's Budd p=q=1 field and N: max 5,489 m/yr, 0 vertices > 10 km/yr; 7 grounded vertices
  > 3x SSA and > 1 km/yr. Fastest at (-838, 985) km, a C = 0 grounded patch (obs ~1,200, SSA ~1,930).
  Never-pinned grounded: speed/obs median 0.98, RMSE 100.9 (SSA 79.8). Formerly pinned grounded:
  speed/obs median 0.69, RMSE 143.8 (SSA 89.6). B95/SSA by SSA speed: 1.17 (1-10 m/yr), 1.14, 1.02,
  0.96 (> 200).
- S95, HO Schoof field: max 2.1e8 m/yr at (-2271, 944) km (Peninsula, H 23 m); 5,147 grounded
  vertices > 3x SSA (2,441 on pinned columns, 142 with H <= 150 m); never-pinned 1-10 m/yr ice runs
  1.64x SSA.
- So the Schoof field needs the pins from the start; the Budd field does not. A pin-free HO track
  is possible with the SSA friction: rerun the HO historical 1995-2019 from AIS3_ho_relaxed.nc with
  the Budd field and no pins, then re-initialize the projections. Expected costs: fit is worse than
  SSA (RMSE 101 vs 80), and formerly pinned regions start ~30% too slow.

### 6.28 Pin-free HO historical run started (2026-11-18)
- New option `AIS3_HIST_FRICTION=budd_nopins` in historical_dhdt_tuning (fresh start only; needs a
  new `AIS3_HIST_TAG`): `use_ssa_budd_friction_no_pins` copies the SSA Budd p=q=1 coefficient and N
  (models/AIS3_proj_ssa_init.nc, coupling 3, limit 0.07) up every column and sets every
  spcvx/spcvy/spcvz to NaN. On resume the checkpoint is checked for default friction and no pins.
- `config/launch_historical_budd_nopins.pbs` (tag `_budd_nopins`; driver 24 cpu / 720 GB instead of
  48 / 1450 GB, which used at most 382 GB). Chain: 184510702 -> 184510703 -> 184510705 -> 184510706
  (afterok), logs launch_historical_budd_nopins_chunk{1-4}.pbs.{out,err}.
- Expected per chunk: solve ~6.5 h / ~1,950 SU (relaxfix chunk 1), driver ~10.5 h / ~750 SU; ~11k SU
  for the four chunks, plus hugemem queue waits.
- Then: compare with the relaxfix run and CPOM dH/dt; if usable, start HO projections from it
  (AIS3_PROJ_HIST_TAG=_budd_nopins).
- 2026-11-19: chunk 1 solve (184526288) runs ~12.5 min/step (relaxfix: ~6.5 min), so ~12.5 h of its
  14 h limit. historical_dhdt_tuning's md.cluster.time raised to 20 h (under the 24 h 96-cpu hugemem
  cap and the 40 h waitonlock); chunks 2-4 pick it up when they start.
- 2026-11-19, chunk 1 (1995-2001) done: driver 184510702 16h24m, 1,181 SU; solve 184526288 12h25m,
  3,577 SU (~4,760 SU per chunk, so ~19k SU for the run, not ~11k). Chunk 2 (184510703) running.
  Against the pinned relaxfix run over the same years (base-layer history files):
  - |dH| > 300 m by 2001: 589 vertices, 6% grounded, 19% on former pin columns (relaxfix: 428, 35%
    grounded, 49% pinned). The largest pin-free changes are on floating ice (Amundsen shelf around
    (-1548, -457) km +458 m; EAIS shelves), which the pinned run has too.
  - Spike sites, dH 1995-2001 pinned / pin-free: Peninsula (-2328, 1051) +989 / +167 m;
    (-2392, 1157) +454 / +168; (-608, 304) +454 / -238; (2192, -1204) +270 / +26; (-1179, -1049)
    +477 / -87; (-443, -532) +831 / -181. The grounded runaway is gone except on thin Peninsula
    ice (H 40-100 m in 1995), which still thickens ~28 m/yr. Watch it in chunks 2-4.
  - Whole-domain grounded mass change (mean since 1995) +389 Gt/yr pin-free vs +419 pinned;
    SLE by 2001 -4.5 vs -5.0 mm.

### 6.29 Melt calibration re-scored with the real J2 target and J4: gamma_0=300 does not survive (2026-11-19)
- Finding: the J2 that picked gamma_0=300 (4c/4k; J1 is ~0 by construction once deltaT is refit) used
  a synthetic target: each basin's observed melt spread over the BFRN bins in proportion to floating
  area (melt_ismip7_calibration.py, ssa_melt_deltaT_refit_recalibrate.py). ISMIP7 ships the real
  target (meltobs/melt_target_term2_v3.nc). Real vs proxy, Gt/yr: bin 8 162 vs 106, bin 9 392 vs 170;
  bins 8-9 hold 51% of the melt (proxy 33%). A uniform-per-area target rewards uniform melt, i.e.
  small gamma_0 with large deltaT. J4 (ocean_observations_data/) had never been used.
- `melt_rescore_real_targets.py` (job 184638901, 1.7 SU, Python melt as ssa_melt_j3_refit.py; no ISSM
  solves; output models/melt_rescore_real_targets.json). The rebuilt proxy J2 reproduces the
  published J2 exactly (sanity check). Shelves: PIG id 110 (5,693 km^2 on the mesh), Dotson id 97
  (8,558 km^2, probably with Crosson). Observed-year TF covers 99-100% of both shelves.

| gamma_0 | J2 real | J2/sigma | J3 | warm-cold ratio | J4 min (PIG 09+12) | PIG 2009 / 2012 (obs 72 / 29) | deltaT range (degC) |
|---|---|---|---|---|---|---|---|
| 300 | 564 | 0.45 | 3062 | 0.04 | 18.2 | 53 / 47 | 0.87 to 12.26 |
| 1000 | 529 | 0.41 | 2943 | 0.08 | 15.6 | 57 / 45 | 0.27 to 5.90 |
| 3000 | 471 | 0.34 | 2775 | 0.16 | 11.1 | 66 / 45 | -0.23 to 2.77 |
| **5538** | 423 | 0.29 | 2780 | 0.20 | **7.9** | 72 / 44 | **-0.67 to 1.58** |
| 8307 | 365 | 0.24 | 2809 | 0.25 | 13.2 | 81 / 46 | -0.93 to 1.04 |
| 11075 | 335 | 0.24 | 2871 | 0.28 | 14.4 | 85 / 44 | -1.12 to 0.65 |
| 15000 | 312 | 0.26 | 2975 | 0.33 | 16.1 | 90 / 43 | -1.28 to 0.28 |
| 25000 | 255 | 0.28 | 3309 | 0.44 | 21.8 | 102 / 42 | -1.52 to -0.05 |

- Real J2 now falls monotonically with gamma_0 (still at the edge of the tested range: even 25000
  puts only 292 Gt/yr in bin 9 vs 392); gamma_0=300 is the worst converged candidate. In sigma units
  every candidate is within the (large) target uncertainty, minimum 8307-11075.
- J4: no candidate reproduces the 2009-2012 PIG drop (all 42-47 Gt/yr in 2012 vs 29): the quadratic
  law under-represents Amundsen variability, as the protocol paper finds. J4 over all years is
  1.6-2.6 sigma for every candidate (structural misfit).
- Protocol Monte Carlo (100k; converged candidates), optimal gamma_0 5th/50th/95th pct:
  J1+J2 5538/25000/25000; J1+J2+J3 5538/25000/25000; J1+J2+J3+J4 1000/15000/25000;
  J3+J4 (sensitivity terms only) 1000/5538/15000 with 5538 winning 46%. gamma_0=300 wins < 1% in
  every combination.
- The protocol's deltaT cap (|deltaT| <= 2 degC) holds only for gamma_0 >= 5538 (3000 has one basin at
  2.77; 300 has 13 of 16 basins above 2, Amundsen 12.3).
- Reading: gamma_0 ~ 5,500 (median deltaT 0.15 degC) is the reference that satisfies the cap, gives the
  best J4 minimum and near-minimum J3, with ~15,000 (= the existing "high melt" g7) as the upper end.
  Caveats: only 9 candidates and the J2 optimum is at the range edge; J3 not sigma-normalised;
  weights still the protocol's. Next: dense gamma_0 grid with history matching (sigma-normalised,
  explicit structural-error term, deltaT capped), then decide whether to re-run the ssp projections.

### 6.30 History-matching melt calibration on a dense gamma_0 grid (2026-11-19)
`melt_history_match.py` (job 184639879, 7.2 SU, 18 min; Python melt; output models/melt_history_match.json).
40 gamma_0 from 300 to 100,000 (log); deltaT refit per basin to present-day melt (bisection on
element totals), with and without a +-2 degC cap. Terms J1 (16 basins), J2 (10 BFRN bins, real target),
J3 (warm-cold basin-mean change, 7 ocean models, sigma from both states in quadrature), J4 (PIG/Dotson,
all observed years). Each term: I = RMS of (model - target)/sqrt(sigma^2 + (delta |target|)^2 + 1^2);
ruled out if any I > 3; pseudo-posterior exp(-sum I^2/2) on a log-uniform prior.

| case | not ruled out | posterior 5 / 50 / 95 | MAP |
|---|---|---|---|
| refit, delta 0.1 | none | 1451 / 3093 / 4874 | 3774 |
| refit, delta 0.3 | 300-7948 | 404 / 2547 / 6782 | 4381 |
| refit, delta 0.5 | 300-14422 | 346 / 2264 / 8507 | 4381 |
| capped, delta 0.1 | none | 1733 / 2696 / 3766 | 3252 |
| capped, delta 0.3 | 300-7948 | 713 / 2167 / 5200 | 2802 |
| capped, delta 0.5 | 300-14422 | 409 / 1865 / 6869 | 2414 |

Protocol MC on the same grid (median-normalised, U[0,1] weights): refit 3252 / 12427 / 74237; capped
2802 / 3774 / 63964.
- J3 sets the upper limit: I(J3) is lowest at gamma_0 ~3,000-3,800 (2.34) and grows fast above ~9,000
  (4.4 at 15,000, 28 at 100,000), although the median warm-cold ratio is still < 1 there (0.38 at
  19,400, 0.92 at 100,000): large gamma_0 overshoots some model/basin pairs while undershooting
  others. The quadratic law cannot match the spread of sensitivities.
- The deltaT cap binds below gamma_0 ~4,400 (Amundsen); with the cap, J1 and J4 penalise small gamma_0.
- With 10% structural error nothing survives: the law cannot fit all four terms; the structural
  term is needed.
- J2 in sigma units is weakly informative (large bin uncertainties, bin 0 +- 296 Gt/yr), mildly
  favouring large gamma_0. J4 under the refit is ~2 sigma for every gamma_0 (2009-2012 PIG drop not
  reproduced).
- Recommendation: reference gamma_0 ~ 4,000 (deltaT within ~+-2 degC, near-minimum J3; protocol capped MC
  median 3,774); range ~1,000-8,000. Warm-cold response ~0.17 of the ocean models' median vs 0.03 at
  gamma_0=300. The existing "high melt" gamma_0=15,000 is ruled out at delta 0.3 (survives at 0.5).
- Not done: rerunning the ssp projections with the new calibration (and the pin-free HO historical,
  which uses gamma_0=300).

### 6.31 New SSA melt calibration adopted: gamma_0 = 4,000 (tag _hm4000) (2026-11-19)
- `melt_calibration_hm_refit.py` (job 184641476, 0.8 SU): per-basin deltaT at gamma_0 = 1000 / 4000 / 8000
  matching every basin's present-day melt exactly (exact vertex-basin totals, simultaneous bisection);
  models/melt_calibration_hm.json, members hm1000 / hm4000 / hm8000.
  - hm4000 deltaT -0.44 .. 2.21 degC (median 0.30; only Amundsen above 2); hm8000 -0.93 .. 1.06;
    hm1000 0.28 .. 6.04 (6 basins above 2).
  - hm4000 by basin: -0.439 0.052 0.250 0.884 0.642 0.770 -0.060 -0.098 1.095 2.209 0.560 -0.107 -0.052
    0.519 0.041 0.351.
- `ais_0.1_SSA.py`: new `AIS3_SSA_CALIB` ('' = production gamma_0=300; '_hm<g>' = a member of
  melt_calibration_hm.json). Appended to AIS3_melt_final_ssa, AIS3_historical_1995_2019_SSA (refuses to
  overwrite a tagged file), AIS3_proj_ssa_init/baseline and every projection experiment name; passed on
  when projection chunks chain; requires AIS3_PROJ_MELT unset. Defaults unchanged. SSA notebooks
  regenerated (24/24 static OK).
- Submitted: melt step 184641641 (AIS3_STEPS=melt_gamma_tuning_ssa, 190 GB) -> SSA historical 1995-2019
  184641642 (afterok, 48 GB driver), both with AIS3_SSA_CALIB=_hm4000, launch_ais_0.1_SSA.pbs, logs
  launch_ais_0.1_SSA_{melt,hist}_hm4000.pbs.{out,err}.
- Next: compare the _hm4000 historical with the production one and CPOM; then projections
  (AIS3_SSA_CALIB=_hm4000: ctrl first, then ssp585 CESM2).

### 6.32 Friction-law calibration runs: SSA historical 1995-2019 with four friction laws (2026-11-19)
- New `AIS3_SSA_FRICTION` in ais_0.1_SSA.py (historical_dhdt_tuning_ssa only): '' = production Budd p=q=1
  with prescribed Ehrenfeucht N (coupling 3); p1_flotN, p3_fixedN, p3_flotN (p5_flotN allowed) = exponent
  p (q=1) with prescribed N or flotation N (coupling 2). `rescale_friction_drag_preserving` rescales C once
  at the 1995 start so the initial drag is unchanged (C_new = C_old sqrt(N_old u0^(1-1/p) / N_new^(1/p)),
  as projection_ssp_ssa's coupledN/coupledN_p3), so all variants fit present-day velocity equally and
  differ only in their transient response. Tag _<variant> appended to the historical run name/file.
- 2x2 design, all with the new melt (AIS3_SSA_CALIB=_hm4000): production p1 fixed N = 184641642;
  p1_flotN 184642396, p3_fixedN 184642397, p3_flotN 184642398 (logs
  launch_ais_0.1_SSA_hist_hm4000[_<variant>].pbs.{out,err}).
- Planned scoring: velocity change 2001 -> 2019 against MIPKIT vx/vy_timeseries (2001-2020, 16 epochs),
  focused on Amundsen, and CPOM dH/dt by basin and period on the fixed-coverage mask (6.13), with the
  history-matching approach of 6.30.
- 2026-11-20: all four inner solves finished (exit 0, 27 GB outbin each): production-friction _hm4000
  5.0 h / 512 SU; p1_flotN 4.5 h / 458 SU; p3_fixedN 13.9 h / 1,429 SU; p3_flotN 7.0 h / 716 SU. All four
  48 GB drivers were OOM-killed (exit 137) loading the results (~770 SU of driver time). New
  `AIS3_HIST_LOAD_ONLY=1` in historical_dhdt_tuning_ssa loads a finished run (lock + outbin required)
  instead of resubmitting. Load-only jobs at 190 GB: 184686628-31; comparison requeued after them
  (184686632). Old driver logs kept as *.oom_driver_20261120.
- Rescale check: grounded flotation-N / prescribed-N median 2.01; C factor median 0.71 (p1_flotN),
  1.64 (p3_fixedN), 1.41 (p3_flotN).
- Correction: the OOM-killed drivers had started saving: each left a 4 GB AIS3_historical_1995_2019_SSA_hm4000*.nc
  without TransientSolution (truncated write). Renamed to *.incomplete_oom_20261120.nc; the first load-only jobs
  refused to overwrite them (logs *.refused_exists_20261120). Load-only resubmitted at 190 GB (184687874-78);
  comparison (ISSM_DIR fix) requeued after them (184687879).

### 6.33 Friction-law comparison against CPOM and the 2007-2019 speed-up: not discriminating (2026-11-20)
`compare_ssa_historical_friction.py` (job 184738367, 3.8 SU; models/ssa_historical_friction_scores.json).
Load-only jobs 184687874-78 saved the four complete historical files (27.6 GB each). Two failed comparison
attempts kept as *.failed_issmdir_20261120 / *.failed_veltime_20261120 (2019 mosaic dated 2019.09, runs end 2019.0).
Fixed-coverage mask 599,285 vertices / 7.563 Mkm^2 (identical to 6.13).

| run | domain Gt/yr 96-01 / 02-13 / 14-19 | RMSE | Amundsen (b9) | RMSE b9 | Amundsen speed 2019/2007 (obs 1.180) | RMS log speed ratio |
|---|---|---|---|---|---|---|
| CPOM | +33.0 / -24.5 / -56.9 | | -32.4 / -65.3 / -97.1 | | | |
| prod300 (p1 fixedN) | -103.5 / -44.4 / -47.9 | 0.0110 | -89.1 / -88.6 / -73.2 | 0.106 | 0.990 | 0.192 |
| hm4000 (p1 fixedN) | -105.7 / -49.7 / -53.6 | 0.0112 | -90.0 / -91.2 / -76.4 | 0.108 | 0.994 | 0.191 |
| hm4000 p1_flotN | -121.3 / -68.3 / -76.7 | 0.0129 | -95.3 / -103.2 / -95.9 | 0.121 | 1.031 | 0.189 |
| hm4000 p3_fixedN | -126.3 / -25.5 / -23.8 | 0.0129 | -84.8 / -80.5 / -66.1 | 0.102 | 0.992 | 0.236 |
| hm4000 p3_flotN | -146.3 / -41.8 / -41.4 | 0.0142 | -90.4 / -89.6 / -78.4 | 0.108 | 1.005 | 0.209 |

- The melt recalibration changes 1995-2019 very little: the historical ocean is the constant climatology, so
  melt sensitivity to TF change has nothing to act on.
- Every run shares the two dominant misfits: a 1996-2001 start-up loss of 100-150 Gt/yr (CPOM +33) and no
  Amundsen acceleration (observed speed +18%, models +/-3%; observed loss growing -32 -> -97, models flat
  or declining). The observed change was ocean- and shelf-driven, which these runs do not contain, so the
  record cannot discriminate the friction laws. The laws do differ substantially (2014-2019 domain loss -24 to
  -77 Gt/yr, Amundsen -66 to -96), but each metric favours a different law, and the shared misfits in the temporal
  pattern are larger than the differences between laws, so no law is consistently better. The friction law
  matters (factor ~3 here, x2.0-2.6 in LARMIP); this record just cannot tell which is right.
- LARMIP-3 main set (coupledN_p3 = p3_flotN): neither contradicted nor supported; no case for resubmission.
- Next, if pursued: historical runs with time-varying ocean forcing (ISMIP7 contemporary OCX pathways, incl.
  the "varying" Amundsen thermocline pathway) with the new melt law, scored from 2002 to avoid the start-up
  transient. Only then can friction law and melt calibration be tested against the Amundsen speed-up.

### 6.34 OCX-forced SSA historical runs ('vary' ocean pathway) (2026-11-20)
- `ais_0.1_SSA.py`, historical_dhdt_tuning_ssa: new `AIS3_SSA_OCEAN` (main/cold/warm/vary) applies the ISMIP7
  contemporary ocean pathway (OCX/ocean/<pathway>/v2) as an anomaly on the calibration climatology,
  TF(t) = TF_Zhou + (OCX_<pathway>(t) - OCX_main(t)), annual columns at year + 0.5 (set_ocx_ocean_forcing);
  tag _ocx<pathway>. After 1995 'main' equals the climatology; 'vary' has the 16-yr thermocline cycle in
  Amundsen/Getz/Aurora: at 510 m the anomaly is up to about +-2 degC, warm 2008, cold 2000 and 2016, zero in
  2004/2012, ~121k grid cells.
- New `AIS3_HIST_SUBMIT_ONLY=1`: the driver submits the inner solve and exits (the 190 GB drivers of 6.32 cost
  more idle than the solves); results are loaded afterwards with AIS3_HIST_LOAD_ONLY=1.
- Submitted (submit-only, 96 GB): prod300 vary 184749304; hm4000 vary 184749305; hm4000 p1_flotN / p3_fixedN /
  p3_flotN vary 184749307-09 (logs launch_ais_0.1_SSA_hist<tag>.submit.{out,err}).
  compare_ssa_historical_friction.py now also scores the *_ocxvary runs (AIS3_SCORE_TAG for a new output name).

### 6.35 v2 projection set: stage A launched, SMB-elevation feedback added (2026-11-20)
Plan: docs/projection_experiments.md, "Next set (v2)". v2 reference = `AIS3_SSA_CALIB=_hm4000` +
`AIS3_PROJ_FRICTION=coupledN_p3` (C rescaled once at 2019 from the _hm4000 historical, as for LARMIP-3).
- Measured October costs (inner + driver, retries included): ~2.5-3k SU to 2100; to 2300 ctrl 6.4k,
  ssp585 MRI 8.5k, ssp126 CESM2 10.6k, ssp585 CESM2 16.6k, high melt 18.2k, collapse 17.3k. The idle
  32-cpu / 128 GB driver is 40-50% of each. LARMIP coupledN_p3 runs cost 1.14-2.0x their standard twins.
- Smoke 184750273 (ctrl, 0.2 yr, 13 SU): built models/AIS3_proj_ssa_init_hm4000.nc and
  AIS3_proj_ssa_baseline_hm4000.npz from AIS3_historical_1995_2019_SSA_hm4000.nc. gamma_0 4000, deltaT
  -0.439..2.209 (= hm4000); coupledN_p3 rescale grounded N_new/N_old median 2.09, C factor median 1.372.
  VAF +1.09e10 m^3 over the 0.2 yr; no NaN.
- Stage A submitted (all to 2300; logs config/logs/proj_ssa_<exp>_2019.{out,err}):
  ctrl_coupledN_p3_hm4000 184750757; ssp585 CESM2 184750758, MRI 184750759; ssp126 CESM2 184750760,
  MRI 184750761; ssp534-over CESM2 184750762, MRI 184750763 (first ssp534-over runs; TF and dEBM2
  acabf present to 2299/2300 for both); ssp585 CESM2 collapse 184750764. Names end _coupledN_p3_hm4000.
- New `AIS3_PROJ_SMBFB=1` (+ `AIS3_PROJ_SMBFB_GCM` for ctrl) in projection_ssp_ssa: md.smb = SMBgradients,
  SMB = smbref + dacabfdz (s - href); href = init-state surface (models/AIS3_proj_ssa_href<calib>.npy);
  dacabfdz (dEBM2-8000m, same grid as acabf) annual from the scenario with SMB forcing, else the GCM's
  historical 1995-2014 mean (models/AIS3_proj_ssa_dsmbdzclim_<gcm>_dEBM2-8000m_1995-2014.npy). ISSM's
  SmbGradientsx has no clamping or unit conversion; pyissm scales smbref and b by 1/yts. Collapse
  sub-solves window smbref/b_pos/b_neg like mass_balance. History files of feedback runs hold `surface`.
  CESM2 dacabfdz, m ice eq/yr per km, p1/p50/p99: 2000 -0.59/0.00/+0.53; ssp585 2100 -0.76/0.00/+1.54;
  2275 -0.49/0.00/+3.71.
- Chain hand-off now also passes AIS3_PROJ_SMBFB, AIS3_PROJ_SMBFB_GCM, AIS3_PROJ_MELTINTERP and
  AIS3_ISSM_DIR (MELTINTERP was dropped after the first chunk before, which would have broken any
  multi-chunk SEM1 run).
- Feedback smokes (0.2 yr): ctrl_coupledN_p3_smbfb-cesm2_hm4000 184750806,
  ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000 184750807. Check: config/check_smb_feedback_smoke.py
  (ctrl history: SMB - SMB0 = b (s - href) per vertex).
- 2026-11-21: both feedback smokes exit 0 (7.4 SU each); feedback term zero at the start. Check PASS once
  corrected for timing: ISSM evaluates SMB at the start of each step, so the SMB written at output k uses the
  surface of output k-1. At t=2019.2: SMB - SMB0 vs b (s_2019.1 - href) slope 0.999999, max residual 3.0e-7
  m/yr (float32), feedback term up to 5.3e-3 m/yr. (The first check compared with the same output's surface:
  slope 0 / 0.52, a false FAIL; script fixed.)
- Stage B submitted (to 2300): ctrl_coupledN_p3_smbfb-cesm2_hm4000 184791719, ctrl_..._smbfb-mri_hm4000
  184791720, ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000 184791721, ssp585_mri_... 184791722.
- Stage A progress 2026-11-21: all chains running; checkpoints to 2050 (ctrl, ssp585 MRI, ssp534-over MRI),
  2075 (ssp126 MRI), 2100 (ssp585 / ssp126 CESM2), 2125 (ssp534-over CESM2).
- Collapse chain stopped at 2071: the 2071 sub-solve failed twice within 1 min on gadi-cpu-clx-2260 with an
  MPI/UCX network fault ("UD endpoint ... unhandled timeout error", SIGABRT), not a solver failure; 2058 and
  2066 had each failed once and passed on retry. Run dir archived as ..._2050-2075_sub2071.0_failed_ucx_20261121;
  resumed from the 2071 checkpoint (184791699, logs ..._2071-resume.{out,err}).
- au88 available 632.8k SU (2026-11-21).
- 2026-11-21: OCX-vary solves: prod300 5.4 h / 522 SU; hm4000 5.1 h / 488 SU; hm4000 p1_flotN 4.4 h / 426 SU (all
  exit 0). The submit-time watcher was killed with the session; load-only jobs submitted by hand
  (184791760, -61, -63). hm4000 p3_flotN running at ~5.6 min/step (climatology run: ~1.75), at step 182/240 after
  17 h. hm4000 p3_fixedN ran at ~80 min/step (climatology run: ~3.5 min/step): 13 of 240 steps in 17 h, so it
  could not finish within its 48 h limit; stopped by hand (184750375, ~1,650 SU). p=3 with fixed N is numerically
  intractable once ocean variability reaches the grounding zones (with N held fixed, near-flotation ice responds
  very nonlinearly): a practical argument against that law. Comparison (AIS3_SCORE_TAG=_ocx) queued after the
  four loads.
- Pin-free HO historical (6.28) complete: chunk 4 (2013-2019) exit 0, driver 1,170 SU.
- hm8000 member started: melt step 184792047 (AIS3_STEPS=melt_gamma_tuning_ssa, AIS3_SSA_CALIB=_hm8000) ->
  historical submit-only 184792049 (afterok; logs config/launch_ais_0.1_SSA_{melt,hist}_hm8000*). Load-only,
  then ctrl + ssp585 CESM2 with _hm8000, still to do.
- MRI attribution (v2, to 2300): ssp585_mri_ref_oceanonly_coupledN_p3_hm4000 184792079, ..._smbonly_... 184792083.
- New `AIS3_PROJ_DRIFTCORR=<ctrl experiment>`: fixed SMB flux correction = minus that ctrl's
  (H(2020.0) - H(2019.1)) / 0.9 yr on every vertex with ice at 2019.1; refuses a ctrl with other friction or
  calibration tags; name _driftcorr; passed down the chain. v2 ctrl initial drift: VAF +110.9 Gt/yr (2019.1-2020),
  +133.6 Gt/yr (2020-2030); correction -113.5 Gt/yr grounded, +242.7 Gt/yr floating (shelves thin at the start).
  Smoke 184792610 (6 SU): VAF change over 0.2 yr +2.8e8 m^3 vs +1.09e10 without (97% of the drift removed).
- Drift test submitted (to 2150): ctrl_coupledN_p3_driftcorr_hm4000 and
  ssp585_cesm2_ref_ref_coupledN_p3_driftcorr_hm4000 (job IDs in config/logs/proj_ssa_<exp>_2019.out).
- SEM1 not started: production ISSM is commit 40c764c7 (spack, vk83); the local /g/data/au88/jh7060/ISSM tree is
  99 commits past it on a diverged branch and Element.cpp differs, so the SEM1 build must be 40c764c7 + the
  one-line patch with the production dependencies. Waiting on that decision and on the deltaT refit choice.

### 6.35 Pin-free HO historical vs pinned vs CPOM (2026-11-21)
`compare_ho_pinfree_historical.py` (job 184792081, 1.4 SU; models/ho_pinfree_historical_scores.json); fixed-coverage
mask as 6.13/6.33. Gt/yr 1996-2001 | 2002-2013 | 2014-2019, yearly RMSE (m/yr):

| region | CPOM | HO pinned (relaxfix) | HO pin-free (budd_nopins) | RMSE pinned -> pin-free |
|---|---|---|---|---|
| domain | +33.0 / -24.5 / -56.9 | -80.3 / -29.8 / -29.1 | -32.5 / +7.0 / -0.6 | 0.0096 -> 0.0083 |
| Amundsen (9) | -32.4 / -65.3 / -97.1 | -86.9 / -90.9 / -73.9 | -60.1 / -63.3 / -47.7 | 0.1045 -> 0.0893 |
| Filchner-Ronne (14) | +16.1 / +31.6 / +29.3 | +6.9 / +16.3 / +22.8 | +40.7 / +41.7 / +42.7 | 0.0146 -> 0.0169 |
| Ross (7) | -12.0 / +1.9 / +19.2 | -9.2 / -1.2 / +3.5 | +14.8 / +20.0 / +23.6 | 0.0122 -> 0.0196 |
| basin 0 | +8.0 / +9.3 / +20.5 | -5.1 / +3.2 / -1.9 | -7.7 / +1.1 / -3.7 | 0.0323 -> 0.0354 |

- Pin-free: best whole-domain fit of any historical run so far (SSA 0.0110); start-up loss -80 -> -33 Gt/yr;
  Amundsen much closer early. Ross and Filchner-Ronne (the formerly pinned regions, which start ~30% slow
  with the SSA Budd field, 6.27) now thicken too much. Thickness extremes 1995-2019: max |dH| 1,141 m (pinned
  3,132), > 1 km at 8 vertices (pinned 38), > 300 m at 3,009 (3,853).
- Like every run it misses the late Amundsen acceleration (-48 vs -97 in 2014-2019; constant ocean).
- Next for HO: forward-only friction updates in the formerly pinned regions to bring their speeds to observed.
- Note: other jobs on the account (another session): SSA _hm8000 historical and _hm4000 projections
  (ctrl / ssp585 coupledN_p3 driftcorr, ssp585 collapse coupledN_p3).
- SEM1 set up (2026-11-21). Recommendation agreed: build and refit now, decide on the ~70k SU SEM1 runs once
  the v2 ocean term is known.
  - ISSM: `git archive 40c764c7` (= production spack issm-git.2026.05.18) -> /g/data/au88/jh7060/ISSM-40c764c7-sem1;
    one-line patch in Element::Ismip6FloatingiceMeltingRate: `!this->IsAllFloating()` -> `this->IsAllGrounded()`
    (melt on fully and partly floating elements; fully grounded still skipped; original kept as
    Element.cpp.orig_40c764c7). Built by config/build_issm_sem1.pbs (184795521, 1.9 SU, 3.5 min): sources the
    recorded spack-build-env.txt with the prefix swapped in SPACK_STORE_RPATH_DIRS and the recorded configure
    args; CXXFLAGS -fiopenmp -O3 -DNDEBUG -fPIC -fp-model precise (as production). issm.exe RPATH = its own
    install/lib, and libISSMCore resolves there even with production lib on LD_LIBRARY_PATH. Installed file set =
    production's apart from .la/.md5. Marker install/SEM1_PATCH.
  - Calibration: config/melt_calibration_hm_refit_sem1.py (184795378, 0.5 SU) -> models/melt_calibration_hm_sem1.json,
    member hm4000sem1. Element-integral form (mean melt x area x floating fraction, as SubelementMelt1's
    (1-phi) fmb + phi gmb); target = each basin's no-SEM1 total at hm4000. Sanity: no-SEM1 element integral 863.4
    vs stored 865.0 Gt/yr (max basin diff 0.37). 67,332 partly floating elements (42,292 km^2 floating). SEM1
    without refit: +6.1% (basins +3.1 .. +19.8%). Refit deltaT -0.486 .. 2.101, shifts -0.150 .. -0.007 degC;
    basin errors 0.0000 Gt/yr.
  - Driver: AIS3_SSA_CALIB=_hm<g>sem1 requires $AIS3_ISSM_DIR/SEM1_PATCH and sets melt_interpolation
    SubelementMelt1 in historical_dhdt_tuning_ssa and every projection chunk; members looked up in all
    models/melt_calibration_hm*.json; Subelement* interpolation with ismip6 melt refused on an unpatched build.
  - Submitted: melt step 184796538 -> historical submit-only 184796539 (both AIS3_SSA_CALIB=_hm4000sem1,
    AIS3_ISSM_DIR=.../ISSM-40c764c7-sem1/install); then load-only and a ctrl smoke (patch check: same total
    floating melt as the _hm4000 smoke, melt now on partly floating elements).
- hm8000: melt step done (11.5 SU); historical inner solve 184794259 running. Chained: load-only 184796585 ->
  ctrl smoke 184796586 (builds AIS3_proj_ssa_init_hm8000.nc) -> ctrl_coupledN_p3_hm8000 184796587 and
  ssp585_cesm2_ref_ref_coupledN_p3_hm8000 184796588 (to 2300).
- 2026-11-21 evening, incidents and fix:
  - Three stage-A 2050-2075 inner solves (ctrl, ssp585 MRI, ssp534-over MRI) died after ~1 min on node
    gadi-cpu-clx-0145 with the MPI/UCX fault ("UD endpoint ... unhandled timeout error", SIGABRT). Their drivers
    (pyissm's blind lock-file wait) sat idle 14-17 h (~3k SU). Drivers 184763380/184763555/184767507 deleted; run
    dirs archived as *_2050-2075_failed_ucx_clx0145_20261121; resumed from the 2050 checkpoints: ctrl 184813567,
    ssp585 MRI 184813568, ssp534-over MRI 184813569 (logs ..._2050-resume.{out,err}).
  - ssp126 MRI 2075-2100 (inner 184768503 on gadi-cpu-clx-2050) stalled: outbin ~54% of a full chunk after 14 h
    and no write for 2 h with 48 cores busy (previous chunk 3 h). Deleted with its driver 184768454; dir archived
    as *_2075-2100_stalled_clx2050_20261121; resumed from 2075 (184813632). If it stalls again, it is the
    state, not the node.
  - Fix in ais_0.1_SSA.py: every projection chunk now uses the watched solve (`_solve_watched`, collapse only
    before): an inner job that ends without a lock file is detected within minutes and retried once after
    10 min; a second failure stops the chain at the last checkpoint. Chunks already running keep the old wait.
  - SEM1: historical inner solve is 184799625; chained load-only 184813499 -> ctrl smoke 184813500.
- 2026-11-22 01:00 check: no chain failed since the restarts. hm8000 load + smoke passed (gamma_0 8000, deltaT
  -0.930..1.058; same coupledN_p3 rescale as hm4000); ctrl/ssp585 hm8000 chains running. SEM1 historical loaded
  (184813499, exit 0); SEM1 ctrl smoke passed (184813500).
  - SEM1 patch active: grounded vertices carrying ismip6 melt at 2019.1: 26,517 with hm4000sem1 vs 40 with hm4000.
  - ISSM's Tria::TotalFloatingBmb integrates melt over the floating part of every element (sub-element Gauss
    points), so it counts partly floating elements. At 2019.1: hm4000 709.2, hm4000sem1 683.9 Gt/yr; the refit's
    element-mean form on the same fields gives 702.7 / 677.6 (ISSM ~1% above in both), so the 3.6% lower SEM1
    melt at 2019 is real. It comes from the divergent 1995-2019 histories, not the integral: the calibration
    matches at the 1995 relaxed state.
  - Early SLE, run minus its own ctrl, mm, 2030/2040/2050 (ISSM VAF): October ssp585 CESM2 -2.5/-5.8/-11.5,
    v2 -2.3/-5.1/-10.1; October ssp585 MRI -6.1 at 2050, v2 -5.9; v2 MRI ocean-only +0.05/+0.04/+0.17,
    SMB-only -6.1 at 2050; v2 ssp126 CESM2 -9.2 (October -10.4); v2 collapse -10.0; SMB feedback changes
    nothing to 2050 (CESM2 -10.09 vs -10.07). ctrl drift by 2050: October -16.2, v2 -10.7 mm. Too early to gate
    SEM1: the ocean term is still tiny at 2050 and grew mostly after 2150 in October.

### 6.36 OCX 'vary' historical runs: forcing applied, grounded ice does not respond (2026-11-22)
All four OCX-vary runs loaded (p3_flotN solve 20 h; load 184813476); comparison 184813477 (8 SU,
models/ssa_historical_friction_scores_ocx.json). OCX historical files are 110 GB each (vs 27.6 GB): pyissm
save_model writes basalforcings.tf (30 x 1,588,887 x 25) as a string array (~80 GB); data intact, the
load path converts it back. Worth fixing in pyISSM io (save list-of-arrays fields as float).
- The anomaly reaches the model: Amundsen floating ice (x -1900..-1300, y -800..0 km, 29,034 vertices)
  differs from the climatology run by up to 3.6 / 16 / 43 / 46 / 54 / 68 m in 1996 / 2000 / 2004 / 2008 /
  2012 / 2019 (mean shift a few metres).
- Grounded response is negligible: domain 2014-2019 changes by <= 1.2 Gt/yr (e.g. hm4000 -53.6 -> -52.8;
  p1_flotN -76.7 -> -75.5; p3_flotN -41.4 -> -40.6); Amundsen speed 2019/2007 changes by <= 0.002
  (hm4000 0.994 -> 0.995; observed 1.180); per-basin speed ratios identical to ~3 decimals.
- Same weak shelf-to-grounded coupling as 6.24-6.26 (Amundsen trunks +8% even with every shelf removed), and
  the vary pathway oscillates about zero. So the historical period cannot discriminate the friction laws
  with this setup. The observed PIG speed-up is attributed to ice-shelf retreat (Joughin et al. 2021), which a
  fixed calving front cannot produce.
- Next, if pursued: prescribe the observed ice front (MIPKIT icemask_greene, 24 epochs 1997-2021) in a
  historical run, removing ice beyond the observed front as the collapse runs do, and test whether the
  Amundsen speed-up appears.

### 6.37 Prescribed observed ice front in the SSA historical run (2026-11-22)
- `ais_0.1_SSA.py`, historical_dhdt_tuning_ssa: new `AIS3_SSA_FRONT=greene` (set_greene_front): moving front
  (transient.ismovingfront=1, calving.default with zero calving rate, frontalforcings.default with zero melt,
  kill_icebergs=1) with levelset.spclevelset fixed at every vertex as a time series: the model's own levelset
  at the start, then one column per Greene et al. (2022) mask (MIPKIT icemask_greene, 1997.75, 2000.2, ...,
  2021.2; nearest 500 m cell). Only observed change is imposed: floating model ice is set ice-free (+1) at an
  epoch if it is ice in the 1997 mask and not ice at that epoch (re-advance allowed). The model geometry
  already has ~present-day fronts, so ice the model has but the 1997 mask lacks is left alone (mismatch).
  Applied after the transient flags are reset. New `AIS3_HIST_END` (default 2019) for short runs.
  Tags _frontgreene and _to<end>.
- Smoke test: _hm4000, production friction, 1995-2001 (crosses the 2000.2 epoch), driver waits (96 GB).
- Smoke 1995-2001 (_hm4000_frontgreene_to2001): moving front ran cleanly (1 h 51 min, 184 SU; driver 109 SU).
  Removed 0 / 3 / 4 / 9 floating vertices at 1997.75 / 2000.2 / 2000.75 / 2001.2. Offline count per epoch (model ice
  that is ice in 1997 and not at the epoch): 2008 51 vtx / 64 km^2; 2014 136 / 284; 2016 542 / 821; 2018 874 / 4,570
  (Larsen C, A68); 2019 1,025 / 4,794; Amundsen 2016 454 / 584, 2019 660 / 918 km^2. Only 92 floating model vertices
  were not ice in the 1997 mask. The front signal is mostly 2016-2019 (PIG/Thwaites calving, Larsen C).
- Full run 1995-2019 (_hm4000, production friction, _frontgreene) submitted in submit-only mode: 184854000.gadi-pbs.
- 2026-11-22 15:30 check (au88 573.9k SU left). Progress: ctrl 2125, ssp585 CESM2 2100, ssp585 MRI 2125, ssp126 CESM2
  2200 / MRI 2175, ssp534-over CESM2 2200 / MRI 2125, collapse 2100; feedback ctrl CESM2 2125 / MRI 2150, ssp585
  CESM2 / MRI 2100; MRI ocean-only 2125, SMB-only 2150; drift ctrl 2100; hm8000 ctrl 2100. The ssp126 MRI 2075
  resume ran through, so its earlier stall was not the model state.
  - ssp585 CESM2 2100-2125 walltime-killed at 16 h (outbin ~96% of a full chunk); old-code driver 184776630 idle.
    Deleted; dir -> *_2100-2125_walltime_20261122; resumed from 2100 with AIS3_PROJ_TIME_FACTOR=5 (184854144;
    the factor is passed down the chain).
  - ssp585 CESM2 driftcorr 2019-2050 inner died (SIGABRT); old-code driver 184793499 idle 23 h. Deleted; dir ->
    *_failed_sigabrt_20261122; restarted from 2019 (184854145).
  - Stalls: hm8000 ssp585 2019-2050 (inner 184817855, clx-0181, 87% done, no write for 2.5 h) and ssp534-over
    CESM2 2200-2225 (184838418, clx-0379, no write for 4.3 h), both with 48 cores busy. Inner jobs deleted; their
    drivers are watched-solve and retry after 10 min.
  - All running drivers now started after the watched-solve fix (old ones deleted above).
  - New: `AIS3_PROJ_STALL_MIN` (default 180; 0 off) in _solve_watched kills a non-collapse inner job whose outbin has
    not grown for that long, so it is retried like a crash (passed down the chain; applies from each chain's next
    chunk). New read-only config/check_projection_chains.sh lists DEAD / STALL / ok for every unfinished chunk.
- 2026-11-22 21:15 check (526.7k SU left): health check clean apart from the collapse 2109 sub-solve (1-yr
  sub-solve, 142 min since its first output; hard collapse years took up to 3.7 h in October; covered by its 8 h
  walltime and the retry). check_projection_chains.sh now uses STALL_MIN_COLLAPSE (300 min) for collapse runs.
  hm8000 ssp585 (83 min) and MRI feedback ssp585 2100-2125 (87 min) have pre-stall-code drivers: a background
  watcher qdels either if its outbin is unchanged past 180 min (the driver then retries).
- New config/proj_v2_sle_table.py (login node; stitches chunk histories; ISSM VAF, run minus its own ctrl, mm SLE).
  First readout at 2100 / 2125 / 2150:
  - ssp585 CESM2: October -75.1 / -117.5 / -148.1, v2 -63.2 / - / -; with SMB feedback -63.5 / -92.4.
  - collapse adds (collapse - no collapse) +20.5 mm by 2100 under v2 vs +10.7 in October.
  - MRI ssp585: October -43.6 / -67.9 / -83.7, v2 -40.2 / -57.7; ocean-only +3.3 / +9.6 (October CESM2
    ocean-only +1.7 / +3.2), SMB-only -42.5 / -65.8 / -79.3; ocean + SMB within 1.5 mm of the combined run.
  - ssp126: CESM2 -35.0 / -48.6 / -62.5 (October -42.9 / -61.8 / -82.3), MRI -16.4 / -24.0 / -33.8.
  - ssp534-over CESM2 -42.9 / -59.9 / -73.0 (= ssp126 CESM2 at 2100 within 8 mm), MRI -26.4 / -39.2 / -49.2.
  - ctrl drift by 2150: October -59.1, v2 -36.9, feedback ctrls -37.3/-37.4, hm8000 -30.8 (2125), drift-corrected
    ctrl -0.1 mm through 2125 (the correction removes the drift almost completely).
  - SEM1 gate: the v2 ocean term (MRI) is ~3x October's by 2125 and accelerating; decide at ~2200.
- 2026-11-23 12:30 check (497.4k SU left). Health: all chunks writing except ssp534-over MRI 2225-2250 (112 min;
  post-stall-code driver, auto-kill at 180 min). Retries passed: ssp534-over CESM2 2200-2225, MRI feedback ssp585
  2100-2125. ctrl_coupledN_p3_driftcorr_hm4000 complete through 2150; its ssp585 partner at 2075 (restarted).
  Progress: ctrl 2225, ssp585 CESM2 2125 / MRI 2175, ssp126 CESM2 2200 / MRI 2250, ssp534-over 2225 / 2225,
  collapse 2100 (2121 sub-solve), feedback ctrls 2225 / 2250, feedback ssp585 2125 / 2125, MRI ocean/SMB-only
  2175 / 2175, hm8000 ctrl 2200 / ssp585 2100.
  - SLE (run - own ctrl, mm, ISSM VAF), 2150 / 2175: v2 MRI ocean-only +21.0 / +38.8 (October CESM2 ocean-only
    +5.5 / +8.7): ~4x October and doubling every ~25 yr. v2 ssp585 MRI turns already: -61.5 (2150) -> -50.5 (2175);
    October MRI was still falling (-83.7 / -91.4). SMB-only -79.3 / -83.2; ocean + SMB -44.4 vs combined -50.5 at
    2175 (6 mm non-additive). ssp534-over CESM2 -101.2 at 2225 vs ssp126 -86.5 at 2200; hm8000 ssp585 CESM2
    -58.9 at 2100 (hm4000 -63.2). Feedback: no effect yet (CESM2 -92.4 vs -92.0 at 2125). v2 ctrl drift -53.9 by
    2225 (October -88.0).
  - SEM1 gate effectively met: on the current trend the v2 ocean term passes ~100 mm well before 2300.
- 2026-11-23: SEM1 2x2 approved and launched (to 2300; AIS3_SSA_CALIB=_hm4000sem1, AIS3_ISSM_DIR=
  /g/data/au88/jh7060/ISSM-40c764c7-sem1/install, coupledN_p3; ssp585 runs with AIS3_PROJ_TIME_FACTOR=5 from the
  start): ctrl_coupledN_p3_hm4000sem1 184910466, ssp585_cesm2_ref_ref_coupledN_p3_hm4000sem1 184910467,
  ctrl_coupledN_p3_smbfb-cesm2_hm4000sem1 184910468, ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000sem1 184910469.
  With stage A (v2) and stage B (v2+fb) CESM2 runs this completes the SEM1 x feedback 2x2. Added to
  config/proj_v2_sle_table.py.
- 2026-11-23: full _hm4000_frontgreene solve finished (exit 0) in 13.8 h / 1,326 SU (about 2.7x the fixed-front run:
  the moving front is costly). The detached watcher died again; load 184910843 and comparison 184910844
  (AIS3_SCORE_TAG=_front) submitted by hand.

### 6.38 HO track: melt refit at gamma_0 = 4000 and friction refinement in the formerly pinned regions (2026-11-23)
- (a) `melt_calibration_hm_refit_ho.py` (job 184915551): as the SSA refit, on the HO 1995 relaxed geometry
  (base-layer draft and masks of AIS3_ho_relaxed.nc; mesh/basins from the SSA melt file). Members
  hm1000/hm4000/hm8000 in models/melt_calibration_hm_ho.json, all basins matched exactly. hm4000 deltaT
  -0.424 .. 2.323 degC (median 0.307; Amundsen 2.323, the only basin > 2), close to the SSA hm4000 (-0.439 .. 2.209).
  By basin: -0.424 0.188 0.255 0.924 0.753 0.807 0.015 -0.092 1.145 2.323 0.650 -0.065 -0.007 0.607 0.051 0.358.
- (b) `ho_friction_fixedpoint.py` (driver 184915690, launch_ho_friction_fixedpoint.pbs, 24 cpu / 720 GB):
  forward-only C updates on the 1995 HO state with the SSA Budd field and no pins, on formerly pinned grounded
  vertices with obs > 10 m/yr: C *= sqrt(u_model/u_obs), smoothed (two element-averaging passes), limited to
  [0.67, 1.5] per iteration, C in [0.05, 900]; 3 iterations, each a 96-cpu hugemem stress-balance solve
  (execution/AIS3_ho_fricfp_it<k>); iteration 0 speeds from B95 (6.27). Output models/ho_friction_fixedpoint.npz.
- Then (c): pin-free HO historical with the refined C and gamma_0 = 4000 (needs an option to read both).

### 6.39 Observed front retreat barely reaches the grounded ice (2026-11-23)
Load 184910843, comparison 184910844 (models/ssa_historical_friction_scores_front.json). hm4000 with the observed
Greene front vs the same run with a fixed front:
- Amundsen 2014-2019 -76.4 -> -77.0 Gt/yr (CPOM -97.1); domain -53.6 -> -54.2 (CPOM -56.9); Filchner-Ronne
  unchanged; Amundsen speed 2019/2007 0.994 -> 0.998 (observed 1.180).
- Local, Amundsen 2019, front/fixed speed ratio: floating median 1.000, p95 1.10, max 1.91; grounded < 5 km of
  the GL p95 1.03 (max 1.53 at a few vertices); 5-20 km p95 1.02. The floating ice near the removed front speeds
  up; the grounded ice barely does (observed PIG grounded speed-up after the 2017-2020 calving ~12%, Joughin et
  al. 2021).
- Fourth independent test of the same property (shelf removal 6.24, softened margins 6.26, OCX vary 6.36,
  observed front here): in this SSA setup, Amundsen grounded ice responds very weakly to losing or thinning
  its shelves. This, not the melt or friction calibration, is the main limit on the model's melt sensitivity.
- Open suspects: the SSA approximation itself (no HO shelf-removal test yet), and how the inverted drag and the
  prescribed N hold the grounding zone. Next decisive check: the shelf-removal test on the pin-free HO state
  once the HO friction refinement (6.38) is done.

### 6.40 HO shelf-removal test queued (2026-11-23)
`check_ho_shelf_removal.py` (184919663, afterok on the friction fixed-point driver 184915690): on the pin-free HO
1995 state with the refined friction (last fixed-point iteration), remove all floating ice (+ unanchored pieces,
column-wise) and solve once (execution/AIS3_ho_shelfrm_F1); the baseline F0 is the fixed-point run's last solve.
Reports F1/F0 per LARMIP-3 sector and distance band next to SSA X1/S0 (6.24) on the same 2D vertices, and the
Amundsen fast trunks. Output models/ho_shelf_removal_test.npz.

### 6.41 HO friction refinement and HO shelf-removal results (2026-11-24)
- ho_friction_fixedpoint.py (driver 184915690, 5.1 h; 3 hugemem solves): formerly pinned grounded vertices with
  obs > 10 m/yr (124,250): speed/obs median 0.668 (B95) -> 0.714 -> 0.752 -> 0.782; RMSE 160.1 -> 154.5 -> 150.8 ->
  148.8 m/yr; C factor median 0.85 / 0.88 / 0.90 (p10 at the 0.67 limit). Never-pinned grounded unchanged (0.972-0.973,
  RMSE 105.9 -> 105.6). Not converged; ~3-4 more iterations would approach 0.9-0.95.
- check_ho_shelf_removal.py (184919663, 1.4 h, 100 SU driver): median speed-up F1/F0 (flux-weighted, capped) vs SSA
  X1/S0 on the same vertices, grounded, baseline > 100 m/yr:
  - Amundsen 0-5 km HO 1.11 (1.36) | SSA 1.06 (1.28); 5-20 km 1.03 | 1.01; trunks (obs > 1000 m/yr, < 20 km) HO 1.08 |
    SSA 1.06 (n 8,888).
  - Weddell 0-5 km HO 1.86 (3.26) | SSA 1.50 (2.32); Ross 1.54 | 1.66; East Antarctica 1.10 | 1.04 (trunks 1.11 | 1.03);
    Peninsula 1.08 | 1.05.
  - One cut grounding-line spot reaches 2.3e8 m/yr (as in the SSA test; excluded by medians).
- Reading: HO responds only slightly more than SSA (clearly more only in Weddell). The SSA approximation is not the
  main reason Amundsen grounded ice ignores its shelves; the cause is shared by both tracks (inverted friction,
  prescribed N with linear sliding, or the geometry).
- Next: SSA shelf removal with more plastic sliding (p5_flotN, regularised Coulomb) to test the friction law's role;
  finish the HO friction refinement, then the pin-free HO historical with gamma_0 = 4000.
- 2026-11-24 13:30 check (433.2k SU left). Complete through 2300: ctrl_coupledN_p3_hm4000, both feedback ctrls,
  ctrl_coupledN_p3_hm8000, ssp126 MRI, ssp534-over CESM2 and MRI. Two chains stopped after attempt + retry both hit
  the 16 h walltime at the same point (deterministically slow, not stalled): ssp126 CESM2 2200-2225 (96%) and MRI
  SMB-only 2175-2200 (92%). Dirs -> *_walltime_20261124(_attempts); resumed with AIS3_PROJ_TIME_FACTOR=5:
  185034786 / 185034787. MRI feedback ssp585 2200-2225 at 169 min without a write (auto-kill at 180).
  - SLE (run - own ctrl, mm, ISSM VAF):
    - v2 ssp585 CESM2 turns near 2150: -99.0 (2150), -75.3 (2175); October minimum -160 near 2185.
    - v2 collapse +16.8 at 2150 (collapse adds +116 mm by 2150 vs +50 in October).
    - v2 ssp585 MRI positive from ~2210: -18.9 (2200), +27.8 (2225), +75.7 (2250); October MRI -64.4 at 2250.
      MRI ocean-only +63.6 / +94.9 / +132.6 / +175.0 at 2200-2275 (~6x October CESM2 ocean-only). Under v2 the
      MRI reversal is ocean-driven.
    - SMB feedback: CESM2 -10.6 at 2200, MRI -11.9 at 2200 vs -18.9 without (+7 mm).
    - ssp534-over at 2300: CESM2 -119.8, MRI -130.4, MRI more negative than ssp126 MRI (-109.0): no reversal
      on the overshoot path.
    - hm8000 ssp585 CESM2 -71.3 at 2150 vs hm4000 -99.0 (+28 mm from the stronger melt sensitivity).
    - Drift test at 2100: drift-corrected ssp585 - ctrl -63.6 vs -63.2 uncorrected (0.6%): ctrl subtraction holds.
    - SEM1 at 2100: -63.3 (no feedback) / -63.6 (feedback) vs -63.2 / -63.5: no effect yet.
    - v2 ctrl drift -69.4 mm by 2300 (October -116.0); hm8000 ctrl the same (-70.6).

### 6.42 Friction-law role in the shelf-removal response; HO refinement continued; pin-free HO historical at gamma_0 = 4000 queued (2026-11-24)
- check_shelf_removal_plastic.py (driver 185035699, 15 min; four 48-cpu SSA solves, 3-11 min each; ~65 SU in
  total): the 6.24 shelf-removal test on models/AIS3_proj_ssa_init.nc with two more plastic laws, both with
  flotation N (coupling 2) and the production basal drag kept where the law allows it:
  - Budd p=5, q=1 (drag-preserving rescale).
  - Schoof law (ISSM friction.schoof, the MISMIP+ form; m = 1/3, Cmax = 0.4). It is often called a regularised
    Coulomb law, but it is not ISSM's regcoulomb (Joughin et al. 2019), which has no N.
  - In the Schoof runs, 9,617 grounded vertices (0.8%) sit at the r = tau0/(Cmax N) = 0.95 cap, so their drag
    is lower than production. Grounded RMSE against obs: Schoof 122.5 m/yr; p=1 85.2, p=3 85.3, p=5 86.7.
  - Results (grounded, baseline > 100 m/yr): median speed-up with shelves removed over shelves as is (flux-weighted, capped):

    | law | Amundsen 0-5 km | 5-20 km | trunks | Weddell 0-5 km | Ross 0-5 km | EA trunks |
    |---|---|---|---|---|---|---|
    | Budd p=1, prescribed N (production) | 1.09 (1.35) | 1.02 | 1.07 | 1.84 | 2.32 | 1.06 |
    | Budd p=3, flotation N | 1.20 (1.53) | 1.06 | 1.15 | 2.48 | 2.82 | 1.15 |
    | Budd p=5, flotation N | 1.25 (1.61) | 1.10 | 1.19 | 2.73 | 3.11 | 1.19 |
    | Schoof, flotation N | 1.19 (1.54) | 1.06 | 1.18 | 2.36 | 2.76 | 1.13 |

- Reading:
  - The friction law matters: plastic sliding with flotation N roughly triples the instantaneous speed-up of the
    Amundsen trunks (+7% to +18-19%) and raises it everywhere else.
  - Even fully plastic, Amundsen's grounded ice is still the least sensitive of the marine sectors (Weddell and
    Ross speed up 2.4-3.1x at the grounding line). The weak shelf coupling there is therefore not mainly the
    linear-Budd-plus-fixed-N choice. The remaining suspects are the geometry and buttressing of the Amundsen
    shelves themselves (Thwaites' unconfined tongue, PIG's narrow shelf) and the inverted C, which puts the
    resistance on the bed.
  - These are diagnostic solves. Under flotation N, thinning near the grounding line lowers N and so weakens the
    bed further; that feedback only appears in a transient. The p3_flotN historical (6.33) showed no extra Amundsen
    acceleration over 1995-2019, but the 2019-2300 projections could differ: a p3_flotN or p5_flotN projection
    is the test.
  - The Schoof baseline drifts from production: its capped vertices sit on the fast trunks near the grounding
    line, where N goes to 0. Its ratios therefore compare against a faster baseline.
- HO friction refinement continued: AIS3_FP_CONTINUE=1 runs iterations 4-7 from C3 in
  models/ho_friction_fixedpoint.npz. Driver 185035566 (hugemem 24 cpu / 720 GB) writes
  models/ho_friction_fixedpoint_cont.npz; run dirs are execution/AIS3_ho_fricfp_it4..7; the it1-3 logs are
  archived as *.it1-3_20261123.
- ais_0.1.py, new historical options (only with AIS3_HIST_FRICTION=budd_nopins):
  - AIS3_HIST_FRICTION_C='<npz>:<key>' replaces the Budd C.
  - AIS3_HIST_MELT='hm<g>' sets gamma_0 and delta_t from models/melt_calibration_hm_ho.json on a fresh start
    (asserted on resume).
- Pin-free HO historical with gamma_0 = 4000 and the refined C7: launch_historical_budd_nopins_hm4000.pbs, tag
  _budd_nopins_hm4000. Four chunks 185035602 / 604 / 605 / 606, chained afterok on 185035566.
- 2026-11-24 16:00 check (413.8k SU left): health clean. SEM1 ctrls at 2200, SEM1 ssp585 runs at 2125. SLE at
  2125: SEM1 -92.4 vs v2 -92.0, SEM1+fb -92.9 vs v2+fb -92.4 (no SEM1 effect yet); SEM1 ctrl drift -30.3 vs -31.0
  (v2). Drift test at 2125: corrected -94.7 vs -92.0 uncorrected (2.9%, under the 10% caveat threshold). Figures:
  config/plot_proj_v2.py -> docs/figures/projections_v2_{sensitivity,scenarios}.png.
- HO refinement iterations 4-7 done (driver 185035566, 7.2 h; solves 1.5 h / ~430 SU each; ~2,250 SU in total
  including the 527 SU driver). Formerly pinned region speed/obs median 0.782 (it3) -> 0.803 -> 0.818 -> 0.828 ->
  0.837; RMSE 148.8 -> 146.3 m/yr; never-pinned grounded unchanged (0.973, RMSE 105.4). Gains shrink each step
  (+0.021 ... +0.009) and the result stops short of 0.9. 29% of the region's vertices sit at the C = 0.05 lower
  bound, and 43% of the vertices still below 0.6 of obs are at the bound: there the ice is too slow with almost
  no basal drag, so stiff ice (B) or the geometry limits it, not the friction.
- Chunk 1 of the pin-free HO historical (185035602) started from C7: C changed at 93,030 vertices (factor median
  0.585), gamma_0 = 4000, delta_t [-0.424, 2.323].
- 2026-11-25 00:00 check (399.6k SU left). First automatic stall kill: SEM1 ssp585 2125-2150 (no write after its
  first output for 3 h) was qdel'd by its driver ~17:34 and retried 17:46 (185065086); the retry runs normally
  (~88% at 23:52). Complete through 2300: ssp585 MRI v2, MRI ocean-only. Drift-test pair complete through 2150.
  - SLE (mm, run - own ctrl): v2 ssp585 MRI +194.1 at 2300 (October -41.9); MRI ocean-only +222.9 at 2300
    (October CESM2 ocean-only +36.4, ~6x). v2 ssp585 CESM2 -18.8 at 2200 (October -152.0). v2 collapse +118.1 at
    2175 (October -81.1). hm8000 CESM2 -28.9 at 2175 vs hm4000 -75.3 (+46 mm).
  - SMB feedback: CESM2 -10.6 vs -18.8 at 2200 (+8 mm), +107.1 at 2225; MRI +41.2 vs +27.8 at 2225 (+13 mm).
  - Drift test final (2150): corrected -104.1 vs uncorrected -99.0, 5.2%; 2.9% at 2125, 0.6% at 2100. Subtracting
    the ctrl holds within ~5% to 2150: no caveat needed for the 2100 numbers.
- 2026-11-25 11:35 check (375.0k SU left): health clean; all unfinished chains advanced overnight (ssp585 CESM2
  2225, ssp126 CESM2 2225, collapse 2175, feedback ssp585 CESM2 2250 / MRI 2275, MRI SMB-only 2200, hm8000 2200,
  SEM1 ctrls 2275 / 2250, SEM1 ssp585 2150 / 2150).
  - SLE (mm, run - own ctrl): v2 ssp585 CESM2 +87.9 at 2225 (October -103.6). SMB feedback grows: CESM2 +107.1 vs
    +87.9 at 2225 (+19), +273.5 at 2250; MRI +172.9 vs +134.7 at 2275 (+38). hm8000 CESM2 +46.4 vs -18.8 at 2200
    (+65). MRI SMB-only turns too: -83.2 (2175) -> -71.0 (2200).
  - SEM1 at 2150: -102.2 vs v2 -99.0 (3 mm MORE negative); SEM1+fb -99.3 vs v2+fb -99.2. No added loss yet;
    the SEM1 2019 state has ~3.6% less floating melt than hm4000 (see SEM1 smoke check), which may offset the
    grounding-line melt early. Watch 2175-2250.
- 2026-11-25 13:35 check: health clean (375.4k SU). SEM1 ssp585 CESM2 vs v2 (mm, minus own ctrl), 2150/2160/2170/
  2175: -102.2/-95.9/-87.5/-79.8 vs -99.0/-92.4/-83.2/-75.3. A steady 3-4.5 mm offset to more negative, slowly
  growing; SEM1 does not add loss so far (cf. x1.35 in the LARMIP SEM1 test).
- 2026-11-26 check (342.8k SU left). Complete through 2300: ssp126 CESM2, feedback ssp585 CESM2 and MRI, MRI SMB-only,
  both SEM1 ctrls. Running: ssp585 CESM2 2275, hm8000 2250, SEM1 ssp585 2225 / 2225.
- **v2 collapse run blew up at ~2180; chain stopped.** The 2175-2200 driver died 2026-11-25 ~06:10 before submitting
  sub2183: pyissm consistency check "equality thickness = surface-base violated" (a Python exception, so no checkpoint
  past 2175 and no chaining). Cause found by parsing the sub-solve outbins:
  - max thickness stays ~4.6 km through sub2179; end of sub2180 21.7 km, sub2181 439 km, sub2182 1.3e8 m (speeds to
    1e16 m/yr); the 1e-9 m absolute tolerance of the check then fails on rounding.
  - runaway cluster: 84 grounded vertices with H > 6 km at the end of sub2180, ~23 x 7 km around x 2490, y -411 km
    (~81E, 67.5S, West Ice Shelf coast, East Antarctica), bed +430..+610 m, 2019 thickness 200-320 m, speed ~3 km/yr.
  - precursor since 2156: the first step after a collapse removal often has a one-step speed spike (3e5 to 4e16 m/yr,
    sub2156/2159/2171/2172/2179) that recovers by the year-end output; survivable until 2180.
  - The 2150-2175 history and the 2175 checkpoint are clean (max H 4.61 km, max speed 21.7 km/yr), so v2 collapse
    results stand through 2175 (+118.1 mm). Not yet rerun.
- 2026-11-26 17:00 check: health clean (341.6k SU). v2 ssp585 CESM2 +235.7 (2250), +410.0 (2275) (October -21.3 / +89.3);
  with feedback +273.5 / +475.6. SEM1 vs v2 at 2175/2200/2225: -79.8/-24.8/+80.7 vs -75.3/-18.8/+87.9 (SEM1 4.5-7 mm
  lower, gap growing slowly); with feedback -73.6/-11.5/+105.9 vs -73.3/-10.6/+107.1 (~1 mm lower). So far SEM1
  (with the matched-present-day refit) does not add loss in the ssp585 projections, unlike the x1.35 LARMIP test.
  Collapse rerun from 2175 awaiting approval of the fix.
