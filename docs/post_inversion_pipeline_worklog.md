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
| 3. Ocean melt (gamma) calibration | `melt_gamma_tuning` | **invalid — RMSE metric confirmed gamed** (§4a), use the ISMIP7-informed calibration instead | `AIS3_melt_gamma_tuning.nc` (do not use) |
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
Confirms the chunking mechanism works end-to-end. Follow-up sanity check
(`check_historical_chunk1.py`, job 179486566) submitted to inspect the chunk-1 checkpoint
(field sanity, grounding-line migration activity, dH distribution, early dH/dt vs
`dhdt_cpom` as a 6-of-24-year directional check only) — pending.

**Chunk 2 (job 179486058, t=2001→2007): submitted, running** (queued ~44h before starting —
hugemem congestion has gotten noticeably worse than chunk 1's queue wait). Chunks 3
(2007→2013) and 4 (2013→2019, final) to follow the same check-then-resubmit pattern; the
shared `steps=['historical_dhdt_tuning']` toggle in `ais_0.1.py` stays set until the full
run completes, then reverts to `['ho_friction_inv']`.
