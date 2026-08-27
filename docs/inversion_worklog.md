# ACCESS-AIS3 inversion work log

Diagnosis and repair of the present-day friction / rheology inversion.
Goal: a well-initialised Antarctic state with **grounded velocity RMSE < 50 m/yr** and
spatially-white residuals (the reference result achieved with shelf-only rheology
inversion plus a friction inversion, grounded B from temperature).

---

## 1. Headline progress

| configuration | grounded RMSE (m/yr) | shelf RMSE | notes |
|---|---|---|---|
| null model (`RMS(v_obs)`) | 310.6 | — | baseline to beat |
| original setup | 323.5 | — | **worse than null** — model was effectively static |
| joint friction + B, 70 iters | 171.2 | — | 70 % of grounded within ±50 |
| friction-only, 200 iters (`friconly_fz`) | 110.6 | — | 83 % within ±50; median residual −0.1 |
| **`friconly_nfix`, 192 iters (dxmin), corrected 100 m floor** | **98.9** | 252.5 | **best converged, self-consistent state — current production candidate** |
| forward-only: `C(nfix)` + damage-relaxed shelf B | 67.3 | 172.8 | **not an inverted state — see §5.2, does not survive re-optimisation** |

`friconly_nfix` supersedes `friconly_fz`: same method, corrected thickness-floor geometry
(§4, floored into the base rather than rebuilding the surface). It is the current best
converged, self-consistent (C and B mutually optimal) result and the candidate for
production promotion.

Residual structure at 110.6 (fz; the last run this breakdown was computed for):

```
median  -0.1     mean  -4.9     std 110.5     |res| < 50 : 83 %
grounded, >100 km inland : RMSE  55.3   ratio 1.00   <- already at target
grounded, 0-5 km from GL : RMSE 185.9   ratio 1.18   <- grounding zone
Transantarctic Mountains : RMSE 269.6   ratio 2.00   <- worst region
shelves (out-of-sample)  : RMSE 237.9   ratio 1.03
```

**The error is highly concentrated: 9.01 % of grounded vertices carry 90.6 % of the total
squared error.** The interior is already at the target; what remains is a small number of
identifiable regions.

---

## 2. Bugs found and fixed

### 2.1 Friction coefficient initialised in a dead zone
With Budd `p=q=3`, `tau_b = C^2 · N · u^(1/3)`, so **`u ∝ C^-6`**. The initial guess
`C = 100` put the model six orders into a region where `∂u/∂C ≈ 0`; grounded ice was
static (median velocity 0.0 m/yr against 71.9 observed).

Fixed: `C_init = 1.8`, bounds `[0.1, 10]`, `control_scaling_factors = 1.8`.

**Consequence:** every earlier tuning result is void. The "best config" from the original
regularisation campaign was optimised against `fastGrnd RMSE ≈ 498`, which was literally
`RMS(v_obs) = 499.1` — the metric never measured model skill because the model did not move.

### 2.2 `rheology_B` was a uniform placeholder
`cuffey(273.15 − 20) = 2.027e8` everywhere, collapsing a ~17× real spatial viscosity range
(8.5e7 → 1.43e9 over the RACMO range) to one value. 51 % of grounded vertices were wrong by
> 20 %. No friction inversion can correct this — friction acts on the bed, the error is in
the ice.

Fixed in `config/ais_0.1_param.py`: B derived from the RACMO surface-temperature field.
Caveat retained in the code: surface temperature is a proxy for depth-averaged, biased stiff.

### 2.3 Wrong cost-function ID for friction regularisation
The friction L-curve used **502** (`RheologyBbarAbsGradient`), which is not a control in a
friction inversion — the term was inert and the L-curve flat. Fixed to **501**
(`DragCoefficientAbsGradient`) in `config/ais_0.1.py`.

### 2.4 Element-based friction mask stripped drag from grounded ice
The zero-friction mask flagged *all* vertices of any element containing one floating
vertex, removing basal drag from a band of genuinely grounded ice at the grounding line.
92.7 % of the velocity blow-up lived in that band.

Fixed: vertex-based test `ocean_levelset < 0`.

### 2.5 Absolute-misfit weighting ignored the slow interior
`cf101 = 1000 / cf103 = 0.1` weights absolute error, so the slow interior contributed
almost nothing. Fixed: log-weighted `cf101 = 10 / cf103 = 100`.

---

## 3. Dead ends (with evidence)

### 3.1 Friction regularisation (501) is unusable
Sweeping 501 upward produced **five byte-identical non-starters** — `C moved 0.0 %`,
roughness exactly 0.0000.

Root cause: the friction field steps `1.8 → ~0` across the grounding line because floating
C is pinned. At the initial guess that single ring of elements (2.3 % of the mesh) supplied
**100.0000 %** of `|grad C|^2`; grounded ice contributed exactly **0**. Since floating C is
bounded to `[0,0]`, the penalty is irreducible — a fixed cost with a gradient ~10^7 times
the misfit gradient, which kills the m1qn3 line search for every `501 >= 1e2`.

Masking the grounding-line ring out of the functional removed the artefact (verified:
`|grad C|^2` at init `1.423e5 → 0`) and the runs then iterated 17–33 steps. But they still
stalled early with a **ruined fit** (RMSE 2900–3300, worse than null; C moved only ~25 %).
The smooth fields were smooth because they had barely moved, not because they were coherent.

**Conclusion:** the data term wants sharp features at ice-stream margins, the smoothing term
fights them, and the line search cannot reconcile the two. Abandoned; 501 left inert.

### 3.2 Joint friction + B inversion is worse than friction-only
| | abs misfit (101) | log misfit (103) | grounded RMSE |
|---|---|---|---|
| joint (stopped, dxmin @ 70) | 684.1 | 1.399e6 | 171.2 |
| friction-only (200 iters) | 615.9 | 6.852e5 | **110.6** |

B freedom let the optimiser **stop early at a worse velocity fit**. A previously-reported
"invariant fast-ice deficit of 0.70–0.74 across every configuration" was an artefact of
this early stopping — friction-only running to 200 iterations moved it to **0.87** on its own.

### 3.3 Dropping the thin-ice floor entirely
Removing the floor *and* excluding thin ice (H < 50 m) from the misfit gave
**RMSE 1917.4, maxvel 1.6e5** — a runaway. The lower cost function was an illusion created
by the smaller misfit set: excluding thin ice did not fix it, it merely stopped constraining
it. Thin ice must stay **in** the misfit.

---

## 4. The Transantarctic Mountains failure

Confirmed geographically: 30,739 grounded vertices, **RMSE 269.6, ratio 2.00** (2× too fast).

Residual binned by physical regime (baseline joint run):

| regime | n | RMSE | ratio |
|---|---|---|---|
| N < 1e5 Pa | 18,355 | 678.9 | **14.56** |
| H floored to 100 m | 71,578 | 368.7 | 2.19 |
| slope > 0.1 | 88,152 | 357.7 | 2.56 |
| H 0–150 m | 85,783 | 342.8 | 1.93 |
| bed 1000–4000 m | 66,695 | 265.3 | 1.58 |
| *real ice, H > 100 m* | 1,158,642 | 150.8 | 1.01 |

Every failing regime is **too fast**, and they all describe thin ice on steep, high bedrock.

### Root cause: a self-inflicted geometry error (in the experiment scripts, not production)
The experiment scripts floored thickness at 100 m and then rebuilt
`surface = bed + H`, which **raised the surface** wherever real ice was thinner and corrupted
the **surface slope**. Since `tau_d = rho·g·H·grad(s)`, that fabricates the driving stress —
worst exactly in the mountains.

Production (`config/ais_0.1_param.py:62`) does it correctly, flooring at 10 m and absorbing
the change into the **base**, leaving the observed surface intact.

### Hypothesis that was refuted
Effective-pressure starvation was proposed as the cause and **disproved**: above sea level
`N/overburden = 1.016`, i.e. already correct. Genuine N-starvation exists but only in the
small 18k-vertex group (`N/overburden = 0.007`) responsible for the 14.56× outliers.

---

## 5. Shelf rheology is inconsistent with the coupled model

`md.extract()` imposes Dirichlet BCs from `vel_obs` on **every** new boundary node
([`Model.py:558-560`](../../../pyISSM/src/pyissm/model/Model.py)). For the floating-only
extraction used by the shelf inversion, that clamped **both** the grounding line and the
**calving front** to observations. The coupled run instead uses a **Neumann** shelf front.

So B was fitted while artificially restrained; released, the shelf accelerates:

| distance from GL (km) | RMSE | median residual | ratio |
|---|---|---|---|
| 0–5 | 227.4 | −2.9 | 1.08 |
| 15–30 | 260.9 | +7.5 | 1.03 |
| 60–120 | 343.4 | +108.6 | 1.19 |
| 120–250 | 296.5 | **+226.9** | **1.26** |

The error **grows** with distance from the grounding line — i.e. it is worst at the calving
front, not inherited from grounded ice.

**A shelf-only domain cannot be made BC-consistent.** Freeing the calving front on floating
ice (which has no basal drag) leaves the system nearly singular; the solve fails outright
(`Recovery solver failed`, 0 iterations). The original workflow only worked because the
all-Dirichlet extraction was propping it up — the very artefact that biased B soft.

Therefore shelf B must be inverted **inside the coupled domain**, where the front BC and the
grounding-line inflow are both exactly what production uses.

### 5.1 Shelf-only B chain (works, stays the method)

Inverting B on the **shelf-only domain** (all-Dirichlet — the one boundary treatment that
actually converges) in three steps, each warm-started from the last, C held fixed throughout:

| B state | grounded RMSE | shelf RMSE | front bias |
|---|---|---|---|
| `execution_newB_rheology` (`friconly_nfix`'s own shelf B — a genuine shelf-only inversion, but on the pre-floor geometry) | 98.9 | 252.5 | +202 |
| geometry-matched (re-inverted on the corrected 100 m floor geometry, same as production) | 68.0 | 201.8 | −210 |
| damage-relaxed (bound `0.2 × cuffey(0°C)`) | **67.3** | **172.8** | **−114** |

Only 6 % of shelf vertices actually use the damage freedom (concentrated in shear margins);
median B lands intermediate (1.68e8, between the geometry-matched 1.50e8 and the damage
bound 1.82e8). All-ice misfit weighting beats grounded-only (109.5 vs 127.2 grounded; 196.7
vs 269.5 shelf) — the shelf residual is informative even though C cannot respond to it.

This chain is real progress on B **in isolation**, with C frozen. It does not mean the
resulting *(C, B)* pair is jointly optimal — see §5.2.

### 5.2 Coupled-domain re-inversion is unstable: 67.3 does not survive optimisation

The natural next question: take the 67.3 forward pairing — `C` from `friconly_nfix`, B
swapped to the damage-relaxed shelf field — and let the optimiser refine `C` (or B) around
it inside the coupled domain, where the boundary conditions are correct. **Every attempt to
do this has made the fit worse, never better: 11 independent runs, 11 failures.**

Three fresh attempts, warm-started at exactly the 67.3 state (`iscontrol` bug from an
earlier failed attempt fixed, confirmed running to real convergence — `dxmin`, not the
22-second no-op the bug produced):

| run | iter 1 cost | final cost | Δcost | grounded RMSE | shelf RMSE |
|---|---|---|---|---|---|
| `dmg_nfixcfg` (warm-start = the 67.3 state exactly) | 3.18e5 | 5.25e5 | **+65 %** | 99.6 | 240.2 |
| `noreg2` (matched-geometry B, chained) | 3.47e5 | 5.81e5 | **+67 %** | 127.1 | 269.4 |
| `prod_lowC2` (damage B, all-ice misfit, C bound 0.01) | 3.86e5 | 6.51e5 | **+69 %** | 117.3 | 202.8 |

`dmg_nfixcfg` is decisive: it is warm-started *exactly* at the 67.3 pairing, so iteration 1
reproduces it, and the optimiser immediately moves away — cost jumps to 4.8e7 by iteration 2
and only partially recovers, settling 65 % above where it started. The final `C` is
numerically indistinguishable from `friconly_nfix`'s own C (all changes < 1 % on every
grounded vertex) — the line search rejected essentially everything and the run drifted back
toward the *old* optimum, not a better one. `noreg2` and `prod_lowC2` land worse still, and
none beats `friconly_nfix`'s 98.9.

This joins 8 earlier coupled-domain B/Schoof attempts that also ended worse than their own
iteration 1 (`control_scaling_factors` confirmed inert via bit-identical trajectories across
scalings; "control magnitude predicts failure" was also refuted, since the shelf-only B
inversion at a comparable magnitude, §5.1, works fine).

**Conclusion: 67.3 is a coincidence, not a floor.** The pairing of `C(nfix)` with a
differently-fitted shelf B happens to score well precisely because neither side has adjusted
to the other; the moment either is allowed to respond, the joint landscape pulls back toward
`friconly_nfix`'s own optimum. **`friconly_nfix` (grounded RMSE 98.9) is therefore the best
available converged, self-consistent state**, and coupled-domain re-inversion of shelf B
against a fixed-C friction field should be considered a closed dead end, not a pending item.

---

## 5.3 Siple Coast: three mechanisms ruled out, unresolved

Regional zoom-ins (Thwaites/PIG, Totten, Siple Coast, Filchner-Ronne) at the request of a
review of `friconly_nfix`/derived states show Siple Coast is the one trunk with a
**persistent, coherent deficit** across every configuration tested — trunk velocity ratio
(model/obs, `vel_obs > 500`) stuck at **0.35–0.42**, while Thwaites/PIG sits at 0.96–1.00,
Totten at 0.79–0.84, and Filchner-Ronne at 0.70–0.87. This is the "consistently lower
velocity through a whole glacier trunk" failure mode called out as the important one to
chase, unlike diffuse East Antarctic noise.

Ruled out:
- **Basal drag magnitude** — lowering the C bound 0.1 → 0.01 cut Siple Coast drag 225× with
  essentially no velocity response (ratio 0.36 → 0.37). The trunk is not drag-limited.
- **Mesh resolution** — Siple Coast has ~9 elements across the trunk, *better* resolved than
  Totten (6.5) and Filchner-Ronne (3.1), both of which fit better. Under-resolution is not
  the cause.
- **Grounded rheology B** — cannot currently be inverted in the coupled domain (same
  instability as §5.2 applies to any control that touches the coupled system).
- **Regularised Coulomb friction law (Schoof)** — tested directly, see §5.4. Ruled out: the
  trunk ratio does not move regardless of the Coulomb cap strength.

**Open.** Remaining candidates: basal topography/bed error specific to the Siple Coast
ice-plain geometry, or till/hydrology physics not captured by *either* Budd or a simple
Coulomb-capped law (§5.4) — a friction-law swap alone is not the fix.

---

## 5.4 Schoof (regularised Coulomb) friction: tested and ruled out

Motivation: Siple Coast is a textbook till-dominated, hydrology-controlled ice-stream system
(weak, water-saturated till — Tulaczyk-style), which a smooth Weertman/Budd power law cannot
represent but a regularised Coulomb law (Schoof) is designed for — Budd and Schoof coincide
at low velocity and diverge exactly at fast, near-yield flow, i.e. exactly at Siple Coast.

**Coupled-domain adjoint inversion of Schoof's `C` is unstable, continent-wide and scoped.**
Warm-started from the converged Budd `C` via the exact Weertman-limit match
`C_schoof = C_budd·√N` (valid only in that limit — worst exactly where the Coulomb cap
`τ_b → Cmax·N` binds, i.e. fast near-yield ice):
- **Continent-wide** (`schoof_warm`): cost blew up 60× at iteration 2, repeated
  `maximum number of nonlinear iterations (100) exceeded`. Final state: grounded RMSE
  **1089.6**, max velocity **140,216 m/yr** — a runaway, not a fit.
- **Scoped to Siple Coast only** (`C` free on 111,380 Siple vertices, pinned to the
  Weertman match everywhere else), with `stressbalance.isnewton=2` (hybrid Newton, up from
  the untried default Picard) and `maxiter=300` (up from the default 100 that the first
  attempt hit): the unperturbed warm start converges cleanly (residual 0.006% → 0.4%), but
  the instant m1qn3 perturbs `C` within Siple Coast, the nonlinear solve stops converging
  and **oscillates at 12–27% residual** for 50+ iterations without settling — a genuine
  limit cycle, not a slow-but-working solve. Killed after confirming the pattern; not a
  solver-setting problem (Newton vs Picard, iteration cap, and tighter control bounds all
  made no difference to the failure mode itself, only where it manifests).

**Conclusion: the instability is specific to gradient-based (adjoint) perturbation of `C`
near the Coulomb cap**, not a general property of the friction law. Confirmed by:

**Forward-only Cmax sweep** (`C` fixed at the Weertman match everywhere, no inversion, no
adjoint — only Iken's bound `Cmax` varied over its documented typical range 0.17–0.84):
all 5 forward solves converged cleanly in ~2 minutes each, no oscillation. But:

| Cmax | grounded RMSE | Siple trunk ratio (`vel_obs > 500`) |
|---|---|---|
| — (`friconly_nfix`, Budd) | 98.9 | 0.40 |
| 0.17 | 10200.3 | 0.41 |
| 0.30 | 5488.5 | 0.40 |
| 0.50 | 2322.5 | 0.40 |
| 0.65 | 1358.2 | 0.39 |
| 0.84 | 872.8 | 0.39 |

Two independent findings here:
1. **The Siple Coast trunk ratio does not move**, at all, across the entire physically
   plausible Cmax range — median-based, so robust to the RMSE blowup below. Coulomb capping
   does not improve the fit at Siple Coast. This is a direct test of the till/hydrology
   hypothesis via the simplest possible mechanism (cap the existing Budd-derived traction),
   and it does not help.
2. **The catastrophic RMSE is a separate, unrelated failure.** Traced every one of the
   worst-residual vertices (velocities up to 132,939 m/yr against observed speeds of single
   digits) to an exact signature: `H = 100 m` (the numerical thickness floor) and
   `N ≈ 6300 Pa` (`= 0.07·ρ_ice·g·100m`, the N floor at that thickness) — the same thin-ice
   group flagged in §4 as marginal under Budd. Schoof's Coulomb cap limits basal shear to
   `Cmax·N`; where `N` is a *numerically*-motivated floor (never meant to represent real
   till strength — it exists purely to keep Budd's law stable), that floor becomes the
   *actual* yield ceiling under Schoof. At N≈6300 Pa, `Cmax·N` is only ~1000–5300 Pa, far
   too weak to hold thin ice on steep terrain; Budd's law has no such hard ceiling (drag
   keeps scaling with velocity, however weakly) and never runs away there. Schoof is far
   more sensitive to the *absolute value* of `N` than Budd; reusing Budd's numerical N floor
   under Schoof is not physically valid.

**Both findings close the Schoof avenue for now**, on physics grounds, not just numerics:
it doesn't fix Siple Coast (finding 1), and it isn't safe to deploy over the current N field
without redoing the effective-pressure treatment specifically for Coulomb sensitivity
(finding 2). Siple Coast remains open, with Budd as the friction law.

---

## 5.5 Alternative C initialisations: five failures, `nfix`'s uniform start stands

A transient relaxation of `friconly_nfix` (§7 item 7) surfaced a specific defect: 8,094
grounded vertices (0.66%) have `C` pinned exactly at the old upper bound (10.0) *and*
effective pressure sitting exactly at its numerical floor (91% overlap between the two
groups) — the optimiser wanted more resistance than the bound allowed. One such vertex
(marginally grounded, bed −401 m, thickness only ~29 m above flotation — MISI-susceptible
geometry) accelerated from 16.5 to ~1450–1470 m/yr over a 20-year relaxation (obs 4.9 m/yr)
the moment the frozen state was allowed to evolve, in both a dynamics-only and a
real-forcing run. This motivated two follow-up tests, both of which failed cleanly:

**Raising the bound (10 → 20), warm-started from `nfix`'s own converged C:** converged
normally (22 iterations, `dxmin`, cost 5.19e5 — marginally *better* than nfix's 5.26e5,
grounded RMSE 98.7). But **100% of the 8,852 previously-pinned vertices settled right back
at ~10** (median 10.00, range 9.90–10.00) — none moved toward the new ceiling. The bound was
never actually the constraint; C≈10 is a genuine local optimum for the static fit. This
disproves the "wants more than the bound" hypothesis; the transient runaway is more likely a
genuine MISI-adjacent dynamic instability at that specific marginally-grounded, deep-marine
-bed location, not a friction-parameter artifact — a harder problem with no simple parameter
fix, and left as a documented, unresolved limitation rather than chased further.

**Driving-stress analytical `C` as a fresh initial guess** (`C = sqrt(tau_d / (N*|u|^s))`,
`tau_d = rho_ice*g*H*|grad(surface)|` approximating basal shear stress): three variants, all
failed via the *same* mechanism — an enormous initial gradient (up to 4.6e15) causing m1qn3
to report **"Convergence reached (gradient satisfies stopping criterion)" after only 13–14
iterations**, not the `dxmin` criterion every successful run in this project has stopped on.
`C` barely moved from its (badly-scaled) starting values in every case:

| variant | p,q | cf101/cf103/501 | scaling factors | grounded RMSE | max vel (obs max 4194) |
|---|---|---|---|---|---|
| `budd_analytical_inv` | 3,3 | 10/100/1e-8 | default | 418,711 | 1,224,486 m/yr |
| `budd_p1q1` | 1,1 | 10/100/1e-8 | default | 2,406,704 | 6,738,615 m/yr |
| `budd_p1q1_v2` | 1,1 | 9000/40/1.6e-6 | 1.0 (explicit) | 2,406,955 | 6,738,946 m/yr |

The last two rows are the decisive comparison: `cf101` changed 900×, `control_scaling_factors`
went from default to explicit, and the outcome was **essentially bit-identical** (RMSE and
`C` median agree to 4 significant figures). Cost weighting and scaling factors were both
therefore ruled out as the cause — whatever determines where this trajectory lands is set by
the analytical starting field itself, not by these downstream optimiser settings. The `p=3`
vs `p=1` comparison (rows 1 vs 2) also failed by the same mechanism regardless of exponent
(`u ~ C^-6` under p=3 vs `C^-2` under p=1 changes *how bad* the resulting runaway is, not
*whether* the false-convergence happens at all).

**Conclusion: analytical/physically-motivated initial guesses are not compatible with this
inversion pipeline as configured**, regardless of friction-law exponent, cost weighting, or
explicit control scaling. `friconly_nfix`'s plain uniform `C_init=1.8` remains the only
starting point that has ever converged properly (`dxmin`, not a false gradient-based stop) in
this entire investigation. This joins the Schoof attempts as a second, independently-arrived
-at instance of the same lesson: alternatives to `nfix`'s exact validated setup have a 0/5
track record so far (2 Schoof, 3 analytical-init). If revisited, the next lever is m1qn3's own
internal convergence tolerances (`dxmin`/`epsg`-class settings), not p/q, weighting, or
`control_scaling_factors` — none of those were the cause here.

---

## 6. Production config status

Committed:
- `ais_0.1_param.py` — temperature-derived `rheology_B`
- `ais_0.1.py` — 502 → 501 fix, `extract_friction_inversion_domain()` (Neumann fronts),
  `friction_coupling` variable

**Promoted** (previously "still carrying known-wrong values" — all of these were measured
against the C_init=10 dead-zone bug, §2.1, and never re-validated after the fix):

| parameter | was | now |
|---|---|---|
| `friction.coefficient` init | 10.0 | **1.8** |
| Budd bounds (`friction_law_info`) | `(0.01, 1e4)` | **`(0.1, 10)`** |
| `friction_cf101 / cf103` | 1000 / 0.1 | **10 / 100** |
| friction inversion `maxsteps`/`maxiter` | 30 / 50 | **500 / 500** (`friconly_nfix` needs 192) |
| friction inversion misfit mask | all observed ice | **grounded-only** (§5.1's FIX 4) |
| `501` regularisation mask | none (whole domain) | **grounded, GL-adjacent excluded** (§3.1) |
| `501` L-curve sweep range | `1e-20 .. 1e-12` (below the unregularised point) | **`1e-3 .. 3e-1`** |
| `cluster.moduleload` | `access-issm/2025.11.0` | **`access-issm_ad/2026.05.0`** (stale-module mismatch) |
| `cluster.np` / `memory` | 32 / 100 GB (under the ~130 GB this mesh needs at 32 ranks) | **48 / 190 GB** |

The `maxsteps=30` cap and its cited OOM history were consequences of the under-provisioned
cluster memory, not of a large step budget being inherently unsafe — fixing `cluster.memory`
addresses the actual cause. Schoof's bounds (`friction_law_info`) were left as-is with a note
explaining why: ruled out on physics grounds, §5.4.

Correct as-is, unchanged: thickness floor 10 m into the base; `N_floor_frac = 0.07`;
Budd `p=q=3`; `coupling = 3`.

**Not promoted — deliberately different from `friconly_nfix`:** the production pipeline still
builds its own shelf B via its own L-curve step (`rheology_lcurve_run`, cf502) rather than
reusing `friconly_nfix`'s literal converged fields. This is correct, not an oversight —
`friconly_nfix`'s own shelf B was inverted on the pre-floor geometry (§5.1), so transplanting
its exact (C, B) pair would carry that same inconsistency into production's differently-floored
domain. Production should re-run its own inversion with the settings above and validate the
result against the ~98.9 grounded RMSE benchmark, not copy nfix's numbers directly.

---

## 7. Open items

1. **Siple Coast trunk deficit** — ratio 0.35–0.42, persistent across every configuration.
   Basal drag magnitude, mesh resolution, grounded-B freedom, and a regularised Coulomb
   friction law (Schoof) all ruled out (§5.3, §5.4). Unresolved; likely needs a bed/geometry
   or process-level look (e.g. a real, non-numerically-floored effective-pressure field),
   not more friction-law or parameter search.
2. **N-starved vertices** — 18,355 with `N/overburden = 0.007`, running 14.56× too fast.
3. **Grounding zone** — grounded ice 0–5 km from the GL is 18 % too fast (RMSE 185.9 vs 55.3 inland).
4. **Steep-slope tail** — slope > 0.1, 2.56× too fast.
5. **C field is noisy** — roughness 0.164, 3 % at bounds. Not blocking the fit, but 501
   cannot fix it; would need a different smoother (e.g. inverting `log C`).
6. ~~**Promote validated parameters** to the production config~~ — **done** (see §6):
   C init/bounds, `cf101`/`cf103`, `501` masking and sweep range, `maxsteps`/`maxiter`, cluster
   module/np/memory. Production has not yet been *re-run* with these settings — that's the
   natural next step, evaluated against the ~98.9 grounded RMSE benchmark, not item 6 itself.
7. **Forward / relaxation validation** — done, and it surfaced a real defect. Two 20-year
   relaxations of `friconly_nfix` ran to completion: dynamics-only (SMB and basal melt both
   zero) and forced (real RACMO SMB climatology + ITS_LIVE sub-shelf melt, grounded melt still
   zero pending a thermal solve). Both show grounded ice thinning without SMB / thickening
   with it (−6.6 m vs +4.2 m mean over 20 yr) — the qualitatively correct response to adding
   real forcing. But both also show the same large, coherent velocity acceleration (+150–200
   m/yr) along the Institute/Möller/Foundation grounding-line strip feeding Filchner Ice
   Shelf, in the same location regardless of forcing — traced to a marginally-grounded,
   deep-marine-bed vertex where `nfix`'s friction inversion was pinned at its upper C bound
   with N at its numerical floor (§5.5). Tested and ruled out as a bound artifact; left open
   as a specific, located, MISI-adjacent limitation on `nfix` for transient use.
8. **MISI-adjacent transient instability at the Filchner feeder grounding line** — new item
   from #7. A specific, located weakness (not domain-wide) in `friconly_nfix` for any dynamic
   (not purely diagnostic) use. Candidates not yet tested: a proper thermal solve for
   grounded/near-grounding-line effective pressure instead of the numerical N floor; a
   physically-based (not numerically-motivated) N field near flotation specifically at this
   location, informed by till/hydrology data if available.

**Closed, not to revisit:**

- Coupled-domain re-inversion of shelf B (or friction C) against a fixed counterpart field.
  11 independent attempts, 11 failures, all ending worse than their own iteration 1 (§5.2).
  The shelf-only B chain (§5.1) is the method that works; its output feeds forward into
  production as a fixed field, not as a further coupled-domain control.
- Schoof (regularised Coulomb) friction for Siple Coast (§5.4). Ruled out on physics, not
  just numerics: a forward-only Cmax sweep across the full documented range (0.17–0.84)
  left the Siple Coast trunk ratio completely unchanged (0.39–0.41 vs Budd's 0.40), and
  adjoint-based inversion of Schoof's `C` is unstable near the Coulomb cap regardless of
  solver settings (Newton vs Picard, iteration cap, control bounds, or scoping the free
  region to Siple Coast alone).
- Raising the friction `C` upper bound to fix the pinned-vertex transient runaway (§5.5).
  Disproved directly: warm-started from `nfix`'s own converged C with the bound doubled to
  20, 100% of the previously-pinned vertices settled back at ~10 anyway. Not a bound problem;
  left as a documented, unresolved MISI-adjacent dynamic-stability limitation instead.
- Analytical (driving-stress-based) initial guesses for friction `C` (§5.5). 3/3 failures, all
  via the identical mechanism — m1qn3 reports false convergence (gradient stopping criterion)
  after 13–14 iterations, `C` barely moving from its start. Tested and ruled out independently:
  friction-law exponent (p=q=3 vs p=q=1), cost-function weighting (10/100/1e-8 vs
  9000/40/1.6e-6, cost scaled 900× with no change in outcome), and explicit
  `control_scaling_factors` (vs default). `nfix`'s plain uniform `C_init=1.8` is the only
  starting point that has converged properly (`dxmin`) anywhere in this investigation.

## 8. Practical notes

- `qcat -o <jobid>` streams a running job's stdout; ISSM writes `.outlog` only at exit.
- 192 cores (4 nodes, 760 GB) gives **6.5×** the throughput of 32 cores: 35.2 vs 5.45
  iterations/hour, essentially linear scaling. Memory *rises* with rank count
  (130 GB @ 32 → 450 GB @ 192) due to halo duplication.
- Friction-only inversions need ≥ 200 iterations; the 110.6 run hit the maxsteps ceiling
  still improving.
- Cost-function values are **not comparable** between runs with different misfit masks —
  always compare grounded RMSE on a common mask.
