## Title
Friction inversion initialisation: root causes found and fixed, Budd vs Schoof + effective-pressure source comparison in progress

## Summary

The `ssa_friction_inv_sensit` friction inversion was producing unphysical results (whole-domain `vel_rmse` in the tens of thousands of m/yr, vs. real Antarctic ice topping out ~4,000 m/yr). Root-caused and fixed in stages; a robustness/physics comparison (Budd vs Schoof friction law, and two effective-pressure sources) is now running to pick the final setup.

## Root causes found

1. **Initial friction coefficient too weak.** Param initialised Schoof `C = 500`; a uniform-coefficient forward-solve sweep showed this is ~10x too small — grounded ice ran away to ~4e5 m/yr regardless of inversion coefficients. `C = 5000` brings it into a physical range.
2. **Effective pressure (N) had NaNs and negative values.** The Ehrenfeucht et al. (2024) subglacial hydrology dataset used for `md.friction.effective_pressure` has data gaps and non-physical negatives. In the Schoof law, drag is capped at `Cmax * N`, so `N <= 0` gives zero drag at that node → local blow-up. Fixed with a positive floor `N >= N_floor_frac * rho_ice * g * H` (`N_floor_frac = 0.07`).
3. **Schoof's Coulomb ceiling leaves a small residual permanently unfittable.** Even after the N floor, ~0.3-0.7% of nodes (steep/thin cells) have `Cmax*N` below the local driving stress — no friction coefficient can fit them, and the inversion (`cost_101`) plateaus fighting these nodes instead of converging on the fittable bulk. An a-priori cost mask to exclude them (matching the driving-stress condition) was tried and reverted — the static proxy didn't match the actual dynamic blow-up cells and destabilised some runs.
4. **Ice-front boundary condition.** The original setup extracts ice *including* the ice-front elements and sets Neumann BCs on the floating front, leaving ice-free front nodes unanchored — under certain friction states this produced a ~1e7 m/yr runaway at the shelf front (~20k+ nodes). Excluding the ice-front from the extraction (`ice_levelset_elements < 0` instead of `< 1`) makes the calving front the extracted-mesh boundary, where `extract()` applies Dirichlet (observed velocity) — this is far more stable. Verified both ways under Budd: Dirichlet-front bulk_rmse ~376-491 vs Neumann-front ~491-2283 depending on coefficient.

## Fix: switched friction law from Schoof to Budd (Weertman power law)

Schoof's `Cmax*N` ceiling is the structural cause of (3) above — no coefficient can raise drag past it. Switched to the Budd/Weertman power-law class (`pyissm.model.classes.friction.default`, no ceiling) as an experiment:

| | Schoof (BC-fix) | Budd (BC-fix) |
|---|---|---|
| Sensit sweep convergence | 18/25 (7 thrashed to 24h+/OOM) | **25/25 clean** |
| Blow-up nodes (uniform C) | ~4,830 (0.34%) | ~313 (0.02%) |
| Bulk vel_rmse (masked) | ~481 | ~376 |

Budd is now the working friction law. `friction_law` switch added to `ais_0.1_param.py` so Schoof stays available if needed.

## Also testing: effective-pressure source (`friction.coupling`)

Rather than relying on the patched Ehrenfeucht dataset (`coupling=3`), ISSM can compute Neff internally from a simple "uniform sheet" hydrology proxy (`coupling=2`, clamped non-negative — **do not use `coupling=0`, it allows negative N** which destabilises Budd's linear N-dependence the same way it did Schoof's ceiling).

Forward-check coefficient sweep (uniform coefficient, BC-fix):

| coeff | coupling=3 (Ehrenfeucht) | coupling=2 (internal) |
|---|---|---|
| 1 | max 14.2M, blow-up 7.77%, rmse 3607 | max 7.1M, blow-up 3.59%, rmse 2283 |
| 10 | max 1.14M, blow-up 0.08%, rmse 405 | max 280k, blow-up 0.06%, rmse 404 |
| 100 | max 134k, blow-up 0.01%, rmse 355 | max 133k, blow-up 0.012%, rmse 355 |
| 1000 | max 124k, blow-up 0.01%, rmse 353 | max 124k, blow-up 0.011%, rmse 353 |

`coupling=2` matches or beats `coupling=3` at every coefficient tested, with no external dataset dependency. A full sensit sweep under `coupling=2` is running now (isolated output dirs, doesn't touch the `coupling=3` results) to confirm this holds for the real spatially-varying inversion, not just uniform-coefficient forward checks.

## Current status

- Rheology inversion (floating ice): done, `vel_rmse` ~5-12 m/yr, L-curve picked `run_004` (`cf502=1e-17`).
- Friction sensit sweep (Budd, coupling=3, BC-fix): done, 25/25, `cf101=cf103=10` chosen as a balanced default (bulk residual is coefficient-insensitive; balanced combos avoid the systematic interior over-speed seen in absolute-heavy combos like 100/0.1).
- Friction L-curve (Budd, coupling=3, cf101=cf103=10): **running now**.
- Friction sensit sweep (Budd, coupling=2): **running now**, in parallel, isolated paths (`execution_coupling2/`, `models/AIS3_ssa_friction_inv_sensit_coupling2/`).

## Next steps

- [ ] Pick `cf502` from the friction L-curve corner.
- [ ] Compare `coupling=2` vs `coupling=3` sensit residual maps once both are loaded; decide which to carry forward.
- [ ] Run `ssa_inverted_solve` (full-domain forward solve with inverted B + friction field) and check global velocity RMSE.
- [ ] Run `ssa_relaxation` (20yr transient) to damp initialisation shock.
- [ ] Longer term: the residual is still concentrated at ice-stream shear margins (mesh-resolution artefact, law-independent) and one persistent Siple Coast mismatch region — worth a closer look once the pipeline is otherwise stable.

## Files changed

- `config/ais_0.1_param.py`: `friction_law` switch (schoof/budd), `friction_coupling` switch (2/3), N floor logic.
- `config/ais_0.1.py`: `friction_law_info()` helper (auto-detects control parameter/bounds from friction class), BC-fix extraction in all four friction blocks, new `ssa_friction_forward_check` / `ssa_friction_forward_check_budd` diagnostic steps.
