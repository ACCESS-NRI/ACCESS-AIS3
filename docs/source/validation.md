# Validation and science notes

Model completion, numerical convergence, and scientific validation are different claims.
Keep them distinct in status pages and in any new results summary.

## Three levels of evidence

1. **Execution:** the driver and solver finished and wrote an output.
2. **Numerical behavior:** the solve converged under its configured criteria, and the
   saved checkpoint contains the expected fields and final state.
3. **Scientific validation:** diagnostics test the intended physical quantity against
   observations, with suitable spatial coverage, area weighting, and time sampling.

A stage can satisfy one level without satisfying the next. Record what was checked and
which output it applies to.

## Historical-run metrics

The worklogs caution that a single spatial RMSE against a final-year observation slice
does not establish temporal skill or reveal a domain-wide bias. For historical evolution,
prefer the documented area-weighted, fixed-coverage, per-basin time series and state the
comparison period and observational product. The scripts under `config/check_*` provide
examples of the current checks; see the run and pipeline worklogs for which check applies
to each result.

## Boundary conditions and inversion choices

Subdomain extraction can introduce new boundaries and therefore change the boundary
conditions relative to the full-continent solve. This is especially relevant around the
floating calving front and grounding line. The root [boundary-condition notes](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/RUN_GUIDE.md)
and [inversion worklog](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/inversion_worklog.md) explain the current choices and evidence.

Likewise, friction law, effective-pressure source, mesh/geometry treatment, cost-function
weights, and checkpoint provenance are part of the result—not incidental settings. Link
reported diagnostics to the exact model file and configuration used.

## Higher-order projection caveat

The HO projection chain was implemented and extended beyond 2100, but its current long-term
thickness output is not suitable for interpretation without qualification. The HO initialization
contains observed-velocity Dirichlet pins on about 8.4% of grounded area. In pinned columns,
velocity cannot adjust as thickness changes; extreme thickness growth appeared in the late HO
projection years. Releasing the pins in a diagnostic solve instead reintroduced extreme speeds,
so this is not a simple constraint-removal fix. The CTRL and SSP585 HO chains have been extended
and resumed with watched-solve/retry handling, but their long-term results remain unqualified while
the growth issue is investigated. The SSA projections are the primary projection set; the HO
comparison is currently useful through 2100 with this caveat. See pipeline worklog sections
6.20–6.22 for the diagnosis and latest operational notes.

The SSA projection work also includes prescribed shelf-collapse masks and LARMIP melt
sensitivities. These are scenario/parameter experiments, not observations or probabilistic
predictions. For collapse experiments, detached floating remnants are removed as unanchored
pieces; state that implementation choice when discussing their results.

## Evidence and status records

- [Inversion worklog](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/inversion_worklog.md): present-day friction/rheology investigation,
  including corrections and superseded experiments.
- [Post-inversion pipeline worklog](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/post_inversion_pipeline_worklog.md): HO and SSA
  downstream stages and status history.
- [Projection experiments](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/post_inversion_pipeline_worklog.md#6-projections-ssa-track-setup-and-melt-sensitivity-check): forcing design and projection
  experiment matrix.
- [Gadi run guide](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/run_guide.md): stage inputs/outputs and operational caveats.

These records are detailed and date-sensitive. The summary pages here should remain
high-level, and should not silently flatten competing results into a single “best” number.