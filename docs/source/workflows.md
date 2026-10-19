# Workflow map

The main driver exposes named stages in `config/ais_0.1.py`. The independent pure-SSA
driver exposes its own stages in `config/ais_0.1_SSA.py`. The tables below summarize the
workflow shape; dated status details can change and are maintained in the linked
worklogs and run guide.

## Shared preparation and inversion

| Stage | Responsibility | Typical product |
|---|---|---|
| `process_domain` | Prepare the Antarctic model-domain outline. | `assets/ais_domain.exp` |
| `mesh` | Generate and refine the triangular mesh. | `models/AIS3_mesh.nc` |
| `param` | Apply data, masks, geometry, observations, and physical parameterization. | `models/AIS3_param.nc` |
| SSA rheology/friction inversion stages | Explore sensitivities and regularization, then select fields for an inversion. | Run directories and inversion result files |
| `ssa_inverted_solve` | Assemble and solve the main driver's present-day SSA state. | `models/AIS3_inverted.nc` |

Exact step names and hand-offs are defined in the driver; this table is an orientation,
not a substitute for reading the stage implementation before changing it.

## Higher-order track

The HO sequence extends the present-day initialization with vertically resolved physics.
As documented in the 2026-09-29 pipeline record, the intended sequence is:

1. `ho_thermal_steadystate` — establish a thermal state from the inverted model.
2. `ho_friction_inv` — refine friction under HO physics; this has used chunked restarts.
3. `melt_gamma_tuning` — apply the calibrated basal-melt parameterization.
4. `ho_relaxation` — produce a relaxed state before transient evolution.
5. `historical_dhdt_tuning` — evolve the historical period, with chunked warm restarts.
6. `projection_ssp` — HO projection scaffold; forcing interpolation and prerequisites remain
   explicit development concerns.

Consult the [post-inversion pipeline worklog](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/post_inversion_pipeline_worklog.md)
for the dated stage status and known qualification issues. In particular, a completed
solver job is not automatically equivalent to a validated science product.

## Pure-SSA track

The SSA driver has its own downstream chain:

1. `ssa_inverted_solve_budd` — assemble the SSA initial state using the saved Budd result.
2. `ssa_relaxation_budd` — run the SSA relaxation stage; its geometry hand-off is deliberately
   different from the HO track.
3. `melt_gamma_tuning_ssa` — apply the selected melt calibration.
4. `historical_dhdt_tuning_ssa` — run the historical transient.
5. `projection_ssp_ssa` — run the SSA projection experiments in chained chunks.

See [projection experiments](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/projection_experiments.md) for projection forcing,
experiment naming, output products, and operational choices. For SSA historical validation
caveats, see [validation](validation.md).

## Data-flow rule of thumb

Each stage should be documented in terms of four things:

- **Input state:** model file or upstream result it loads.
- **Transformation:** physical operation and the important numerical choices.
- **Output state:** file or execution result it writes.
- **Qualification:** whether the result is scaffolded, exploratory, completed, or validated.

That record makes it easier to understand dependencies without treating output filenames
or folder names as sufficient documentation.