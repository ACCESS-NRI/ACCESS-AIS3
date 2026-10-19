# Architecture

ACCESS-AIS3 composes an Antarctic model by configuring ISSM through pyISSM. The
repository's primary unit of organization is a **workflow stage**: a named block in a
driver script that loads or constructs a model, applies a physical or numerical
operation, runs ISSM when required, and saves a checkpoint.

## From inputs to model state

1. **Domain and constraints.** The `.exp` geometry files in `assets/` describe the
   continental domain and selected constraint regions. The domain file includes an
   ocean buffer around the ice sheet.
2. **Observations and geophysical data.** The scripts load gridded products through
   ccdtools' data catalog and interpolate fields onto the model mesh. Some ancillary
   inputs are read directly from project storage; the repository does not contain all
   source datasets.
3. **Mesh.** The primary workflow constructs a triangular Antarctic mesh, with
   resolution adapted to velocity and ice/ocean masks. The SSA configuration is 2-D;
   the HO workflow extrudes the 2-D mesh into vertical layers.
4. **Parameterization.** `config/ais_0.1_param.py` assigns masks, geometry, observed and
   initial velocities, rheology, friction, and other model fields. It is the key place
   to trace how external inputs become ISSM quantities.
5. **Physics and time evolution.** The workflow drivers select inversion, stress-balance,
   thermal, relaxation, historical, or projection stages. Each stage consumes a model
   checkpoint and may produce a new `.nc` model state, diagnostics, or time-series output.

## Two distinct tracks

| Track | Driver | Model | Purpose |
|---|---|---|---|
| Higher order (HO) | `config/ais_0.1.py` | 3-D, vertically extruded mesh | Thermal state, HO friction work, relaxation, historical evolution, and a projection scaffold. |
| Pure SSA | `config/ais_0.1_SSA.py` | 2-D mesh | A separate, lower-cost workflow for inversion, historical evolution, and implemented projection experiments. |

The SSA driver is intentionally independent: it does not import or execute the HO driver,
and keeps its own step selection and settings. This prevents queued jobs from sharing
mutable driver state, at the cost of some duplicated workflow logic. Do not assume that
the two tracks use identical friction laws, geometry hand-offs, resource requirements, or
validation status.

## Runtime and output boundaries

`config/launch_*.pbs` launchers run on Gadi. The outer driver loads and marshals a model;
pyISSM's cluster interface submits the inner ISSM compute job. After waiting for the
result, the driver loads it and writes model files. Thus a submitted workflow step may
involve two PBS jobs and two queue waits.

- `models/` contains named model states, restart checkpoints, and selected diagnostic outputs.
- `execution/` and related execution directories contain ISSM run inputs, binaries, queue
  files, and solver logs.
- `config/logs/` and launcher output files contain job-driver logs.

These outputs are operational artifacts, not source modules. For the exact current
inputs, outputs, and restart rules, use the [Gadi run guide](gadi.md) and
[workflow map](workflows.md).

## Source-of-truth guide

- Driver stage logic: `config/ais_0.1.py` and `config/ais_0.1_SSA.py`.
- Model fields and physical parameterization: `config/ais_0.1_param.py`.
- Cluster submission details: the corresponding `config/launch_*.pbs` launcher.
- Detailed design history and evidence: `docs/inversion_worklog.md`,
  `docs/post_inversion_pipeline_worklog.md`, and `docs/projection_experiments.md`.

The worklogs preserve investigation history and may contain earlier conclusions followed
by corrections. Use their reconciliation notes and dates; do not treat every historical
paragraph as a statement of the current configuration.