# Repository layout

The repository combines executable workflow code, input geometry, model checkpoints,
HPC execution records, and scientific development notes. The distinction between source
and generated run products matters: the many `execution_*` directories are not Python
packages or modules.

| Path | Role | Start with |
|---|---|---|
| `config/ais_0.1.py` | Main HO driver and upstream SSA setup; selects work by named steps. | [HO workflow](workflows.md#higher-order-track) |
| `config/ais_0.1_SSA.py` | Independent pure-SSA driver. | [SSA workflow](workflows.md#pure-ssa-track) |
| `config/ais_0.1_param.py` | Model masks, geometry, velocity, rheology, friction, and other fields. | [Parameterization](architecture.md#from-inputs-to-model-state) |
| `config/launch_*.pbs` | PBS launchers for drivers and standalone scripts. | [Running on Gadi](gadi.md) |
| `config/*.py` | Stage-specific analysis, finalization, plotting, and validation scripts. | [Workflow map](workflows.md) |
| `config/*.ipynb` | Development and investigation notebooks; some overlap with exported scripts. | Check the current driver before treating a notebook as canonical. |
| `assets/` | Domain outlines and constraint `.exp` inputs. | [Architecture](architecture.md) |
| `models/` | Saved model states, restart checkpoints, and selected products. | [Workflow outputs](workflows.md) |
| `execution/`, `execution_SSA/`, `execution_*` | ISSM run directories, solver inputs/outputs, and experiment artifacts. | [Runtime and outputs](architecture.md#runtime-and-output-boundaries) |
| `docs/` | Human-authored run guides, worklogs, figures, and Sphinx source under `docs/source/`. | [Development record](development-record.md) |
| `README.md`, `LICENSE`, `COPYRIGHT.txt` | Project overview and legal notices. | Project root |

## Finding the canonical implementation

For a production workflow stage, start from its step name in the driver rather than from
a similarly named notebook, old launcher output, or experiment directory. The driver's
stage block identifies the model input, settings, save path, and launch pattern. Then
follow the matching launcher and the parameterization code as needed.

Notebooks and worklogs remain valuable evidence for why choices were made. They are not
automatically interchangeable with the current driver: some are historical snapshots,
and development notes can describe superseded runs.

## Generated files and large data

Model `.nc` files, ISSM `.bin` / `.queue` / `.toolkits` files, compressed run archives,
plots, and logs are outputs or intermediate artifacts. When adding documentation or a
new workflow, keep new explanatory source in the relevant maintained directory and avoid
copying large run outputs into the docs tree. Record output names and their provenance
instead.