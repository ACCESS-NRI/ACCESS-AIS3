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
| `config/site.env`, `config/ais3_site.py`, `config/submit` | Site-specific environment defaults, path resolution, and a project/storage-aware `qsub` wrapper. | [Running on Gadi](gadi.md) |
| `config/notebooks/` | Generated step-by-step notebooks, sourced from the Python drivers. | Follow the driver and notebook-generation instructions in the [run guide](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/run_guide.md). |
| `config/*.py` | Stage-specific analysis, finalization, plotting, and validation scripts. | [Workflow map](workflows.md) |
| `config/*.ipynb` | Development and investigation notebooks; some overlap with exported scripts. | Check the current driver before treating a notebook as canonical. |
| `assets/` | Domain outlines and constraint `.exp` inputs. | [Architecture](architecture.md) |
| `assets/larmip3/`, `assets/larmip2/` | LARMIP sector masks and comparison data used by the projection sensitivity work. | Projection details in the [pipeline worklog](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/post_inversion_pipeline_worklog.md). |
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