# ACCESS-AIS3 documentation

ACCESS-AIS3 is an open community whole-Antarctic configuration of the Ice-sheet and
Sea-level System Model (ISSM) for the ACCESS climate model suite. This site explains
how the repository is organized, how its model workflows fit together, and where to
find the operational guidance.

> **Scope:** these pages are a manually maintained guide to the project. They do not
> import or execute workflow scripts to generate API documentation. For pyISSM's API
> and installation instructions, use the [pyISSM documentation](https://pyissm.readthedocs.io/).

## Start here

The [development record dashboard](../index.html) remains the project website homepage;
this documentation guide is published alongside it.

- [Architecture](architecture.md): the model-building path and the separate HO and SSA tracks.
- [Repository layout](repository-layout.md): where maintained code, inputs, outputs, and records live.
- [Workflow map](workflows.md): stage responsibilities, inputs, outputs, and status caveats.
- [Running on Gadi](gadi.md): the job model and safe step selection.
- [Validation and science notes](validation.md): how to interpret results and where detailed evidence lives.
- [Contributing and documentation](contributing.md): safe ways to change workflows and maintain this site.
- [Development record](development-record.md): project status dashboard and investigation records.

## The short version

The repository is organized around scripts that configure and run pyISSM models, not
around a reusable Python package. The main HO pipeline is in `config/ais_0.1.py`; the
separate pure-SSA pipeline is in `config/ais_0.1_SSA.py`. Both use
`config/ais_0.1_param.py` as the parameterization reference, but the SSA driver is
deliberately self-contained and does not import the HO driver.

The model inputs are assembled from the geometry and constraints in `assets/` plus
datasets accessed through ccdtools. A PBS launcher starts a driver job; pyISSM then
submits the ISSM computation. Model checkpoints and execution files are written under
`models/` and execution directories, rather than being part of the documentation source.

```text
assets + external datasets
            |
            v
  domain -> mesh -> parameterized model
            |
            +--> SSA inversion -> SSA historical run -> SSA projections
            |
            +--> SSA initialization -> HO thermal/inversion -> melt/relaxation
                                                        -> HO historical/projections
```

## Project status

The project is in active development. Pipeline stages do not all have the same maturity;
some are validated, some are experimental, and some remain scaffolds. Treat the
[development record](development-record.md) and dated worklogs as status snapshots, not
as a guarantee that every output is production-ready.

```{toctree}
:maxdepth: 2
:caption: Project guide

architecture
repository-layout
workflows
gadi
validation
contributing
development-record
```