# Running on Gadi

This page explains the execution model and the safest way to choose a stage. It is not a
replacement for the detailed [run guide](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/run_guide.md), which includes current resource
settings, file paths, restart instructions, known failure modes, and monitoring commands.

## Two-job execution model

1. Submit a `config/launch_*.pbs` launcher from `config/`.
2. The launcher starts an outer Python driver job.
3. The driver loads and marshals a model, then pyISSM submits the inner ISSM compute job.
4. The driver waits for the inner job, loads its results, and saves the model checkpoint.

The outer job may therefore wait for both the inner queue delay and the compute time. A
launcher that appears to be waiting indefinitely can also indicate a rejected inner
submission; the run guide describes how to inspect the generated queue file.

## Select a step with the environment

For the main HO/upstream driver, use `AIS3_STEPS` at submission time rather than editing
the shared `steps` list while jobs may still be queued:

```bash
cd config
qsub -v AIS3_STEPS=ho_thermal_steadystate launch_ho_relaxation.pbs
```

For the pure-SSA driver, select a step the same way:

```bash
cd config
qsub -v AIS3_STEPS=historical_dhdt_tuning_ssa launch_ais_0.1_SSA.pbs
```

Use the launcher and resource settings documented for that stage. Do not infer that all
launchers have interchangeable PBS resource requests: HO memory requirements are much
higher than SSA's.

## Run a step as a notebook

Each step can also be run from its generated notebook (see [step notebooks](workflows.md#step-notebooks)).
Submit `launch_notebook.pbs` from `config/notebooks/`, passing the notebook path and the
step's resources. Copy both from the notebook's header rather than guessing them:

```bash
cd config/notebooks
qsub -q hugemem -l ncpus=48,mem=1450GB,walltime=45:00:00 \
     -v NB=ais_0.1/15_ho_relaxation.ipynb launch_notebook.pbs
```

- Step options go in the same `-v` list, e.g. `-v NB=...,AIS3_HIST_TAG=_relaxfix`.
- The executed copy is written to `executed/<name>.<jobid>.ipynb` next to the notebook,
  so the source notebook is never overwritten.
- `PYISSM_ENV` selects the Python environment. It must provide `jupyter` and a kernel
  named `pyissm`. The default points to one user's environment, so set it (or edit the
  launcher) when running from another account.

The same two-job model applies: the notebook job is the outer driver and still submits
the inner ISSM job.

## Before running

- Confirm the stage's input checkpoint and output name in the [workflow map](workflows.md)
  and [full run guide](https://github.com/ACCESS-NRI/ACCESS-AIS3/blob/main/docs/run_guide.md).
- Check the active step/status in the source worklogs; dated status may be stale.
- Confirm the selected Python environment, ISSM build/module, project/storage, paths, and
  user settings for your checkout.
- Choose a new tag or archive an existing output before rerunning. Several stages refuse
  to overwrite outputs; others use the existing checkpoint to resume.
- Check the relevant queue and resource limits before requesting a long HO solve.

The detailed run guide currently documents Gadi project `au88` and contains user- and
filesystem-specific paths. It must be adapted before use from another account or checkout.