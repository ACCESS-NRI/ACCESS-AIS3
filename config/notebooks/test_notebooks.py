"""Test the generated step notebooks without running the pipeline.

    <pyissm-env>/bin/python test_notebooks.py --level static|setup|dry [--only GLOB] [--out DIR]

Run from config/notebooks/. Levels:

static  Parse every notebook: valid nbformat, every cell compiles, and the code is the
        driver's preamble plus that step's block (same check as make_step_notebooks.py).
setup   Execute each notebook's setup cells (imports, helpers, settings, data catalog) with
        `steps = []`, so no step runs.
dry     Execute each notebook up to its first side effect. A harness cell injected after the
        setup replaces pyissm's solve() with: model consistency check + marshalling to
        /dev/null, then stop. It also stops at the first save_model, file write, rename or
        delete outside the run's own output folder, and at any qsub. Reads, including
        solve(..., load_only=True), go through unchanged. Notebooks get extra environment
        options so they test a fresh path instead of resuming a finished run (DRY_ENV).
        HO notebooks load 23.8M-vertex models: run this level on hugemem.

Results: a table on stdout, results.json, and every executed notebook in the output folder
(default test_runs/<level>_<time>/, ignored by git). Exit status 1 if any notebook failed.
"""
import argparse
import ast
import datetime
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_step_notebooks as gen  # noqa: E402

STEPS_PREFIX = "steps = ['"

# Extra environment per step for the dry level: fresh, tagged paths rather than finished runs.
DRY_ENV = {
    'melt_gamma_tuning': {'AIS3_MELT_TAG': '_drytest'},
    'melt_gamma_tuning_ssa': {'AIS3_MELT_TAG': '_drytest'},
    'historical_dhdt_tuning': {'AIS3_HIST_TAG': '_drytest'},
    'projection_ssp': {'AIS3_PROJ_SMOKE': '1', 'AIS3_PROJ_SCENARIO': 'ssp585', 'AIS3_PROJ_GCM': 'CESM2-WACCM'},
    'projection_ssp_ssa': {'AIS3_PROJ_SMOKE': '1', 'AIS3_PROJ_SCENARIO': 'ssp585', 'AIS3_PROJ_GCM': 'CESM2-WACCM'},
}

HARNESS = r'''
# ---- dry-run harness, injected by test_notebooks.py (not part of the generated notebook) ----
import builtins, io, os, shutil, subprocess
import matplotlib.figure
import matplotlib.pyplot as plt
import xarray as xr
import pyissm

class DryRunStop(BaseException):
    """Raised where the real run would submit a job or write a file. BaseException, so the
    step's own `except Exception` handlers cannot swallow it."""

_DRY_DIR = os.path.realpath(os.environ['AIS3_DRY_DIR'])

def _allowed(p):
    if isinstance(p, int):
        return True
    p = os.path.realpath(os.fspath(p))
    return p in ('/dev/null', _DRY_DIR) or p.startswith(_DRY_DIR + os.sep)

_real_open = builtins.open
def _open(file, mode='r', *a, **k):
    if any(c in mode for c in 'wax+') and not _allowed(file):
        raise DryRunStop(f'first write: open({os.fspath(file)!r}, {mode!r})')
    return _real_open(file, mode, *a, **k)
builtins.open = _open
io.open = _open
open = _open  # IPython puts its own `open` wrapper (bound to the real io.open) in the notebook namespace

def _guard(mod, name, checked):
    """Allow mod.name only if every path argument at the indices in `checked` is allowed
    (copies only write their destination; moves and deletes change their source too)."""
    real = getattr(mod, name)
    def f(*a, **k):
        if all(_allowed(a[i]) for i in checked if i < len(a)):
            return real(*a, **k)
        raise DryRunStop(f'first file change: {mod.__name__}.{name}{tuple(map(str, a[:2]))}')
    setattr(mod, name, f)
for _m, _n, _k in [(os, 'replace', (0, 1)), (os, 'rename', (0, 1)), (os, 'remove', (0,)),
                   (os, 'unlink', (0,)), (shutil, 'move', (0, 1)), (shutil, 'copy', (1,)),
                   (shutil, 'copy2', (1,)), (shutil, 'copyfile', (1,)), (shutil, 'rmtree', (0,))]:
    _guard(_m, _n, _k)

for _cls in (xr.Dataset, xr.DataArray):
    _real_nc = _cls.to_netcdf
    def _to_netcdf(self, path=None, *a, _real=_real_nc, **k):
        if path is None or _allowed(path):
            return _real(self, path, *a, **k)
        raise DryRunStop(f'first write: to_netcdf({os.fspath(path)!r})')
    _cls.to_netcdf = _to_netcdf

def _savefig(*a, **k):
    print(f'[dry] savefig skipped')
plt.savefig = _savefig
matplotlib.figure.Figure.savefig = _savefig

def _save_model(md, path, *a, **k):
    raise DryRunStop(f'save_model({os.fspath(path)!r}) reached')
pyissm.model.io.save_model = _save_model

for _n in ('run', 'Popen', 'call', 'check_call', 'check_output'):
    _real_sp = getattr(subprocess, _n)
    def _sp(*a, _real=_real_sp, **k):
        args = a[0] if a else k.get('args', '')
        if 'qsub' in (args if isinstance(args, str) else ' '.join(map(str, args))):
            raise DryRunStop(f'qsub reached: {args}')
        return _real(*a, **k)
    setattr(subprocess, _n, _sp)

_real_solve = pyissm.model.execute.solve
_SOLUTIONS = {'stressbalance': 'StressbalanceSolution', 'sb': 'StressbalanceSolution',
              'transient': 'TransientSolution', 'tr': 'TransientSolution',
              'thermal': 'ThermalSolution', 'th': 'ThermalSolution',
              'steadystate': 'SteadystateSolution', 'masstransport': 'MasstransportSolution'}
def _dry_solve(md, solution_string, *a, **k):
    if k.get('load_only'):
        # pyissm copies the run's outlog/outbin into the working directory while loading
        # results; run the load from the test folder so that copy is allowed and lands there.
        _cwd = os.getcwd()
        os.chdir(_DRY_DIR)
        try:
            return _real_solve(md, solution_string, *a, **k)
        finally:
            os.chdir(_cwd)
    md.private.solution = _SOLUTIONS[solution_string.lower()]
    if k.get('check_consistency', True):
        pyissm.model.execute.is_model_self_consistent(md)
    pyissm.model.execute.marshall(md, '/dev/null')
    raise DryRunStop(f'solve({solution_string!r}) for {md.miscellaneous.name}: consistency check and '
                     f'marshalling passed')
pyissm.model.execute.solve = _dry_solve
print('dry-run harness active; outputs allowed only under', _DRY_DIR)
'''


def notebooks(only):
    for drv in ('ais_0.1', 'ais_0.1_SSA'):
        for p in sorted((HERE / drv).glob('[0-9][0-9]_*.ipynb')):
            rel = f'{drv}/{p.name}'
            if only is None or any(fnmatch.fnmatch(rel, g) for g in only.split(',')):
                yield rel, p


def step_of(nb):
    for c in nb.cells:
        if c.cell_type == 'code' and c.source.startswith(STEPS_PREFIX):
            return c.source[len(STEPS_PREFIX):].split("'")[0]
    raise ValueError('no steps cell')


def static_check(rel, nb):
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == 'code':
            ast.parse(c.source)
    driver = nb.metadata['ais3_generated_from']
    src = (HERE.parent / driver).read_text()
    tree = ast.parse(src)
    blocks = dict(gen.step_ifs(tree))
    first = gen.step_ifs(tree)[0][1]
    pre = [n for n in tree.body if n.end_lineno < first.lineno]
    code = [c.source for c in nb.cells if c.cell_type == 'code'
            and not c.source.startswith('import os\n\n# Uncomment') and not c.source.startswith(STEPS_PREFIX)]
    got = gen._dump(ast.parse('\n\n'.join(code)).body)
    if got != gen._dump(pre + list(blocks[step_of(nb)].body)):
        raise AssertionError(f'code differs from {driver}: regenerate with make_step_notebooks.py')


def prepare(nb, level):
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and c.source.startswith(STEPS_PREFIX))
    if level == 'setup':
        nb.cells = nb.cells[:k] + [nbformat.v4.new_code_cell("steps = []\nprint('SETUP OK')")]
    else:
        nb.cells = nb.cells[:k] + [nbformat.v4.new_code_cell(HARNESS)] + nb.cells[k:]
    return nb


def execute(rel, nb, level, out, timeout):
    step = step_of(nb)
    env = dict(DRY_ENV.get(step, {})) if level == 'dry' else {}
    env['AIS3_DRY_DIR'] = str(out)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    prepare(nb, level)
    client = NotebookClient(nb, kernel_name='pyissm', timeout=timeout,
                            resources={'metadata': {'path': str(HERE)}})
    status, detail = 'PASS', ('setup cells ran' if level == 'setup' else 'ran to the end without a side effect')
    try:
        client.execute()
    except CellTimeoutError as e:
        status, detail = 'FAIL', f'cell timeout ({timeout} s)'
    except CellExecutionError as e:
        err = next((o for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', [])
                    if o.get('output_type') == 'error'), None)
        ename, evalue = (err['ename'], err['evalue']) if err else (type(e).__name__, str(e)[:300])
        if ename == 'DryRunStop':
            status, detail = 'PASS', f'stopped at {evalue}'
        elif ename == 'SystemExit':
            status, detail = 'GUARD', f'step stopped itself: {evalue}'
        else:
            status, detail = 'FAIL', f'{ename}: {evalue}'
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    nbformat.write(nb, out / rel.replace('/', '__'))
    return status, detail, env


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--level', choices=['static', 'setup', 'dry'], required=True)
    ap.add_argument('--only', help="comma-separated globs on '<driver>/<NN>_<step>.ipynb', e.g. 'ais_0.1_SSA/*,ais_0.1/0*'")
    ap.add_argument('--out', type=Path)
    ap.add_argument('--timeout', type=int, default=4 * 3600, help='per-cell timeout in seconds')
    a = ap.parse_args()
    stamp = os.environ.get('PBS_JOBID', '').split('.')[0] or datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = (a.out or HERE / 'test_runs' / f'{a.level}_{stamp}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'level={a.level} out={out}', flush=True)
    results = []
    for rel, path in notebooks(a.only):
        nb = nbformat.read(path, 4)
        t = time.time()
        env = {}
        if a.level == 'static':
            try:
                static_check(rel, nb)
                status, detail = 'PASS', 'valid, compiles, matches its driver'
            except Exception as e:
                status, detail = 'FAIL', f'{type(e).__name__}: {e}'
        else:
            status, detail, env = execute(rel, nb, a.level, out, a.timeout)
        dt = time.time() - t
        results.append({'notebook': rel, 'status': status, 'detail': detail, 'seconds': round(dt), 'env': env})
        print(f'{status:5s} {dt:7.0f}s  {rel:52s} {detail}', flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=1))
    n_fail = sum(r['status'] == 'FAIL' for r in results)
    print(f'\n{len(results)} notebooks: {len(results) - n_fail} ok, {n_fail} failed. Results in {out}')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
