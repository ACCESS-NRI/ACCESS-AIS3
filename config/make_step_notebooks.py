"""Generate one runnable Jupyter notebook per pipeline step from the driver scripts.

    <pyissm-env>/bin/python make_step_notebooks.py        (run from config/)

The drivers (ais_0.1.py, ais_0.1_SSA.py) stay the single source of truth: each notebook is
their preamble (imports, helpers, settings, data catalog) followed by one step's
`if '<step>' in steps:` block, dedented and split into cells. Reads, writes and AIS3_* run
options are taken from the code; resources come from the step's existing PBS launcher. Edit a
driver, then rerun this script. Never edit the generated notebooks by hand.

Output: notebooks/<driver>/<NN>_<step>.ipynb. Run one with notebooks/launch_notebook.pbs (the
command is in each notebook's header) or open it in a Jupyter session with enough memory for
the step's outer driver.
"""
import ast
import datetime
import re
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CONFIG = Path(__file__).resolve().parent
OUT = CONFIG / 'notebooks'

# One line per step (what it is for). Everything else in the header is read from the code.
PURPOSE = {
    'process_domain': 'Build the model domain outline from the MEaSUREs coastline.',
    'mesh': 'Generate the 2D mesh.',
    'param': 'Parameterise the mesh (geometry, materials, friction law; `ais_0.1_param.py`).',
    'ssa_rheology_floating_inv_sensit': 'Cost-function weight sweep for the floating-ice rheology (B) inversion.',
    'ssa_rheology_floating_inv_lcurve': 'cf502 regularisation L-curve for the floating-ice B inversion; '
                                        '`rheology_lcurve_run` names the chosen run.',
    'ssa_friction_forward_check': 'Forward SSA solves at uniform Schoof friction to bracket the initial C.',
    'ssa_friction_forward_check_budd': 'Forward SSA solves at uniform Budd friction to bracket the initial C.',
    'ssa_friction_inv_sensit': 'cf101/cf103 weight sweep for the grounded friction inversion.',
    'ssa_friction_inv_lcurve': 'Unregularised grounded friction inversion (the baseline for the regularised run).',
    'ssa_friction_inv_reg_lcurve': 'cf501-regularised friction inversion L-curve, warm-started from the baseline.',
    'ssa_inverted_solve': 'Combine the inverted B and C on the full continent and run a forward SSA solve '
                          '(the production Schoof state that seeds the HO track).',
    'ssa_relaxation': 'SSA relaxation of the inverted state (comparison baseline only).',
    'ho_thermal_steadystate': 'HO velocity solve, then thermal steady state; recomputes `rheology_B` from temperature.',
    'ho_friction_inv': 'HO friction re-inversion in chunks; each submission resumes from the last saved chunk.',
    'melt_gamma_tuning': 'Apply the validated ocean-melt calibration (gamma_0=300, refit deltaT) and restore '
                         'floored thickness. The calibration search itself is in the standalone melt scripts.',
    'ho_relaxation': '1-yr HO relaxation (RACMO SMB + calibrated melt); its end state becomes the saved geometry.',
    'historical_dhdt_tuning': 'HO 1995-2019 transient in 6-yr chunks, compared with CPOM dH/dt. '
                              'One submission runs one chunk.',
    'projection_ssp': 'HO projections from the historical end state. Each chunk resubmits its successor '
                      'through `launch_projection_ho.pbs`, so later chunks run the driver, not this notebook.',
    'ssa_inverted_solve_budd': 'Forward SSA solve with the Budd p=q=1 friction field (SSA track start).',
    'ssa_relaxation_budd': '20-yr SSA relaxation. Downstream SSA steps deliberately start from the inverted geometry.',
    'melt_gamma_tuning_ssa': 'Apply the validated SSA ocean-melt calibration (gamma_0=300, refit deltaT).',
    'historical_dhdt_tuning_ssa': 'SSA 1995-2019 transient in one run, compared with CPOM dH/dt.',
    'projection_ssp_ssa': 'SSA projections from the historical end state. Each chunk resubmits its successor '
                          'through `launch_projection_ssa.pbs`, so later chunks run the driver, not this notebook.',
}

# Launcher whose PBS resources the step's outer driver needs (the notebook replaces only the
# command, not the resources).
LAUNCHER = {
    'ais_0.1.py': {
        'ho_thermal_steadystate': 'launch_ho_relaxation.pbs',
        'melt_gamma_tuning': 'launch_ho_relaxation.pbs',
        'ho_relaxation': 'launch_ho_relaxation.pbs',
        'historical_dhdt_tuning': 'launch_historical_dhdt_tuning.pbs',
        'projection_ssp': 'launch_projection_ho.pbs',
        None: 'launch_ho_friction_inv.pbs',   # every other step: normal-queue outer driver
    },
    'ais_0.1_SSA.py': {
        'projection_ssp_ssa': 'launch_projection_ssa.pbs',
        None: 'launch_ais_0.1_SSA.pbs',
    },
}

HEADER_RULE = re.compile(r'^## -{3,}\s*$')


def step_ifs(tree):
    """Top-level `if '<step>' in steps:` nodes, in file order."""
    out = []
    for node in tree.body:
        t = node.test if isinstance(node, ast.If) else None
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Constant)
                and isinstance(t.ops[0], ast.In) and isinstance(t.comparators[0], ast.Name)
                and t.comparators[0].id == 'steps'):
            out.append((t.left.value, node))
    return out


def leading_comment_start(lines, lineno):
    """1-based first line of the comment block directly above line `lineno`."""
    i = lineno
    while i - 1 >= 1 and lines[i - 2].lstrip().startswith('#'):
        i -= 1
    return i


def comments_to_markdown(block):
    """Turn a run of `#` comment lines into markdown; `## ---` rules are dropped and the
    `## Title` between them becomes a heading."""
    md, para = [], []
    for raw in block:
        s = raw.strip()
        if not s.startswith('#'):
            if para:
                md.append(' '.join(para)); para = []
            continue
        if HEADER_RULE.match(s):
            continue
        if s.startswith('## '):
            if para:
                md.append(' '.join(para)); para = []
            md.append('### ' + s[3:].strip())
            continue
        text = s.lstrip('#').strip()
        if text:
            para.append(text)
        elif para:
            md.append(' '.join(para)); para = []
    if para:
        md.append(' '.join(para))
    return '\n\n'.join(md)


def preamble_cells(lines, tree, end_line):
    """Split the preamble at `## ---- / ## Title / ## ----` headers, between top-level statements."""
    stmts = [n for n in tree.body if n.end_lineno <= end_line]
    starts = [1]
    titles = {1: None}
    for n in stmts:
        s = leading_comment_start(lines, n.lineno)
        block = lines[s - 1:n.lineno - 1]
        for j, raw in enumerate(block):
            if HEADER_RULE.match(raw.strip()) and j + 1 < len(block) and block[j + 1].startswith('## ') \
                    and not HEADER_RULE.match(block[j + 1].strip()):
                if s not in titles:
                    starts.append(s)
                    titles[s] = block[j + 1][3:].strip()
                break
    starts.append(end_line + 1)
    cells = []
    for a, b in zip(starts, starts[1:]):
        code = '\n'.join(lines[a - 1:b - 1]).strip('\n')
        if not code.strip():
            continue
        ast.parse(code)  # every chunk is whole statements
        if titles.get(a):
            cells.append(new_markdown_cell(f'#### {titles[a]}'))
        cells.append(new_code_cell(code))
    return cells


def progress_print(node):
    """True for `print("-- ...")` / `print(f"-- ...")`, the drivers' progress markers."""
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print' and node.value.args):
        return False
    a = node.value.args[0]
    if isinstance(a, ast.JoinedStr) and a.values and isinstance(a.values[0], ast.Constant):
        a = a.values[0]
    return isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.lstrip().startswith('--')


def _dump(nodes):
    """ast.dump of `nodes`, with docstring indentation normalised (dedenting the step block shifts
    the continuation lines of docstrings in nested functions; nothing else may change)."""
    import copy
    import inspect
    out = []
    for n in nodes:
        n = copy.deepcopy(n)
        for sub in ast.walk(n):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and sub.body \
                    and isinstance(sub.body[0], ast.Expr) and isinstance(sub.body[0].value, ast.Constant) \
                    and isinstance(sub.body[0].value.value, str):
                sub.body[0].value.value = inspect.cleandoc(sub.body[0].value.value)
        out.append(ast.dump(n))
    return out


def step_body_cells(lines, node):
    """The if-block body, dedented and split at progress prints. Falls back to one cell holding the
    whole `if` block if the body cannot be dedented exactly."""
    if node.lineno != node.body[0].lineno - 1 and not lines[node.lineno - 1].rstrip().endswith(':'):
        raise ValueError(f'unexpected if header at line {node.lineno}')
    body = lines[node.lineno:node.end_lineno]
    if all(not l.strip() or l.startswith('    ') for l in body):
        dedented = [l[4:] for l in body]
        code = '\n'.join(dedented)
        sub = ast.parse(code)
        if _dump(sub.body) == _dump(node.body):
            starts = [1]
            for n in sub.body[1:]:
                if progress_print(n):
                    starts.append(leading_comment_start(dedented, n.lineno))
            starts.append(len(dedented) + 1)
            cells = []
            for a, b in zip(starts, starts[1:]):
                chunk = '\n'.join(dedented[a - 1:b - 1]).strip('\n')
                if chunk.strip():
                    ast.parse(chunk)
                    cells.append(new_code_cell(chunk))
            return cells, True
    whole = '\n'.join(lines[node.lineno - 1:node.end_lineno])
    return [new_code_cell(whole)], False


def segment_strings(src, node, func_names):
    """Source of the first argument of every call to one of `func_names` (e.g. save_model)."""
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and n.args:
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
            if name in func_names:
                arg = n.args[-1] if name == 'save_model' else n.args[0]
                seg = ast.get_source_segment(src, arg)
                if isinstance(arg, ast.Name):  # show what the variable is set to in this step
                    vals = [ast.get_source_segment(src, a.value) for a in ast.walk(node)
                            if isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == arg.id
                                                                 for t in a.targets)]
                    if vals:
                        seg = f'{arg.id} = ' + ' | '.join(dict.fromkeys(vals))
                if seg and seg not in found:
                    found.append(seg)
    return found


def env_options(src, nodes):
    """AIS3_* environment variables read by `nodes`: {name: default source or None}."""
    opts = {}
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'get' \
                    and ast.get_source_segment(src, n.func.value) == 'os.environ' and n.args \
                    and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith('AIS3_'):
                opts.setdefault(n.args[0].value, ast.get_source_segment(src, n.args[1]) if len(n.args) > 1 else None)
            elif isinstance(n, ast.Subscript) and ast.get_source_segment(src, n.value) == 'os.environ' \
                    and isinstance(n.slice, ast.Constant) and str(n.slice.value).startswith('AIS3_'):
                opts.setdefault(n.slice.value, None)
    opts.pop('AIS3_STEPS', None)
    return opts


def preamble_env_for_step(src, tree, end_line, step_node):
    """Preamble assignments like HIST_TAG = os.environ.get('AIS3_HIST_TAG', '') that the step uses."""
    used = {n.id for n in ast.walk(step_node) if isinstance(n, ast.Name)}
    picked = []
    for n in tree.body:
        if n.end_lineno > end_line or not isinstance(n, ast.Assign):
            continue
        names = {t.id for t in n.targets if isinstance(t, ast.Name)}
        if names & used and env_options(src, [n]):
            picked.append(n)
    return env_options(src, picked)


def pbs_resources(launcher):
    text = (CONFIG / launcher).read_text()
    get = lambda pat: (re.search(pat, text) or [None, '?'])[1]
    return {'queue': get(r'#PBS -q (\S+)'), 'ncpus': get(r'ncpus=(\d+)'),
            'mem': get(r'mem=(\d+\w+)'), 'walltime': get(r'walltime=([\d:]+)')}


def header_markdown(driver, step, purpose, banner, reads, writes, opts, res, launcher, nb_rel, dedented):
    q = (f"qsub -q {res['queue']} -l ncpus={res['ncpus']},mem={res['mem']},walltime={res['walltime']} "
         f"-v NB={nb_rel}")
    q += ' launch_notebook.pbs'
    lst = lambda xs: '\n'.join(f'- `{x}`' for x in xs) if xs else '- (none found in the code)'
    parts = [
        f'# `{step}`',
        f'> Generated from `config/{driver}` by `config/make_step_notebooks.py` on '
        f'{datetime.date.today().isoformat()}. Edit the driver and regenerate; changes made here are '
        f'overwritten.',
        purpose,
        f'**Reads** (`load_model` calls in this step):\n{lst(reads)}',
        f'**Writes** (`save_model` calls in this step):\n{lst(writes)}',
        f'**Run as a batch job** (from `config/notebooks/`; resources copied from `config/{launcher}`):\n\n'
        f'```bash\n{q}\n```\n\n'
        + (f'Add run options to the `-v` list, e.g. `NB=...,{next(iter(opts))}=<value>`. ' if opts else '')
        + 
        f'The executed copy is written next to this notebook as `executed/<name>.<jobid>.ipynb`. '
        f'Interactive use needs a Jupyter session with at least those resources.',
        '**Run options**: set them in the next cell (interactive) or with `qsub -v` (batch).'
        if opts else '**Run options**: none.',
        'Status, results and known failures: [`docs/run_guide.md`](../../../docs/run_guide.md) and the '
        'worklogs in `docs/`.',
    ]
    if not dedented:
        parts.append('_The step block could not be split into cells, so it is one cell with its `if`._')
    return '\n\n'.join(parts)


def build(driver):
    path = CONFIG / driver
    src = path.read_text()
    lines = src.split('\n')
    tree = ast.parse(src)
    blocks = step_ifs(tree)
    first = blocks[0][1]
    pre_end = max(n.end_lineno for n in tree.body if n.end_lineno < first.lineno)
    pre_cells = preamble_cells(lines, tree, pre_end)
    outdir = OUT / path.stem
    outdir.mkdir(parents=True, exist_ok=True)
    launchers = LAUNCHER[driver]
    written = []
    prev_end = pre_end
    for i, (step, node) in enumerate(blocks):
        if step not in PURPOSE:
            sys.exit(f'no PURPOSE entry for step {step!r} in {driver} -- add one')
        gap = lines[prev_end:node.lineno - 1]
        prev_end = node.end_lineno
        body_cells, dedented = step_body_cells(lines, node)
        banner = None
        for n in node.body[:4]:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and n.value.args:
                a = n.value.args[0]
                if isinstance(a, ast.JoinedStr) and len(a.values) == 1:
                    a = a.values[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.strip() \
                        and not set(a.value.strip()) <= set('-'):
                    banner = a.value.strip(); break
        opts = {**preamble_env_for_step(src, tree, pre_end, node), **env_options(src, [node])}
        launcher = launchers.get(step, launchers[None])
        nb_name = f'{i:02d}_{step}.ipynb'
        nb_rel = f'{path.stem}/{nb_name}'
        cells = [new_markdown_cell(header_markdown(
            driver, step, PURPOSE[step], banner,
            segment_strings(src, node, {'load_model'}), segment_strings(src, node, {'save_model'}),
            opts, pbs_resources(launcher), launcher, nb_rel, dedented))]
        opt_lines = ['import os', '',
                     '# Uncomment to set an option for an interactive run. Values passed with',
                     '# `qsub -v` already arrive in the environment. Defaults shown are the code\'s own.']
        for k, d in opts.items():
            opt_lines.append(f"# os.environ['{k}'] = ...   # code default: {d if d is not None else '(unset)'}")
        if opts:
            cells.append(new_code_cell('\n'.join(opt_lines)))
        cells.append(new_markdown_cell(f'## Setup\n\nThe preamble of `{driver}`: imports, helpers, '
                                       f'settings and the data catalog.'))
        cells += [nbformat.from_dict(c) for c in pre_cells]
        cells.append(new_markdown_cell(f'## Step `{step}`'))
        cells.append(new_code_cell(f"steps = ['{step}']  # this notebook's step; overrides the driver "
                                   f"default and AIS3_STEPS"))
        note = comments_to_markdown(gap)
        if note:
            cells.append(new_markdown_cell(f'**From the driver:**\n\n{note}'))
        cells += body_cells
        nb = new_notebook(cells=cells, metadata={
            'kernelspec': {'name': 'pyissm', 'display_name': 'pyissm', 'language': 'python'},
            'language_info': {'name': 'python'},
            'ais3_generated_from': driver})
        nbformat.validate(nb)
        nbformat.write(nb, outdir / nb_name)
        written.append((nb_rel, dedented, len(cells)))
    return written


if __name__ == '__main__':
    for drv in ('ais_0.1.py', 'ais_0.1_SSA.py'):
        for nb_rel, dedented, n in build(drv):
            print(f'{nb_rel:55s} {n:3d} cells' + ('' if dedented else '   (step kept as one cell)'))
