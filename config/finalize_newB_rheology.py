"""Save the post-geometry-fix shelf rheology inversion into the place the pipeline reads it from.

execution_newB_rheology/run_001_1_10_1e-17 is a single-point rerun of the
ssa_rheology_floating_inv_lcurve step (cf101=1, cf103=10, cf502=1e-17) made after the thickness
floor and effective-pressure fixes (2026-07-20). Today's step builds exactly that input (358
marshalled fields identical; worklog 2026-11-16), so a fresh run of the step gives the same result
as its 4th grid point, run_004_1_10_1e-17. This script loads the finished run the way the step's
load_only path does and saves it as

    models/AIS3_ssa_rheology_floating_inv_lcurve/run_004_1_10_1e-17/run_004_1_10_1e-17.nc

which every step now reads (rheology_lcurve_run; load_shelf_rheology_B in both drivers). The
June 2026 run_004 it replaces was moved, not deleted, to run_004_1_10_1e-17.jun30_before_geometry_fixes;
AIS3_inverted.nc and the whole HO track were built with that June field.

Refuses to overwrite an existing output.
"""
import ast
import copy
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, 'ais_0.1.py')
SOURCE_RUN = 'run_001_1_10_1e-17'
TARGET_RUN = 'run_004_1_10_1e-17'

# driver preamble (settings, cluster, helpers), then the step body up to its sweep call
src = open(DRIVER).read(); lines = src.splitlines(); tree = ast.parse(src)
first = next(n for n in tree.body if isinstance(n, ast.If) and 'steps' in ast.unparse(n.test))
g = {'__file__': DRIVER, '__name__': 'finalize_newB_rheology'}
exec(compile('\n'.join(lines[:first.lineno - 1]), DRIVER, 'exec'), g)
step = next(n for n in tree.body if isinstance(n, ast.If)
            and "'ssa_rheology_floating_inv_lcurve' in steps" in ast.unparse(n.test))
body = lines[step.lineno:step.end_lineno]
cut = next(i for i, l in enumerate(body) if l.strip() == 'if save:')
exec(compile(textwrap.dedent('\n'.join(body[:cut])), DRIVER, 'exec'), g)

import pyissm
site = g['site']
out_dir = f"{g['model_dir']}/AIS3_ssa_rheology_floating_inv_lcurve/{TARGET_RUN}"
out = f'{out_dir}/{TARGET_RUN}.nc'
if os.path.exists(out):
    sys.exit(f'{out} exists -- move it aside with mv -n first')

mdi = copy.deepcopy(g['mds'])
mdi = pyissm.inversion.sensitivity.assign_cost_functions(
    mdi, {101: 1, 103: 10, 502: 1e-17}, coeff_masks={101: g['mask'], 103: g['mask']})
mdi.cluster.executionpath = f'{site.ROOT}/execution_newB_rheology'
mdi.miscellaneous.name = SOURCE_RUN
mdi = pyissm.model.execute.solve(mdi, 'Stressbalance', runtime_name=False, load_only=True)
mdi.miscellaneous.name = TARGET_RUN

B = np.asarray(mdi.results.StressbalanceSolution.MaterialsRheologyBbar).ravel()
ev = np.asarray(mdi.mesh.extractedvertices).ravel()
print(f'{SOURCE_RUN}: {ev.size} shelf vertices, B {B.min():.4g}..{B.max():.4g}')
os.makedirs(out_dir, exist_ok=True)
pyissm.model.io.save_model(mdi, out)
print('saved', out)
