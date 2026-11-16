"""Salvage AIS3_ho_relaxed.nc from the raw execution outbin -- the actual relaxation compute
job (178026389) completed cleanly (exit 0, wrote its lock file, 6h49m), and the outer driver
successfully printed real diagnostics (max|dH|=982.93m, mean|dH|=13.00m) confirming the
results were genuinely retrieved -- but the driver's own save_model() call was then SIGKILL'd
by OOM (177.52GB/190GB used) before the file finished writing, leaving a truncated/corrupted
.nc with no `results` group. Rather than re-running the ~7h relaxation, rebuild the exact
shell config ho_relaxation itself used and pull the already-computed results directly from
the raw execution/AIS3_ho_relaxed/AIS3_ho_relaxed.outbin (same salvage-from-raw-run technique
validated earlier this session for the p1q1 Budd result, finalize_p1q1_reg_lcurve.py).

Run via hugemem (not the normal-queue 190GB that OOM'd) to give the save step real headroom.
"""
import os
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3')
import numpy as np
import pyissm

MODEL_DIR = '/g/data/au88/jh7060/ACCESS-AIS3/models'
OUTBIN = '/g/data/au88/jh7060/ACCESS-AIS3/execution/AIS3_ho_relaxed/AIS3_ho_relaxed.outbin'

print('-- Loading melt-calibrated model (shell)...', flush=True)
md = pyissm.model.io.load_model(f'{MODEL_DIR}/AIS3_melt_gamma_tuning.nc')
md.basalforcings.tf = [np.asarray(t, dtype=float) for t in md.basalforcings.tf]

md.inversion.iscontrol = 0
md.verbose.solution = 1
md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
md.transient.isstressbalance = 1
md.transient.ismasstransport = 1
md.transient.issmb = 1
md.transient.isthermal = 0
md.transient.isgroundingline = 1
md.groundingline.migration = 'SubelementMigration'
md.transient.requested_outputs = ['default', 'Vel', 'Thickness', 'Surface', 'Base', 'MaskOceanLevelset']
md.timestepping.start_time = 0
md.timestepping.final_time = 1
md.timestepping.time_step = 0.02
md.miscellaneous.name = 'AIS3_ho_relaxed'

print(f'-- Loading results directly from raw outbin ({OUTBIN})...', flush=True)
md = pyissm.model.execute.load_results_from_disk(md, OUTBIN)

dH = np.asarray(md.results.TransientSolution[-1].Thickness).ravel() - np.asarray(md.geometry.thickness).ravel()
print(f'\nRELAXATION DIAGNOSTICS (verification -- expect max~982.93m, mean~13.00m):')
print(f'   Max |dH| over relaxation: {np.nanmax(np.abs(dH)):.2f} m')
print(f'   Mean |dH| over relaxation: {np.nanmean(np.abs(dH)):.2f} m')

out_path = f'{MODEL_DIR}/AIS3_ho_relaxed.nc'
print(f'\nSaving to {out_path}', flush=True)
pyissm.model.io.save_model(md, out_path)
print('DONE')
