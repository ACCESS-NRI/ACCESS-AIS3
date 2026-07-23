#!/usr/bin/env python
"""
Quick field maps for a friction inversion run.

Usage:
    python plot_friction_run.py [run_name] [sensit|lcurve]

Loads the saved run model and dumps PNGs of the inverted friction C, modelled
velocity, observed velocity, velocity residual, and the Coulomb-failure mask into
the run directory. Handy for eyeballing whether the bulk is well-fit and whether
the masked (unfittable) nodes line up with the fast/steep regions.

Examples:
    python plot_friction_run.py run_013_10_10 sensit
    python plot_friction_run.py run_004_1_1_1e-17 lcurve
"""
import os
import sys

os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyissm

model_dir = '/g/data/au88/jh7060/ACCESS-AIS3/models'
run_name = sys.argv[1] if len(sys.argv) > 1 else 'run_013_10_10'
which = sys.argv[2] if len(sys.argv) > 2 else 'sensit'
folder = {'sensit': 'AIS3_ssa_friction_inv_sensit',
          'lcurve': 'AIS3_ssa_friction_inv_lcurve'}[which]

run_dir = f'{model_dir}/{folder}/{run_name}'
print(f'Loading {run_dir}/{run_name}.nc ...')
md = pyissm.model.io.load_model(f'{run_dir}/{run_name}.nc')

vel  = np.asarray(md.results.StressbalanceSolution.Vel).ravel()
vobs = np.asarray(md.inversion.vel_obs).ravel()
C    = np.asarray(md.results.StressbalanceSolution.FrictionC).ravel()
res  = vel - vobs


def save(field, fname, title, **kw):
    fig, ax = pyissm.plot.plot_model_field(md, field, show_cbar = True, **kw)
    ax.set_title(f'{run_name}: {title}')
    out = f'{run_dir}/{run_name}_{fname}.png'
    plt.savefig(out, dpi = 120, bbox_inches = 'tight')
    plt.close(fig)
    print(f'  wrote {out}')


print(f'Plotting {run_name} ({folder})...')
# Friction C and velocities span orders of magnitude -> log scale (floor for log safety).
save(np.maximum(C, 0.1),    'FrictionC', 'inverted friction C', log = True,
     cmap = 'viridis', cbar_kwargs = {'label': 'C'})
save(np.maximum(vel, 1.0),  'vel_mod',   'modelled velocity',   log = True,
     cmap = 'PuOr', vmin = 1, vmax = 4000, cbar_kwargs = {'label': 'm/yr'})
save(np.maximum(vobs, 1.0), 'vel_obs',   'observed velocity',   log = True,
     cmap = 'PuOr', vmin = 1, vmax = 4000, cbar_kwargs = {'label': 'm/yr'})
save(res, 'residual', 'modelled - observed', cmap = 'RdBu',
     vmin = -100, vmax = 100, cbar_kwargs = {'label': 'm/yr'})

# Coulomb-failure mask (same predicate as the inversion cost mask): driving stress > Cmax*N.
_, _, sslope = pyissm.tools.geometry.slope(md)
H_e    = np.mean(md.geometry.thickness[md.mesh.elements - 1], axis = 1)
N_e    = np.mean(md.friction.effective_pressure[md.mesh.elements - 1], axis = 1)
Cmax_e = np.mean(md.friction.Cmax[md.mesh.elements - 1], axis = 1)
fail_e = (md.materials.rho_ice * md.constants.g * H_e * sslope) > (Cmax_e * N_e)
fail   = np.zeros(md.mesh.numberofvertices)
fail[md.mesh.elements[fail_e, :] - 1] = 1.0
save(fail, 'coulomb_mask', f'Coulomb-failure nodes ({int(fail.sum())} excluded)',
     cmap = 'Reds', cbar_kwargs = {'label': 'excluded (1=masked)'})

# Print the masked bulk fit for reference.
m = (vobs > 0) & (vel < 1e4)
rmse_bulk = np.sqrt(np.nanmean(res[m] ** 2))
print(f'\nbulk vel_rmse (obs>0, vel<1e4, n={m.sum()}): {rmse_bulk:.2f} m/yr')
print('done.')
