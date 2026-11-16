"""Shared pieces for running AIS3 SSA solves on the GPU ISSM fork (hypre BoomerAMG on an A100) and
for the matching 48-core MUMPS baselines. Used by solver_hypre_gpu_test.py and
solver_gpu_benchmark.py. Import after setting os.environ['ISSM_DIR'] (pyissm reads it on import).

Background in solver_hypre_gpu_test.py's docstring and docs: the GPU fork (site.env
AIS3_GPU_ISSM_DIR; ACCESS-NRI/ISSM branch justinh2002/GPU_cuda, whose Solverx.cpp returns to the
analysis' toolkit after a recovery solve) is built against the CUDA + hypre PETSc in
externalpackages/petsc/install-gpu-hypre.
"""
import os
import re
import numpy as np
import scipy.sparse as sp
import pyissm
import ais3_site as site

GPU_ISSM = site.GPU_ISSM_DIR
GPU_PETSC = f'{GPU_ISSM}/externalpackages/petsc/install-gpu-hypre'


def hypre_options(ksp_max_it='2000'):
    """hypre BoomerAMG + right-preconditioned GMRES, for md.toolkits.StressbalanceAnalysis only.
    Values are strings: pyissm's write_toolkits_file writes a falsy value (0, '') as a bare flag."""
    return {
        'toolkit': 'petsc',
        'vec_type': 'cuda',
        'mat_type': 'aijcusparse',
        'mat_block_size': '2',          # BoomerAMG systems AMG by node (NumFunctions 2); diverges without
        'ksp_type': 'gmres',
        'ksp_pc_side': 'right',         # rtol then bounds the true residual ISSM checks
        'ksp_rtol': '1e-8',
        'ksp_max_it': str(ksp_max_it),
        'ksp_gmres_restart': '100',
        'pc_type': 'hypre',
        'pc_hypre_type': 'boomeramg',
        'pc_hypre_boomeramg_strong_threshold': '0.25',
        'ksp_converged_reason': '',
    }


# MUMPS on the GPU objects: the CUSPARSE matrix already exists (mat_type is not used again), but ISSM
# creates the direct solve's solution vector from these options, so it must be a CUDA vector too.
RECOVERY_EXTRA = {'vec_type': 'cuda'}


class gadi_gpu(pyissm.model.classes.cluster.gadi):
    """pyissm's gadi cluster for a GPU queue: requests ngpus, runs np MPI ranks on ncpus cores, and
    sets the environment the GPU ISSM fork needs (its PETSc's RUNPATH still points at /g/data1b)."""

    def __init__(self, other=None):
        super().__init__(other=other)
        self.queue = 'dgxa100'
        self.ncpus = 16
        self.ngpus = 1
        self.modules = ['openmpi/4.1.3', 'netcdf/4.8.0p', 'hdf5/1.10.7p', 'cuda/12.3.2']
        self.ld_library_path = f'{GPU_ISSM}/lib:{GPU_PETSC}/lib'
        self.mps = False            # CUDA MPS daemon, so several ranks share the GPU concurrently
        # OpenMPI here is CUDA-aware, so PETSc passes device buffers to MPI by default; UCX's
        # GPUDirect copy (gdr_copy) then failed intermittently with ranks sharing a GPU under MPS
        # (SEGV at time step 3 of the first GPU transient, ~150 solves in). Stage through the host;
        # hypre is built without GPU-aware MPI and does so anyway. Read once at PetscInitialize, so
        # it must be in the environment: ISSM clears the options database between analyses.
        self.env = {'PETSC_OPTIONS': '-use_gpu_aware_mpi 0'}

    def check_consistency(self, md, solution, analyses):
        return md                   # the parent only knows CPU queues

    def build_queue_script(self, dir_name, model_name, solution, io_gather, is_valgrind, is_gprof,
                           is_dakota, is_ocean_coupling, executable='issm.exe', file_prefix=None):
        if file_prefix is None:
            file_prefix = model_name
        with open(file_prefix + '.queue', 'w') as fid:
            fid.write('#!/bin/bash\n')
            fid.write(f'#PBS -P {self.project}\n')
            fid.write(f'#PBS -q {self.queue}\n')
            fid.write(f'#PBS -l ncpus={self.ncpus}\n')
            fid.write(f'#PBS -l ngpus={self.ngpus}\n')
            fid.write(f'#PBS -l mem={self.memory}GB\n')
            fid.write(f'#PBS -l walltime={self.time * 60}\n')
            fid.write('#PBS -l wd\n')
            fid.write('#PBS -j oe\n')
            fid.write(f'#PBS -l storage={self.storage}\n')
            fid.write(f'#PBS -o {self.executionpath}/{dir_name}/{model_name}.outlog \n')
            fid.write(f'#PBS -e {self.executionpath}/{dir_name}/{model_name}.errlog \n\n')
            fid.write('module purge\n')
            for m in self.modules:
                fid.write(f'module load {m}\n')
            fid.write(f'export LD_LIBRARY_PATH={self.ld_library_path}:${{LD_LIBRARY_PATH}}\n')
            for k, v in self.env.items():
                fid.write(f'export {k}="{v}"\n')
            fid.write('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n')
            if self.mps:
                fid.write('export CUDA_MPS_PIPE_DIRECTORY=${PBS_JOBFS}/mps_pipe CUDA_MPS_LOG_DIRECTORY=${PBS_JOBFS}/mps_log\n')
                fid.write('mkdir -p ${CUDA_MPS_PIPE_DIRECTORY} ${CUDA_MPS_LOG_DIRECTORY}\n')
                fid.write('nvidia-cuda-mps-control -d && echo "MPS daemon started"\n')
            fid.write(f'mpiexec -n {self.np} {self.codepath}/{executable} {solution} '
                      f'{self.executionpath}/{dir_name} {model_name}\n')
            if self.mps:
                fid.write('echo quit | nvidia-cuda-mps-control\n')
            if not io_gather:
                fid.write(f'\ncat {model_name}.outbin.* > {model_name}.outbin\n')


def gpu_cluster(execution_dir, np_ranks=16, time_min=60):
    """One A100 GPU + 16 cores (1/8 of a DGX node), np_ranks ranks sharing it via MPS."""
    c = gadi_gpu()
    c.codepath = f'{GPU_ISSM}/bin'
    c.executionpath = execution_dir
    c.storage = site.STORAGE
    c.login = site.USER
    c.project = site.PROJECT
    c.np = np_ranks
    c.mps = np_ranks > 1
    c.memory = 250
    c.time = time_min
    return c


def cpu_cluster(execution_dir, time_min=60):
    """The SSA track's production config: production ISSM, 48 cores on normal (one node), MUMPS."""
    c = pyissm.model.classes.cluster.gadi()
    c.codepath = os.environ['ISSM_DIR'] + '/bin'
    c.executionpath = execution_dir
    c.storage = site.STORAGE
    c.moduleuse = ['/g/data/vk83/modules/']
    c.moduleload = ['access-issm_ad/2026.05.0']
    c.login = site.USER
    c.project = site.PROJECT
    c.np = 48
    c.memory = 190
    c.time = time_min
    return c


def use_gpu_toolkits(md, ksp_max_it='2000'):
    """hypre BoomerAMG for the stress balance; MUMPS (on CUDA vectors) for recovery and every other
    analysis. vec_type=cuda on the defaults too: once hypre has initialised CUDA, host vectors created
    afterwards have hit 'locked for read-only access' in VecSetValues (ISSM GPU_CHANGES.md #24)."""
    mumps = dict(md.toolkits.DefaultAnalysis)
    md.toolkits.StressbalanceAnalysis = hypre_options(ksp_max_it)
    md.toolkits.RecoveryAnalysis = {**mumps, **RECOVERY_EXTRA}
    md.toolkits.DefaultAnalysis = {**mumps, **RECOVERY_EXTRA}
    return md


def element_strain_rate(md, vx, vy):
    """SSA effective strain rate (1/s) per element from P1 velocities given in m/yr."""
    x = np.asarray(md.mesh.x, float).ravel()
    y = np.asarray(md.mesh.y, float).ravel()
    el = np.asarray(md.mesh.elements).astype(int) - 1
    u = np.asarray(vx, float).ravel() / md.constants.yts
    v = np.asarray(vy, float).ravel() / md.constants.yts
    x1, x2, x3 = x[el].T
    y1, y2, y3 = y[el].T
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    dNx = np.column_stack([y2 - y3, y3 - y1, y1 - y2]) / det[:, None]
    dNy = np.column_stack([x3 - x2, x1 - x3, x2 - x1]) / det[:, None]
    ux, uy = (dNx * u[el]).sum(1), (dNy * u[el]).sum(1)
    wx, wy = (dNx * v[el]).sum(1), (dNy * v[el]).sum(1)
    return np.sqrt(ux**2 + wy**2 + ux * wy + 0.25 * (uy + wx)**2)


def smooth_initial_velocity(md, passes=5):
    """Break the piecewise-constant gap fills in md.initialization (vel_obs): each pass replaces every
    vertex's velocity with the mean of itself and its mesh neighbours' mean. Exactly flat patches give
    zero strain rate, so Glen's viscosity and the first Picard matrix blow up (diagonal over 16.8
    orders of magnitude). Only the Picard starting point changes; the converged answer must not."""
    el = np.asarray(md.mesh.elements).astype(int) - 1
    nv = md.mesh.numberofvertices
    i = np.concatenate([el[:, a] for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))])
    j = np.concatenate([el[:, b] for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))])
    adj = sp.csr_matrix((np.ones(i.size), (i, j)), shape=(nv, nv))
    adj.data[:] = 1.0                                   # duplicate edges from shared elements
    deg = np.asarray(adj.sum(axis=1)).ravel()
    vx = np.asarray(md.initialization.vx, float).ravel().copy()
    vy = np.asarray(md.initialization.vy, float).ravel().copy()
    eps0 = element_strain_rate(md, vx, vy)
    for _ in range(passes):
        vx = 0.5 * vx + 0.5 * (adj @ vx) / deg
        vy = 0.5 * vy + 0.5 * (adj @ vy) / deg
    eps1 = element_strain_rate(md, vx, vy)
    md.initialization.vx = vx
    md.initialization.vy = vy
    md.initialization.vel = np.hypot(vx, vy)
    return {'passes': passes,
            'elements_eps_below_1e-15_before': int((eps0 < 1e-15).sum()),
            'elements_eps_below_1e-15_after': int((eps1 < 1e-15).sum()),
            'elements': int(eps0.size)}


def parse_outlog(path):
    if not os.path.exists(path):
        return {'outlog': 'missing'}
    txt = open(path, errors='ignore').read()

    def num(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    return {
        'core_s': num(r'Total Core solution elapsed time:\s*([\d.eE+-]+)'),
        'solver_s': num(r'Linear solver elapsed time:\s*([\d.eE+-]+)'),
        'init_s': num(r'FemModel initialization elapsed time:\s*([\d.eE+-]+)'),
        'su': num(r'Service Units:\s*([\d.]+)'),
        'walltime': (re.search(r'Walltime Used:\s*(\S+)', txt) or [None, None])[1],
        'ksp_its': [int(n) for n in re.findall(r'Petsc: (\d+) KSP iterations', txt)],
        'ksp_reasons': re.findall(r'Linear solve (?:converged|did not converge) due to (\w+)', txt),
        'recovery': len(re.findall(r'Trying Recovery Mode', txt)),
        'back_to_toolkit': len(re.findall(r'Recovery Mode succeeded, back to', txt)),
        'picard_its': num(r'total number of iterations:\s*(\d+)'),
        'picard_per_solve': [int(n) for n in re.findall(r'total number of iterations:\s*(\d+)', txt)],
        'exit_status': num(r'Exit Status:\s*(-?\d+)'),
    }
