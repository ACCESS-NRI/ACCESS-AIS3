"""Site settings for ACCESS-AIS3: paths, NCI project and Python environment.

    import ais3_site as site
    site.ROOT, site.CONFIG, site.USER, site.PROJECT, site.STORAGE, site.PYTHON,
    site.ISSM_DIR, site.ISMIP_DIR, site.GPU_ISSM_DIR, site.qsub_args()

Values come from config/site.env (the one file to edit), with environment variables of the
same name taking precedence -- the same rules the launchers apply when they source site.env.
The repository root is the folder above config/ unless AIS3_ROOT is set; the user is $USER
unless AIS3_USER is set.
"""
import os
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parent
_LINE = re.compile(r'^\s*([A-Z0-9_]+)=(.*?)\s*$')
_DEFAULT = re.compile(r'^\$\{([A-Z0-9_]+):-(.*)\}$')


def _read_site_env(path):
    values = {}
    for raw in Path(path).read_text().splitlines():
        if raw.lstrip().startswith('#'):
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        key, val = m.groups()
        val = val.strip('"\'')
        d = _DEFAULT.match(val)
        if d:  # KEY=${KEY:-default}
            val = d.group(2)
        values[key] = os.environ.get(key) or val
    return values


_env = _read_site_env(CONFIG / 'site.env')

ROOT = Path(os.environ.get('AIS3_ROOT') or CONFIG.parent).resolve()
CONFIG = ROOT / 'config'
USER = os.environ.get('AIS3_USER') or os.environ.get('USER', '')
PROJECT = _env['AIS3_PROJECT']
STORAGE = _env['AIS3_STORAGE']
PYTHON = _env['AIS3_PYTHON']
ISSM_DIR = _env['AIS3_ISSM_DIR']
ISMIP_DIR = _env['AIS3_ISMIP_DIR']
GPU_ISSM_DIR = _env.get('AIS3_GPU_ISSM_DIR', '')   # GPU fork; only the solver_*gpu* tests need it


def qsub_args():
    """Project and storage flags for a qsub issued from Python (override the #PBS lines)."""
    return ['-P', PROJECT, '-l', f'storage={STORAGE}']


if __name__ == '__main__':
    for k in ('ROOT', 'CONFIG', 'USER', 'PROJECT', 'STORAGE', 'PYTHON', 'ISSM_DIR', 'ISMIP_DIR',
              'GPU_ISSM_DIR'):
        print(f'{k:12s} {globals()[k]}')
