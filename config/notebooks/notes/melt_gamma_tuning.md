### What this step does
Builds `AIS3_melt_gamma_tuning.nc`, the input to `ho_relaxation`, from the validated ocean-melt
calibration. There is no cluster solve. The step:
- loads `AIS3_ho_friction_inv.nc`
- restores real thickness where the 100 m floor from `ssa_inverted_solve` is still in the
  geometry (looked up in `AIS3_param.nc`), then syncs `mesh.z`
- sets ISMIP6 quadratic-local melt with IMBIE2 basins, Zhou thermal forcing, `gamma_0=300`
  and the 16 per-basin refit `deltaT` values hard-coded in the step

The calibration search that produced these values is in standalone scripts (see the run guide,
§5.3): `melt_ismip7_calibration.py` → `melt_deltaT_sensitivity_test.py` →
`melt_deltaT_basin_refit.py` → `melt_deltaT_refit_recalibrate.py`.

### Before you run
- The step refuses to overwrite an existing output. Archive it (`mv -n`) or set
  `AIS3_MELT_TAG` to write a tagged copy.
- The outer driver needs hugemem memory to load and save the HO model.

### What to check afterwards
- "N vertices restored": about 2.97 M vertices (12.5%), mean about 19 m.
- 30 thermal-forcing depth layers.
- Not yet checked: a rebuild compared field by field against the current production file.

### Known pitfalls
- `md.friction.C` is kept as loaded. That is chunk 2's field, one chunk behind the inversion
  output. Test C in worklog §5c showed this makes no difference to speed, so it is
  bookkeeping only.
- The old version of this step ran a per-vertex RMSE sweep. That metric is gamed by
  switching melt off almost everywhere (worklog §4b), so don't reintroduce it.
