import pyissm
import ccdtools
import numpy as np
import xarray as xr

"""
TODO:
- Add Shivani's effective_pressure field to CCD (https://github.com/ACCESS-NRI/ccdtools/issues/62)
- Add MIPKIT to CCD (https://github.com/ACCESS-NRI/ccdtools/issues/61) and replace individual datasets with this, where possible
"""

print("---------------------------------------------------------"  )
print(f" RUNNING ais_01_param.py PARAMETERIZATION SCRIPT"          )
print("---------------------------------------------------------\n")

# Initialize data catalog & load required datasets
print(f"\nLOADING REQUIRED DATASETS")
catalog = ccdtools.catalog.DataCatalog()
bedmachine_data = catalog.load_dataset('measures_bedmachine_antarctica', version = 'v3')
velocity_data = catalog.load_dataset('measures_insar_based_antarctica_ice_velocity_map', version = 'v2')
geothermal_data = catalog.load_dataset('antarctic_geothermal_heat_flow_model_aq1', resolution = '20km')
racmo_data = catalog.load_dataset('racmo2.4p1_monthly_11km_1979-2023')

ehrenfeucht_2024 = xr.open_dataset('/g/data/au88/lb9857/access-ais3/data/Antarctica_SubglacialHydrology.nc')

## -----------------------------------------
## --------- STRESS BALANCE FIELDS ---------
## -----------------------------------------

## --------- ICE MASK --------- 
print(f"\nDEFINING MODEL MASK")
    
# Initialise levelset fields to all ice and no ocean
md.mask.ice_levelset = -1 * np.ones(md.mesh.numberofvertices)
md.mask.ocean_levelset = np.ones(md.mesh.numberofvertices)

# Interpolate BedMachine v3 Ice Mask
mask = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'mask', md.mesh.x, md.mesh.y, interpolation_type = 'nearest')

# Set ocean areas (mask == 0) to positive levelset (no ice)
no_ice_mask = (mask < 1)
md.mask.ice_levelset[no_ice_mask] = 1

# Set floating ice (mask == 3) and ocean (mask == 0) to negative levelset (ocean)
ocean_mask = (mask == 3) | (mask == 0)
md.mask.ocean_levelset[ocean_mask] = -1

## --------- GEOMETRY --------- 
print(f"\nDEFINING MODEL GEOMETRY")

# Interpolate bed, surface, and thickness onto mesh; Calculate base.
md.geometry.bed = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'bed', md.mesh.x, md.mesh.y, interpolation_type = 'bilinear')
md.geometry.surface = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'surface', md.mesh.x, md.mesh.y, interpolation_type = 'bilinear')
md.geometry.thickness = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'thickness', md.mesh.x, md.mesh.y, interpolation_type = 'bilinear')
md.geometry.base = md.geometry.surface - md.geometry.thickness

# Enforce minimum surface of 1 m
md.geometry.surface = np.maximum(md.geometry.surface, 1)
md.geometry.thickness = md.geometry.surface - md.geometry.base

# Enforce minimum thickness of 10 m and adjust base accordingly
md.geometry.thickness = np.maximum(md.geometry.thickness, 10)
md.geometry.base = md.geometry.surface - md.geometry.thickness

# Ensure bed is below base
pos = md.geometry.base < md.geometry.bed
md.geometry.bed[pos] = md.geometry.base[pos]

# Set ice shelf base using hydrostatic equilibrium
floating_base = md.materials.rho_ice / (md.materials.rho_ice - md.materials.rho_water) * md.geometry.surface
pos = floating_base > md.geometry.bed
md.geometry.base[pos] = floating_base[pos]
di = md.materials.rho_ice / md.materials.rho_water
md.geometry.thickness = md.geometry.surface - md.geometry.base
md.mask.ocean_levelset = md.geometry.thickness + md.geometry.bed / di

# Ensure bed is below base and bed/base are consistent on grounded ice
pos = (md.geometry.base < md.geometry.bed) | (md.mask.ocean_levelset > 0)
md.geometry.bed[pos] = md.geometry.base[pos]

# Offset the mask by one element so that there are no cliffs at the transition
elems = md.mesh.elements - 1
vals = md.mask.ice_levelset[elems]
pos_e = np.max(vals, axis = 1) > 0
md.mask.ice_levelset[elems[pos_e, :]] = 1
pos = (md.mask.ice_levelset < 0) & (md.geometry.surface < 0)
md.mask.ice_levelset[pos] = 1
pos = (md.mask.ice_levelset > 0) & np.isnan(md.geometry.surface)
md.mask.ice_levelset[pos] = 1

## --------- VELOCITY --------- 
print(f"\nDEFINING MODEL INITIAL & OBS VELOCITY")

# Interpolate velocity data
vx = pyissm.data.interp.xr_to_mesh(velocity_data, 'VX', md.mesh.x, md.mesh.y)
vy = pyissm.data.interp.xr_to_mesh(velocity_data, 'VY', md.mesh.x, md.mesh.y)

# Fill NaN values in vx/vy
for arr in [vx, vy]:

    ## Identify nan points
    nan_mask = np.isnan(arr)

    ## Fill nan values using NN interpolation
    if np.any(nan_mask):
        filled = pyissm.data.interp.points_to_mesh(
            data_x=md.mesh.x[~nan_mask],
            data_y=md.mesh.y[~nan_mask],
            data_values=arr[~nan_mask],
            mesh_x=md.mesh.x,
            mesh_y=md.mesh.y,
            interpolation_type='nearest'
        )

        ## Replace nan values
        arr[nan_mask] = filled[nan_mask]

    # Replace any remaining NaNs with zero
    arr = np.nan_to_num(arr, nan = 0.0)

# Calculate vel
vel = np.sqrt(vx**2 + vy**2)

# Identify nodes belonging to elements containing ANY ice (e.g. including ice-front)
## Average ice-mask to elements to identify elements with ANY ice (e.g. all ice or ice-front)
tmp = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset)
mds = md.extract(tmp < 1)
keep = mds.mesh.extractedvertices - 1 # NOTE -1 for 0-based indexing

# Create observation arrays
## This sets PURELY ocean elements to 0 and assigns observed velocities to all others
md.inversion.vx_obs = np.zeros(md.mesh.numberofvertices)
md.inversion.vy_obs = np.zeros(md.mesh.numberofvertices)
md.inversion.vel_obs = np.zeros(md.mesh.numberofvertices)
md.inversion.vx_obs[keep] = vx[keep]
md.inversion.vy_obs[keep] = vy[keep]
md.inversion.vel_obs[keep] = vel[keep]

# Assign observed velocities to initial velocities
md.initialization.vx = md.inversion.vx_obs.copy()
md.initialization.vy = md.inversion.vy_obs.copy()
md.initialization.vz = np.zeros(md.mesh.numberofvertices)
md.initialization.vel = md.inversion.vel_obs.copy()

## --------- FLOW LAW --------- 
print(f"\nDEFINING FLOW LAW")

md.materials.rheology_n = 3.0 * np.ones(md.mesh.numberofelements)
md.materials.rheology_law = 'Cuffey'
# NOTE: rheology_B is deliberately NOT set here. It is derived from the RACMO surface temperature
# field further below (see "SURFACE TEMPERATURE"), which does not exist yet at this point.
#
# It was previously hardcoded to a uniform cuffey(273.15 - 20) = 2.027e8 placeholder ("assume -20C
# everywhere for now"). That collapsed a ~17x real spatial viscosity range (cuffey(T) spans
# 8.5e7 -> 1.43e9 over the RACMO range -62C..-2C) onto a single value, leaving 51% of grounded
# vertices wrong by >20% (up to 7x too stiff in the cold interior, 2.4x too soft at the margins).
# This was the dominant source of the large, structured (non-white) velocity residuals in the ice
# streams and on the big ice shelves -- an error that NO friction inversion can correct, because
# friction acts on the bed while the error lives in the ice viscosity.

## --------- PRESSURE --------- 
print(f"\nDEFINING INITIAL PRESSURE")

# Assume overburden pressue
md.initialization.pressure = md.materials.rho_ice * md.constants.g * md.geometry.thickness

## --------- FRICTION --------- 
print(f"\nDEFINING INITIAL FRICTION")

# Friction law switch. 'schoof' = regularized Coulomb (drag capped at Cmax*N, more physical for
# fast marine sliding but the cap leaves ~0.34% of steep/low-N cells unfittable). 'budd' = Weertman
# power law (no ceiling; a forward check showed it clears the Coulomb residual ~30x and converges
# robustly at every coefficient, at the cost of the Iken bound). The main script (ais_0.1.py)
# auto-detects the law from the saved friction class, so only this flag needs changing.
friction_law = 'budd'  # 'schoof' or 'budd'

if friction_law == 'schoof':
    md.friction = pyissm.model.classes.friction.schoof()
    # Initial guess for the Schoof friction coefficient. A forward-solve C sweep showed C=500 is
    # ~10x too weak (grounded ice runs away to ~4e5 m/yr), while C~5000 brings the grounded
    # velocity field into the physical range (median ~4 m/yr). Start the inversion here.
    md.friction.C = np.full(md.mesh.numberofvertices, 5000)
    md.friction.Cmax = np.full(md.mesh.numberofvertices, 0.85)
    md.friction.m = np.full(md.mesh.numberofelements, 1/3)

elif friction_law == 'budd':
    md.friction = pyissm.model.classes.friction.default()  # Budd / Weertman power law
    # tau_b = coefficient^2 * Neff^r * |u|^(s-1) * u, with s = 1/p, r = q/p. p=3,q=3 gives
    # tau_b ~ N*|u|^(1/3) (Budd, linear N-dependence, m=3). A forward-coefficient sweep showed
    # ~10-100 gives physical velocities (uniform values over-brake the interior; the inversion
    # assigns coefficient spatially). Start the inversion around 10.
    md.friction.coefficient = np.full(md.mesh.numberofvertices, 10.0)
    md.friction.p = np.full(md.mesh.numberofelements, 3.0)
    md.friction.q = np.full(md.mesh.numberofelements, 3.0)

else:
    raise ValueError(f"Unknown friction_law '{friction_law}' (expected 'schoof' or 'budd')")

# Effective-pressure (Neff) source for the friction law.
#   3 = use the provided md.friction.effective_pressure (Ehrenfeucht et al. 2024, floored below).
#       CURRENT / VALIDATED: Budd + coupling=3 gives clean 25/25 convergence and bulk_rmse
#       ~376-491 m/yr with a Dirichlet-anchored ice-front. Do not change without re-validating.
#   2 = ISSM's internal 'uniform sheet' hydrology proxy, clamped >= 0 by the core. FOLLOW-UP
#       EXPERIMENT (not yet run): sidesteps the Ehrenfeucht dataset's NaNs/negatives entirely
#       instead of patching them with our manual floor -- worth comparing against the coupling=3
#       result once the current inversion pipeline is done. Needs its own forward-check +
#       sensit validation cycle before trusting it (same process used to validate Budd itself).
#   DO NOT use 0 (the friction.default() class default): it explicitly allows negative Neff.
#   Our Budd exponents give r = q/p = 1 (linear in N), so a negative N flips basal drag from
#   resistive to driving -- the same destabilizing failure mode the N floor below was built to
#   fix, just reintroduced from a different source.
friction_coupling = 3  # 3 (Ehrenfeucht, current) or 2 (ISSM internal hydrology, follow-up)
md.friction.coupling = friction_coupling

if friction_coupling == 3:
    # Interpolate effective pressure from Ehrenfeucht et al. (2024)
    md.friction.effective_pressure = pyissm.data.interp.xr_to_mesh(ehrenfeucht_2024, 'effective_pressure', md.mesh.x, md.mesh.y)

    # The Ehrenfeucht (2024) field contains NaNs (data gaps) and some non-physical negative
    # values. In the Schoof law basal drag is capped at Cmax * N, so any node with N <= 0
    # supplies ZERO drag and the forward SSA solve runs away (velocities -> 1e7 m/yr). Enforce
    # a physical positive floor N >= N_floor_frac * (rho_i * g * H), sized so the Coulomb cap
    # Cmax * N stays above the local driving stress (~surface_slope * rho_i * g * H).
    # A forward check with a 0.03 floor left ~0.64% of nodes (95% of them sitting at the floor)
    # running away: their ceiling Cmax*N = 0.85*0.03 = 0.026 of overburden fell below the local
    # driving stress. These are slow-observed interior cells where a higher N is more physical
    # anyway (thick cold-based ice has N ~ 10-50% of overburden), so raise the floor to 0.07
    # (ceiling 0.85*0.07 = 0.06 of overburden) to cover the steep-slope driving stresses.
    N_floor_frac = 0.07  # minimum effective pressure as a fraction of ice overburden
    N_floor = N_floor_frac * md.materials.rho_ice * md.constants.g * md.geometry.thickness

    N = md.friction.effective_pressure
    N = np.nan_to_num(N, nan = 0.0)   # replace data gaps before applying the floor
    N = np.maximum(N, N_floor)        # positive floor: removes negatives and zero-drag holes
    md.friction.effective_pressure = N
else:
    # coupling=2: ISSM computes Neff internally (uniform sheet, clamped >= 0); no external
    # effective_pressure array needed. Still apply the same floor fraction as a safety net.
    N_floor_frac = 0.07

# Register the floor with the ISSM core so it is enforced during the solve (applies regardless
# of coupling mode -- effective_pressure_limit is checked unconditionally in check_consistency).
md.friction.effective_pressure_limit = N_floor_frac

## -----------------------------------------
## --------- THERMAL BALANCE FIELDS --------
## -----------------------------------------

## --------- GEOTHERMAL HEAT FLOW --------- 
print(f"\nDEFINING GEOTHERMAL HEAT FLOW")

# Fill missing values with 0
geothermal_data['Q'] = geothermal_data['Q'].fillna(0.0)

# Interpolate AQ1 data
md.basalforcings.geothermalflux = pyissm.data.interp.xr_to_mesh(geothermal_data, 'Q', md.mesh.x, md.mesh.y, x_var = 'X', y_var = 'Y')

# Set NaN values to 0
md.basalforcings.geothermalflux = np.nan_to_num(md.basalforcings.geothermalflux, nan = 0.0)

## --------- SURFACE TEMPERTURE --------- 
print(f"\nDEFINING INITIAL TEMPERATURE (Surface Temp)")

# Extract surface temperature from racmo
ts = racmo_data['ts']

# Convert lat/lon to xy
[x, y] = pyissm.tools.general.ll_to_xy(racmo_data['lat'].values, racmo_data['lon'].values, -1)

# Extract 1995 values and calculate annual mean
ts_1995 = ts.sel(time = ts['time.year'] == 1995)
ts_1995_mean = ts_1995.mean('time').squeeze()

# Interpolate onto mesh
ts_mesh = pyissm.data.interp.points_to_mesh(x, y, ts_1995_mean.to_numpy(), md.mesh.x, md.mesh.y)

# Assign to model and cap maximum temperature at 0 degC
md.initialization.temperature = np.minimum(ts_mesh, 273.15)

## --------- FLOW LAW: TEMPERATURE-DEPENDENT RHEOLOGY B ---------
print(f"\nDEFINING RHEOLOGY B FROM TEMPERATURE (Cuffey)")

# Derive rheology_B from the temperature field via the Cuffey relation. This replaces the former
# uniform cuffey(-20C) placeholder (see the FLOW LAW block above for why that was wrong).
md.materials.rheology_B = pyissm.tools.materials.cuffey(md.initialization.temperature)

# CAVEAT: this uses the SURFACE temperature as a proxy for the depth-averaged temperature that SSA
# actually wants. Real ice warms with depth (geothermal flux + strain heating), so this biases B
# somewhat too stiff, especially where the ice is thick. It nonetheless captures the dominant
# spatial pattern (cold stiff interior vs warm soft margins/shelves) and is vastly better than a
# single constant. TODO: revisit with a depth-averaged or thermal-model temperature if the
# remaining residual warrants it.
print(f"  rheology_B: min = {np.min(md.materials.rheology_B):.4g}, "
      f"median = {np.median(md.materials.rheology_B):.4g}, "
      f"max = {np.max(md.materials.rheology_B):.4g}")