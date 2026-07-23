import pyissm
import ccdtools as ccdtools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import os


def friction_law_info(md):
    """Return (control_parameter, field_attr, min_bound, max_bound) for md's friction law.

    Schoof (regularized Coulomb) inverts 'FrictionC' (field md.friction.C); Budd/Weertman (the
    'default' class) inverts 'FrictionCoefficient' (field md.friction.coefficient). The saved
    friction class (set by friction_law in ais_0.1_param.py) is the single source of truth.
    """
    if type(md.friction).__name__ == 'default':   # Budd / Weertman power law
        return 'FrictionCoefficient', 'coefficient', 0.01, 1e4
    return 'FrictionC', 'C', 0.05, 250 ** 2        # Schoof (regularized Coulomb)


def extract_friction_inversion_domain(md):
    """Extract the friction-inversion subdomain with a floating/grounded ice-front boundary condition.

    Extracts ALL ice (ice_levelset_elements < 1, includes the ice-front elements), so the new
    mesh boundary coincides exactly with the true, contiguous ice margin -- not an arbitrary
    internal cut. extract() imposes Dirichlet (observed velocity) on every new boundary node by
    default (see Model.py: "Boundary conditions: Dirichlets on new boundary"); this is reverted
    to Neumann (NaN spc) at boundary nodes classified as floating (ocean_levelset < 0), i.e. true
    ice-shelf calving fronts, where the natural ocean-pressure BC is physically correct. Boundary
    nodes classified as grounded (ocean_levelset >= 0) -- both marine-terminating (bed below sea
    level, no shelf) and true land-terminating (bed above sea level, no ocean to push back
    against) -- keep extract()'s default Dirichlet, since Neumann has no obvious physical meaning
    there. Classification is per-vertex (mds.mask.ocean_levelset), not per-element, so it follows
    the true ice-front geometry exactly with no fragmentation.

    A prior version anchored only the Ronne-Filchner/Ross fronts (the two largest floating
    regions, found to blow up under pure Neumann at low friction coefficient) and left everything
    else -- including land-terminating margins -- as Neumann. That fixed Ronne-Filchner/Ross but
    left land-terminating margins with a physically meaningless Neumann BC, which was the actual
    cause of a ~1e10 m/yr blowup at coeff=1 (confirmed: switching those margins to Dirichlet here
    brought coeff=1 down to ~1.8e7 m/yr).
    """
    ice_levelset_elements = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset)
    mds = md.extract(ice_levelset_elements < 1)

    bnd = mds.mesh.vertexonboundary.astype(bool)
    ocean_ls = np.asarray(mds.mask.ocean_levelset).ravel()
    floating_bnd = bnd & (ocean_ls < 0)

    mds.stressbalance.spcvx[floating_bnd] = np.nan
    mds.stressbalance.spcvy[floating_bnd] = np.nan
    mds.stressbalance.spcvz[floating_bnd] = np.nan
    mds.mask.ice_levelset[floating_bnd] = 0

    return mds

## ------------------------------------
## Configure options
## ------------------------------------

# Change directory to gdata to prevent storage limits in $HOME
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3/')
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'

# Should plots be generated?
plot = True
diagnostics = True
save = False
inversion_sensitivity = False

# Define execution directory
execution_dir = '/g/data/au88/jh7060/ACCESS-AIS3/execution'

# Define location to save final models
model_dir = '/g/data/au88/jh7060/ACCESS-AIS3/models'

# Define domain_file
domain_file = ('/g/data1b/au88/jh7060/ACCESS-AIS3/assets/ais_domain.exp')

# Define param_file
param_file = ('/g/data1b/au88/jh7060/ACCESS-AIS3/config/ais_0.1_param.py')

# Define cluster requirements
cluster = pyissm.model.classes.cluster.gadi()
cluster.codepath = os.environ['ISSM_DIR']+'/bin'
cluster.executionpath = execution_dir
cluster.storage = 'gdata/au88+gdata/vk83'
cluster.moduleuse = ['/g/data/vk83/modules/']
cluster.moduleload = ['access-issm/2025.11.0']
cluster.np = 32
cluster.memory = 100
cluster.time = 60*48
cluster.login = 'jh7060'
cluster.project = 'au88'

# List all steps for clarity
all_steps = [
    'process_domain',
    'mesh',
    'param',
    'ssa_rheology_floating_inv_sensit',
    'ssa_rheology_floating_inv_lcurve',
    # 'ssa_rheology_floating_inv',
    'ssa_friction_forward_check',
    'ssa_friction_forward_check_budd',
    'ssa_friction_inv_sensit',
    'ssa_friction_inv_lcurve',
    'ssa_inverted_solve',
    'ssa_relaxation'
]

# Define steps to run
# steps = ['process_domain', 'mesh', 'param']
# steps = ['mesh', 'param']
# steps = ['ssa_rheology_floating_inv_sensit']
# steps = ['ssa_rheology_floating_inv_lcurve']
# steps = ['ssa_friction_forward_check']
# steps = ['ssa_friction_forward_check_budd']
# steps = ['param']
# steps = ['ssa_friction_inv_lcurve']
# steps = ['ssa_inverted_solve']
# steps = ['ssa_relaxation']
# steps = ['param']
steps = ['ssa_friction_inv_lcurve']
# steps = ['ssa_friction_inv_sensit']

## ------------------------------------
## Chosen inversion runs (update after inspecting sensit / lcurve diagnostics)
## ------------------------------------
# Floating-ice rheology B field taken from the rheology L-curve (cf502 regularisation).
rheology_lcurve_run = 'run_004_1_10_1e-17'

# Preferred 101/103 cost-function coefficients for the friction inversion, chosen from the
# floating/grounded-BC + coupling=3 sensit sweep (best cell by the library's combined
# fit+smoothness score: run_021, cf101=1000/cf103=0.1, vel_rmse=960.5 -- the whole 25-cell grid
# was smooth and outlier-free under coupling=3, unlike coupling=2 -- see note below).
friction_cf101 = 1000
friction_cf103 = 0.1

# Effective-pressure source for the friction law. coupling=2 (ISSM internal "uniform sheet"
# hydrology, clamped >= 0) matched or beat coupling=3 (Ehrenfeucht dataset + manual N floor) in
# the earlier *uniform-coefficient forward-check* sweep -- but the full floating/grounded-BC
# inversion sensit sweep told a different story: coupling=2 has a specific, severe pathology at
# certain coefficient cells (cf101=cf103=10 and cf101=cf103=1000 both spiked to vel_rmse~13,100,
# ~13x every other cell) that the simpler forward-check never happened to probe. coupling=3 was
# clean and outlier-free across the entire 25-cell grid (vel_rmse 962-1385, no spikes). Reverted
# to coupling=3 on the strength of that full-grid evidence. AIS3_param.nc itself is also built
# with friction_coupling=3 (see ais_0.1_param.py), so this now matches the param-file default.
friction_coupling = 3

# Grounded-ice friction C field taken from the friction L-curve (cf501 regularisation -- the
# DragCoefficientAbsGradient term; see bug-fix note in the ssa_friction_inv_lcurve block).
# Built below once the friction L-curve has run -- UPDATE the cf501 value to the chosen corner.
friction_lcurve_run = f'run_004_{friction_cf101}_{friction_cf103}_1e-17'


## ------------------------------------
## Initialise Data Catalog
## ------------------------------------
catalog = ccdtools.catalog.DataCatalog()
bedmachine_data = catalog.load_dataset('measures_bedmachine_antarctica', version = 'v3')
velocity_data = catalog.load_dataset('measures_insar_based_antarctica_ice_velocity_map', version = 'v2')
measures_coastline = catalog.load_dataset('measures_antarctic_boundaries', subdataset = 'coastline')


## ------------------------------------
## Process domain file
## ------------------------------------
if 'process_domain' in steps:

    print("-------------------------------------------------------------")
    print(f" PROCESSING DOMAIN FILE"                                     )
    print("-------------------------------------------------------------")

    # Buffer coastline polygon by 100 km
    print(f" - Buffering coastline...")
    coastline_100km_buffer = measures_coastline.buffer(100000)

    # Write buffered extent to file for use as model domain
    print(f" - Saving to file...")
    pyissm.tools.exp.gdf_to_exp(coastline_100km_buffer, '/g/data1b/au88/jh7060/ACCESS-AIS3/assets/ais_domain.exp')


## ------------------------------------
## Create mesh
## ------------------------------------
if 'mesh' in steps:

    print("-------------------------------------------------------------")
    print(f" GENERATING MESH"                                            )
    print("-------------------------------------------------------------")

    # Create empty model with initial 10e3 resolution mesh
    md = pyissm.model.mesh.triangle(pyissm.model.Model(), domain_file, 10e3)

    # Remesh the model twice to refine based on velocity and bedmachine mask
    for i in range(2):

        print(f"REFINEMENT ITERATION: {i+1}")

        # Interpolate velocities onto mesh
        print(f"\n-- Interpolating MEaSURES v2 Velocities...")
        vx = pyissm.data.interp.xr_to_mesh(velocity_data, 'VX', md.mesh.x, md.mesh.y)
        vy = pyissm.data.interp.xr_to_mesh(velocity_data, 'VY', md.mesh.x, md.mesh.y)
        vel = np.sqrt(vx**2 + vy**2)

        # Interpolate ice mask onto mesh
        print(f"\n-- Interpolating Bedmachine v3 Ice Mask...")
        mask = pyissm.data.interp.xr_to_mesh(bedmachine_data, 'mask', md.mesh.x, md.mesh.y, interpolation_type = 'nearest')

        # Fill NaN values and set to 0 ice-free areas (and ocean, but over-ridden below)
        print(f"\n-- Set Velocity to 0 where NaNs exist or mask < 2...")
        vel[np.isnan(vel) | (mask < 2)] = 0.0

        print(f"\n-- Set Velocity to NaN in ocean areas...")
        vel[(mask < 2)] = np.nan

        if diagnostics:
            print(f"\nVELOCITY DIAGNOSTICS:")
            print(f"    Max velocity: {np.nanmax(vel):.2f} m/yr")
            print(f"    Min velocity: {np.nanmin(vel):.2f} m/yr")

        if plot:
            pyissm.plot.plot_model_field(md, vel, cmap = 'PuOr',show_cbar = True, cbar_kwargs = {'label': 'Velocity (m/a)'}); plt.show(block = False)

        if diagnostics:
            unique_vals, counts = np.unique(mask, return_counts=True)
            print(f"\nMASK DIAGNOSTICS:")
            for val, count in zip(unique_vals, counts):
                print(f"    Value {val}: {count} occurrences")

        if plot:
            pyissm.plot.plot_model_field(md, mask, show_cbar = True, cbar_kwargs = {'label': 'Ice Mask'}); plt.show(block = False)

        # Define min/max vertex lengths in specific regions
        print(f"\n-- Setting min/max vertex dimensions...")
        hmax_v = np.full(md.mesh.numberofvertices, np.nan)
        hmin_v = np.full(md.mesh.numberofvertices, np.nan)

        hmax_v[(vel > 50) & (mask == 2)] = 1500 # Max length on fast-flowing grounded ice
        hmin_v[(mask == 3)] = 500 # Min length on ice shelves
        hmax_v[(mask == 3)] = 5000 # Max length on ice shelves
        hmax_v[(mask == 0)] = 5000 # Max length in ocean

        # Adjust mesh with specified metrics
        print(f"\n-- Remeshing with specified metrics...")
        md = pyissm.model.mesh.bamg(md,
                                    hmin = 50,
                                    hmax = 50e3,
                                    hmaxVertices = hmax_v,
                                    hminVertices = hmin_v,
                                    maxnbv = 2e6,
                                    field = vel,
                                    err = 1,
                                    gradation = 1.2)

        # Remove bamg private data to allow additional remeshes
        md.private.bamg = {}

        if diagnostics:
            print(f"\nMESH DIAGNOSTICS:")
            print(f"   Number of elements: {md.mesh.numberofelements}")
            print(f"   Number of vertices: {md.mesh.numberofvertices}")

    # Set georefernce information
    [md.mesh.lat, md.mesh.long] = pyissm.tools.general.xy_to_ll(md.mesh.x, md.mesh.y, -1)
    md.mesh.epsg = 3031

    print(f"\nFinal mesh: {md.mesh.numberofvertices} nodes; {md.mesh.numberofelements} elements")

    if plot:
        areas = pyissm.model.mesh.get_element_areas_volumes(md.mesh.elements, md.mesh.x, md.mesh.y)
        fig, ax = pyissm.plot.plot_model_field(md,
                                               np.sqrt(areas*2)/1e3,
                                               show_cbar = True,
                                               vmin = 0.25,
                                               vmax = 10,
                                               plot_data_on='elements',
                                               cmap = 'plasma_r',
                                               cbar_kwargs = {'label': 'Approx. element edge length (km)'})
        ax.set_title('Final Mesh: Velocity-adapted w/ 2 refinement passes')
        plt.show(block = False)

    if save:
        print(f"\nSaving model to {model_dir}/AIS3_mesh.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_mesh.nc')


## ------------------------------------
## Parameterise model
## ------------------------------------
if 'param' in steps:

    print("-------------------------------------------------------------")
    print(f" PARAMETERIZING MODEL"                                       )
    print("-------------------------------------------------------------")

    print(f"-- Loading model mesh...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_mesh.nc')

    print(f"-- Parameterising model using {param_file}...")
    md = pyissm.model.param.parameterize(md, param_file)

    print(f"-- Set flow equation to SSA...")
    md = pyissm.model.param.set_flow_equation(md, SSA = 'all')

    print(f"-- Setting Boundary Conditions...")
    # -------- Set Stress Balance BCs --------
    ## Initialize empty fields
    md.stressbalance.spcvx = np.full(md.mesh.numberofvertices, np.nan)
    md.stressbalance.spcvy = np.full(md.mesh.numberofvertices, np.nan)
    md.stressbalance.spcvz = np.full(md.mesh.numberofvertices, np.nan)

    ## Find ice nodes on the edge of the domain  (NOTE: There are none when an ocean buffer is included)
    pos = (md.mask.ice_levelset < 0) & (md.mesh.vertexonboundary.astype(bool))

    ## Set Dirichlet BCs on VX and VY fields based on initial velocities; Set VZ as 0
    md.stressbalance.spcvx[pos] = md.initialization.vx[pos]
    md.stressbalance.spcvy[pos] = md.initialization.vy[pos]
    md.stressbalance.spcvz[pos] = 0 #TODO: Specify this here?

    md.stressbalance.referential = np.nan * np.ones((md.mesh.numberofvertices, 6))
    md.stressbalance.loadingforce = np.zeros((md.mesh.numberofvertices, 3))

    # -------- Set thermal Balance BCs --------
    md.thermal.spctemperature = md.initialization.temperature.copy()

    if diagnostics:
        print(f"\nMASK DIAGNOSTICS:")
        print(f" - Ice Levelset:")
        unique_vals, counts = np.unique(md.mask.ice_levelset, return_counts=True)
        for val, count in zip(unique_vals, counts):
            print(f"    Value {val}: {count} occurrences")

        print(f" Ocean Levelset (Binary):")
        ocean_binary = md.mask.ocean_levelset <= 0
        unique_vals, counts = np.unique(ocean_binary, return_counts=True)
        for val, count in zip(unique_vals, counts):
            print(f"    Value {val}: {count} occurrences")

        print(f"\nGEOMETRY DIAGNOSTICS:")
        print(f" - Surface elevation:")
        print(f"   min = {np.min(md.geometry.surface):.2f} m")
        print(f"   max = {np.max(md.geometry.surface):.2f} m")
        print(f" - Bed elevation:")
        print(f"   min = {np.min(md.geometry.bed):.2f} m")
        print(f"   max = {np.max(md.geometry.bed):.2f} m")
        print(f" - Thickness:")
        print(f"   min = {np.min(md.geometry.thickness):.2f} m")
        print(f"   max = {np.max(md.geometry.thickness):.2f} m")

        print(f"VELOCITY DIAGNOSTICS:")
        print(f"   Min observed vx: {np.min(md.inversion.vx_obs):.2f} m/yr")
        print(f"   Max observed vx: {np.max(md.inversion.vx_obs):.2f} m/yr")
        print(f"   Min observed vy: {np.min(md.inversion.vy_obs):.2f} m/yr")
        print(f"   Max observed vy: {np.max(md.inversion.vy_obs):.2f} m/yr")
        print(f"   Min observed vel: {np.min(md.inversion.vel_obs):.2f} m/yr")
        print(f"   Max observed vel: {np.max(md.inversion.vel_obs):.2f} m/yr")

        print(f"INITIAL PRESSURE DIAGNOSTICS:")
        print(f"   Min initial pressure: {np.min(md.initialization.pressure):.2f} Pa")
        print(f"   Max initial pressure: {np.max(md.initialization.pressure):.2f} Pa")

        print(f"GEOTHERMAL HEAT FLOW DIAGNOSTICS:")
        print(f"   Min geothermal heat flux: {np.min(md.basalforcings.geothermalflux):.7f} mW/m2")
        print(f"   Max geothermal heat flux: {np.max(md.basalforcings.geothermalflux):.7f} mW/m2")

        print(f"INITIAL TEMPERATURE DIAGNOSTICS:")
        print(f"   Min initial temp: {np.min(md.initialization.temperature):.2f} K")
        print(f"   Max initial temp: {np.max(md.initialization.temperature):.2f} K")

    if save:
        print(f"\nSaving model to {model_dir}/AIS3_param.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_param.nc')


## ------------------------------------
## SSA Rheology Inversion Sensitivity - Floating Ice
## ------------------------------------
if 'ssa_rheology_floating_inv_sensit' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA RHEOLOGY INVERSION SENSITIVITY - FLOATING ICE"          )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Define general control parameters...")
    md.inversion.iscontrol = 1
    md.verbose.solution = 0
    md.verbose.qmu = 0
    md.verbose.control = 1

    print(f"-- Defining inversion parameters...")
    md.inversion = pyissm.model.classes.inversion.m1qn3(md.inversion)
    md.inversion.iscontrol = 1
    md.inversion.control_parameters = ['MaterialsRheologyBbar']
    md.inversion.min_parameters = pyissm.tools.materials.cuffey(273.15 - 0) * np.ones((md.mesh.numberofvertices, ))
    md.inversion.max_parameters = pyissm.tools.materials.cuffey(273.15 - 70) * np.ones((md.mesh.numberofvertices, ))
    md.inversion.maxsteps = 500
    md.inversion.maxiter = 200

    # Remove icebergs
    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print('-- Extracting floating ice only...')
    mask = (md.mask.ocean_levelset < 0) & (md.mask.ice_levelset < 0) # Floating ice
    mds = md.extract(mask)

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    print(f"-- Setting-up coefficient grid...")
    param_grid = pyissm.inversion.sensitivity.build_parameter_grid(
        {101: [1, 10, 100, 1000],
         103: [1, 10, 100, 1000]})

    print(f"-- Defining mask to exclude 0 velocity...")
    mask = (mds.inversion.vel_obs > 0)

    if save:
        print(f"-- Loading inversion parameter sensitivity...")
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_rheology_floating_inv_sensit',
            run = False,
            load_only = True,
            global_mask = mask)

        print(f"-- Processing inversion parameter sensitivity...")
        diagnostics = pyissm.inversion.sensitivity.compute_sensitivity_diagnostics(manifest, output_dir=f'{model_dir}/AIS3_ssa_rheology_floating_inv_sensit/')

        diagnostics_norm = pyissm.inversion.sensitivity.normalize_diagnostics(diagnostics,
                                                                             columns = ['vel_rmse', 'mean_gradient_magnitude'])
        diagnostics_norm['overall'] = diagnostics_norm['vel_rmse_norm'] + diagnostics_norm['mean_gradient_magnitude_norm']

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (15, 8), constrained_layout = True)
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax1, value = 'vel_rmse')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax2, value = 'ratio_101_103')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax3, value = 'mean_gradient_magnitude')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax4, value = 'cost_total')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax5, value = 'positive_residual_fraction')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics_norm, x = 'cf101', y = 'cf103', ax = ax6, value = 'overall')
        plt.savefig(f'{model_dir}/AIS3_ssa_rheology_floating_inv_sensit/diagnostic_heatmaps.png')

        best_row = diagnostics_norm.loc[diagnostics_norm['overall'].idxmax()]
        print(f"The best run_id is: {best_row['run_id']}. This uses the following coefficient values:")
        print(best_row.filter(regex=r'^cf'))

    else:
        print(f"-- Running inversion parameter sensitivity...")
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_rheology_floating_inv_sensit',
            run = True,
            load_only = False,
            global_mask = mask)


## ------------------------------------
## SSA Rheology Inversion L-Curve - Floating Ice
## ------------------------------------
if 'ssa_rheology_floating_inv_lcurve' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA RHEOLOGY INVERSION L-CURVE - FLOATING ICE"              )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Define general control parameters...")
    md.inversion.iscontrol = 1
    md.verbose.solution = 0
    md.verbose.qmu = 0
    md.verbose.control = 1

    print(f"-- Defining inversion parameters...")
    md.inversion = pyissm.model.classes.inversion.m1qn3(md.inversion)
    md.inversion.iscontrol = 1
    md.inversion.control_parameters = ['MaterialsRheologyBbar']
    md.inversion.min_parameters = pyissm.tools.materials.cuffey(273.15 - 0) * np.ones((md.mesh.numberofvertices, ))
    md.inversion.max_parameters = pyissm.tools.materials.cuffey(273.15 - 70) * np.ones((md.mesh.numberofvertices, ))
    md.inversion.maxsteps = 500
    md.inversion.maxiter = 200

    # Remove icebergs
    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print('-- Extracting floating ice only...')
    mask = (md.mask.ocean_levelset < 0) & (md.mask.ice_levelset < 0) # Floating ice
    mds = md.extract(mask)

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    print(f"-- Setting-up coefficient grid...")
    # Use preferred 101/103 coefficients from sensitivity step
    param_grid = pyissm.inversion.sensitivity.build_parameter_grid(
        {101: [1],
         103: [10],
         502: [1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12]})

    print(f"-- Defining mask to exclude 0 velocity...")
    mask = (mds.inversion.vel_obs > 0)

    if save:
        print(f"-- Loading inversion parameter sensitivity...")
        # Only mask 101 and 103 -- no mask on regularisation.
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve',
            run = False,
            load_only = True,
            coeff_masks = {101: mask,
                           103: mask})

        print(f"-- Processing inversion parameter sensitivity...")
        diagnostics = pyissm.inversion.sensitivity.compute_sensitivity_diagnostics(manifest, output_dir=f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/')

        fig, ax = pyissm.inversion.plot.plot_lcurve(diagnostics)
        ax.set_title('Floating ice rheology inversion - L-curve analysis')
        plt.savefig(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/lcurve.png')

    else:
        print(f"-- Running inversion parameter sensitivity...")
        # Only mask 101 and 103 -- no mask on regularisation.
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve',
            run = True,
            load_only = False,
            coeff_masks = {101: mask,
                           103: mask})


## ------------------------------------
## SSA Friction Inversion Sensitivity - Grounded Ice
## ------------------------------------
## ------------------------------------
## SSA Forward Convergence Check - Friction Domain (no inversion)
## ------------------------------------
# Diagnostic: reproduce the friction inversion domain/BCs exactly, but disable the
# inversion and impose a uniform, physically-reasonable friction C. Solve a SINGLE
# forward stress balance and inspect whether the non-linear (viscosity) iteration
# converges. If the forward solve diverges here, the friction inversion cannot work
# regardless of the cost coefficients -- fix the forward model (BCs) first.
if 'ssa_friction_forward_check' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA FORWARD CONVERGENCE CHECK - FRICTION DOMAIN"            )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Loading SSA floating rheology inversion results ({rheology_lcurve_run})...")
    mds = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/{rheology_lcurve_run}/{rheology_lcurve_run}.nc')

    print(f"-- Updating rheology field from inversion results...")
    md.materials.rheology_B[mds.mesh.extractedvertices - 1] = mds.results.StressbalanceSolution.MaterialsRheologyBbar # Note: -1 for zero-based indexing

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    # Fix negative effective pressure (identical to the inversion setup)
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = 0.07  # match the N floor set in param

    print(f"-- Extracting friction-inversion domain (floating/grounded ice-front BC)...")
    mds = extract_friction_inversion_domain(md)

    print(f"-- Disabling inversion (forward solve only)...")
    mds.inversion.iscontrol = 0
    mds.verbose.solution = 1  # print non-linear convergence to the log

    # Identify purely-floating elements once (these get ~0 friction regardless of the sweep value)
    ocean_elements = mds.mask.ocean_levelset[mds.mesh.elements - 1] # -1 for zero-based indexing
    pos_e = np.where(np.min(ocean_elements, axis=1) < 0)[0]
    flags = np.zeros(mds.mesh.numberofvertices, dtype=bool)
    flags[mds.mesh.elements[pos_e, :] - 1] = True # -1 for zero-based indexing

    mds.transient = pyissm.model.classes.transient.deactivate_all(mds.transient)

    # Same non-linear solver tolerances as the inversion forward solves
    mds.stressbalance.restol = 0.01
    mds.stressbalance.reltol = 0.1
    mds.stressbalance.abstol = np.nan
    mds.settings.solver_residue_threshold = 1e-3

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    # Sweep a few uniform grounded-ice C values. Basal drag scales like C^2, so this brackets
    # the coefficient magnitude needed to tame the ice by orders of magnitude. If a large enough
    # C brings velocities into the physical range, the fix is to raise the inversion bounds /
    # rescale C; if even the largest value blows up, the friction is not coupling (a real bug).
    forward_check_C_values = [5000]

    for Cval in forward_check_C_values:
        run_name = f'AIS3_friction_forward_check_C{Cval}'
        print(f"\n-- Uniform grounded C = {Cval}  ({run_name}) --")

        mds.friction.C = np.full(mds.mesh.numberofvertices, float(Cval))
        mds.friction.C[flags] = 0.05
        mds.miscellaneous.name = run_name

        if save:
            mdi = pyissm.model.execute.solve(mds, 'Stressbalance', load_only = True, runtime_name = False)
            vel = np.asarray(mdi.results.StressbalanceSolution.Vel).ravel()
            vel_obs = np.asarray(mdi.inversion.vel_obs).ravel()
            grounded = (mdi.mask.ice_levelset < 0) & (mdi.mask.ocean_levelset > 0)
            print(f"   obs max={np.nanmax(vel_obs):.0f} | mod max={np.nanmax(vel):.0f} "
                  f"med={np.nanmedian(vel):.0f} | grounded med={np.nanmedian(vel[grounded]):.0f} "
                  f"| nodes>1e4: {(vel > 1e4).sum()}/{vel.size} ({100*(vel > 1e4).mean():.1f}%)")
        else:
            pyissm.model.execute.solve(mds, 'Stressbalance', load_only = False, runtime_name = False)


## ------------------------------------
## SSA Forward Convergence Check - Friction Domain (BUDD law experiment)
## ------------------------------------
# Same friction domain and BCs as ssa_friction_forward_check, but replaces the Schoof
# (regularized-Coulomb) law with the Budd/Weertman power law, which has NO Coulomb ceiling
# (tau_b = coefficient^2 * Neff^r * |u|^(s-1) * u, r=q/p, s=1/p). Removing the Cmax*N cap
# should eliminate the unfittable Coulomb-failure nodes and the inversion overshoot. Sweeps a
# uniform coefficient to find the magnitude that gives physical velocities; if it does, Budd
# is worth adopting for the friction inversion. Self-contained -- does not touch param or the
# Schoof blocks, so both laws remain available for comparison.
if 'ssa_friction_forward_check_budd' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA FORWARD CHECK (BUDD LAW) - FRICTION DOMAIN"             )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Loading SSA floating rheology inversion results ({rheology_lcurve_run})...")
    mdr = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/{rheology_lcurve_run}/{rheology_lcurve_run}.nc')
    md.materials.rheology_B[mdr.mesh.extractedvertices - 1] = mdr.results.StressbalanceSolution.MaterialsRheologyBbar

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    # Fix negative effective pressure (same floored N as the Schoof setup)
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = 0.07

    print(f"-- Extracting friction-inversion domain (floating/grounded ice-front BC)...")
    mds = extract_friction_inversion_domain(md)

    print(f"-- Switching friction law to Budd (default class, no Coulomb ceiling)...")
    # default(other) inherits effective_pressure / limit / coupling from the Schoof friction.
    # p, q are per-element: s = 1/p, r = q/p. p=3, q=3 -> tau_b ~ N*|u|^(1/3) (Budd, linear
    # N-dependence, m=3 sliding). Tune p/q as needed (q=0 -> Weertman, no N-dependence).
    mds.friction = pyissm.model.classes.friction.default(mds.friction)
    mds.friction.p = np.full(mds.mesh.numberofelements, 3.0)
    mds.friction.q = np.full(mds.mesh.numberofelements, 3.0)
    mds.friction.coupling = 3  # use the provided (floored) effective_pressure

    print(f"-- Disabling inversion (forward solve only)...")
    mds.inversion.iscontrol = 0
    mds.verbose.solution = 1

    # Purely-floating elements -> ~0 friction
    ocean_elements = mds.mask.ocean_levelset[mds.mesh.elements - 1]
    pos_e = np.where(np.min(ocean_elements, axis=1) < 0)[0]
    flags = np.zeros(mds.mesh.numberofvertices, dtype=bool)
    flags[mds.mesh.elements[pos_e, :] - 1] = True

    mds.transient = pyissm.model.classes.transient.deactivate_all(mds.transient)
    mds.stressbalance.restol = 0.01
    mds.stressbalance.reltol = 0.1
    mds.stressbalance.abstol = np.nan
    mds.settings.solver_residue_threshold = 1e-3

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    # Sweep uniform Budd coefficient (magnitude unknown a priori; brackets a few decades).
    budd_coeff_values = [1, 10, 100, 1000]

    for cval in budd_coeff_values:
        run_name = f'AIS3_friction_forward_check_budd_{cval}'
        print(f"\n-- Uniform Budd coefficient = {cval}  ({run_name}) --")
        mds.friction.coefficient = np.full(mds.mesh.numberofvertices, float(cval))
        mds.friction.coefficient[flags] = 0.05
        mds.miscellaneous.name = run_name

        if save:
            mdi = pyissm.model.execute.solve(mds, 'Stressbalance', load_only = True, runtime_name = False)
            vel = np.asarray(mdi.results.StressbalanceSolution.Vel).ravel()
            vel_obs = np.asarray(mdi.inversion.vel_obs).ravel()
            grounded = (mdi.mask.ice_levelset < 0) & (mdi.mask.ocean_levelset > 0)
            print(f"   obs max={np.nanmax(vel_obs):.0f} | mod max={np.nanmax(vel):.0f} "
                  f"med={np.nanmedian(vel):.0f} | grounded med={np.nanmedian(vel[grounded]):.0f} "
                  f"| nodes>1e4: {(vel > 1e4).sum()}/{vel.size} ({100*(vel > 1e4).mean():.1f}%)")
        else:
            pyissm.model.execute.solve(mds, 'Stressbalance', load_only = False, runtime_name = False)


## ------------------------------------
## SSA Friction Inversion Sensitivity - Grounded Ice
## ------------------------------------
if 'ssa_friction_inv_sensit' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA FRICTION INVERSION SENSITIVITY - GROUNDED ICE"          )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Loading SSA floating rheology inversion results...")
    mds = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/run_004_1_10_1e-17/run_004_1_10_1e-17.nc')

    print(f"-- Updating rheology field from inversion results...")
    md.materials.rheology_B[mds.mesh.extractedvertices - 1] = mds.results.StressbalanceSolution.MaterialsRheologyBbar # Note: -1 for zero-based indexing

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print(f"-- Define general control parameters...")
    md.inversion.iscontrol = 1
    md.verbose.solution = 0
    md.verbose.qmu = 0
    md.verbose.control = 1

    # Fix negative effective pressure
    # TODO: Update this in param
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = 0.07  # match the N floor set in param

    print(f"-- Extracting friction-inversion domain (floating/grounded ice-front BC)...")
    mds = extract_friction_inversion_domain(md)

    print(f"-- Overriding friction.coupling = {friction_coupling}...")
    mds.friction.coupling = friction_coupling

    print(f"-- Defining inversion parameters...")
    mds.inversion = pyissm.model.classes.inversion.m1qn3(mds.inversion)
    fric_control, fric_field, fric_min, fric_max = friction_law_info(mds)  # Schoof or Budd
    mds.inversion.control_parameters = [fric_control]
    mds.inversion.min_parameters = np.full(mds.mesh.numberofvertices, fric_min)
    mds.inversion.max_parameters = np.full(mds.mesh.numberofvertices, fric_max) ## TODO: Set upper bound once better constrained
    # NOTE: The friction inversion converges in ~15 steps on this mesh; a large budget lets
    # m1qn3 overshoot the minimum into a region where the forward SSA solve stops converging
    # (100 nonlinear iterations exceeded) and the final iterate can be worse than the minimum.
    # VERIFIED: a maxsteps=500/200 test sweep (matching the rheology budget) OOM-killed 17/25
    # cells at the 100GB ceiling -- the well-behaved cells converge at ~14 steps (so more budget
    # changes nothing), while the rest overshoot into non-converging forward solves that run away
    # on memory. Keep the budget small; 30/50 is the deliberate protection against this.
    mds.inversion.maxsteps = 30
    mds.inversion.maxiter = 50

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    # No friction on PURELY floating ice elements
    # TODO: Initialise the friction field as a float in param to prevent the need to convert it here to avoid >0 consistency issue
    ocean_elements = mds.mask.ocean_levelset[mds.mesh.elements - 1] # -1 for zero-based indexing
    pos_e = np.where(np.min(ocean_elements, axis=1) < 0)[0]
    flags = np.zeros(mds.mesh.numberofvertices, dtype=bool)
    flags[mds.mesh.elements[pos_e, :] - 1] = True # -1 for zero-based indexing
    _fld = getattr(mds.friction, fric_field).astype(float)
    _fld[flags] = 0.05
    setattr(mds.friction, fric_field, _fld)
    mds.inversion.min_parameters[flags] = 0.0
    mds.inversion.max_parameters[flags] = 0.0

    mds.transient = pyissm.model.classes.transient.deactivate_all(mds.transient)

    mds.stressbalance.restol = 0.01
    mds.stressbalance.reltol = 0.1
    mds.stressbalance.abstol = np.nan
    mds.settings.solver_residue_threshold = 1e-3

    print(f"-- Setting-up coefficient grid...")
    param_grid = pyissm.inversion.sensitivity.build_parameter_grid(
        {101: [0.1, 1, 10, 100, 1000],
         103: [0.1, 1, 10, 100, 1000]})

    # TODO: Fill NaN obs vel with 0, not NN so that these regions can be excluded. Update in Param
    # TODO: Exclude ice-front from inversion as well -- mds.inversion.cost_functions_coefficients(iceFront, 1:2) = 0
    # NOTE: An a-priori Coulomb-failure cost mask (tau_d = rho*g*H*|grad(s)| > Cmax*N) was tried
    # here to drop unfittable nodes, but the static driving-stress proxy did not match the actual
    # dynamic blowup cells (bulk fit unchanged, and it pushed some runs into overshoot). Reverted.
    print(f"-- Defining mask to exclude 0 velocity...")
    mask = (mds.inversion.vel_obs > 0)

    if save:
        print(f"-- Loading inversion parameter sensitivity...")
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_friction_inv_sensit',
            run = False,
            load_only = True,
            global_mask = mask)

        print(f"-- Processing inversion parameter sensitivity...")
        diagnostics = pyissm.inversion.sensitivity.compute_sensitivity_diagnostics(manifest, output_dir=f'{model_dir}/AIS3_ssa_friction_inv_sensit/')

        diagnostics_norm = pyissm.inversion.sensitivity.normalize_diagnostics(diagnostics,
                                                                             columns = ['vel_rmse', 'mean_gradient_magnitude'])
        diagnostics_norm['overall'] = diagnostics_norm['vel_rmse_norm'] + diagnostics_norm['mean_gradient_magnitude_norm']

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (15, 8), constrained_layout = True)
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax1, value = 'vel_rmse')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax2, value = 'ratio_101_103')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax3, value = 'mean_gradient_magnitude')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax4, value = 'cost_total')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics, x = 'cf101', y = 'cf103', ax = ax5, value = 'positive_residual_fraction')
        pyissm.inversion.plot.plot_sensitivity_heatmap(diagnostics_norm, x = 'cf101', y = 'cf103', ax = ax6, value = 'overall')
        plt.savefig(f'{model_dir}/AIS3_ssa_friction_inv_sensit/diagnostic_heatmaps.png')

        best_row = diagnostics_norm.loc[diagnostics_norm['overall'].idxmax()]
        print(f"The best run_id is: {best_row['run_id']}. This uses the following coefficient values:")
        print(best_row.filter(regex=r'^cf'))

    else:
        print(f"-- Running inversion parameter sensitivity...")
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_friction_inv_sensit',
            run = True,
            load_only = False,
            global_mask = mask)


## ------------------------------------
## SSA Friction Inversion L-Curve - Grounded Ice
## ------------------------------------
if 'ssa_friction_inv_lcurve' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA FRICTION INVERSION L-CURVE - GROUNDED ICE"              )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Loading SSA floating rheology inversion results...")
    mds = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/{rheology_lcurve_run}/{rheology_lcurve_run}.nc')

    print(f"-- Updating rheology field from inversion results...")
    md.materials.rheology_B[mds.mesh.extractedvertices - 1] = mds.results.StressbalanceSolution.MaterialsRheologyBbar # Note: -1 for zero-based indexing

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print(f"-- Define general control parameters...")
    md.inversion.iscontrol = 1
    md.verbose.solution = 0
    md.verbose.qmu = 0
    md.verbose.control = 1

    # Fix negative effective pressure
    # TODO: Update this in param
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = 0.07  # match the N floor set in param

    print(f"-- Extracting friction-inversion domain (floating/grounded ice-front BC)...")
    mds = extract_friction_inversion_domain(md)

    print(f"-- Overriding friction.coupling = {friction_coupling}...")
    mds.friction.coupling = friction_coupling

    print(f"-- Defining inversion parameters...")
    mds.inversion = pyissm.model.classes.inversion.m1qn3(mds.inversion)
    fric_control, fric_field, fric_min, fric_max = friction_law_info(mds)  # Schoof or Budd
    mds.inversion.control_parameters = [fric_control]
    mds.inversion.min_parameters = np.full(mds.mesh.numberofvertices, fric_min)
    mds.inversion.max_parameters = np.full(mds.mesh.numberofvertices, fric_max) ## TODO: Set upper bound once better constrained
    # See note in ssa_friction_inv_sensit: keep budget small (500/200 OOM-killed most cells).
    mds.inversion.maxsteps = 30
    mds.inversion.maxiter = 50

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    # No friction on PURELY floating ice elements
    ocean_elements = mds.mask.ocean_levelset[mds.mesh.elements - 1] # -1 for zero-based indexing
    pos_e = np.where(np.min(ocean_elements, axis=1) < 0)[0]
    flags = np.zeros(mds.mesh.numberofvertices, dtype=bool)
    flags[mds.mesh.elements[pos_e, :] - 1] = True # -1 for zero-based indexing
    _fld = getattr(mds.friction, fric_field).astype(float)
    _fld[flags] = 0.05
    setattr(mds.friction, fric_field, _fld)
    mds.inversion.min_parameters[flags] = 0.0
    mds.inversion.max_parameters[flags] = 0.0

    mds.transient = pyissm.model.classes.transient.deactivate_all(mds.transient)

    mds.stressbalance.restol = 0.01
    mds.stressbalance.reltol = 0.1
    mds.stressbalance.abstol = np.nan
    mds.settings.solver_residue_threshold = 1e-3

    print(f"-- Setting-up coefficient grid...")
    # Use preferred 101/103 coefficients from the friction sensitivity step; sweep regularisation.
    # BUG FIX: this is a FRICTION inversion, so the gradient regularisation is 501
    # (DragCoefficientAbsGradient), NOT 502 (RheologyBbarAbsGradient). Using 502 here regularised
    # a field that isn't a control parameter in this inversion, so it was inert -- which is why the
    # earlier friction L-curve was flat across 5 orders of magnitude. 502 is correct only for the
    # rheology-B inversion (see the ssa_rheology_floating_inv_lcurve block).
    param_grid = pyissm.inversion.sensitivity.build_parameter_grid(
        {101: [friction_cf101],
         103: [friction_cf103],
         501: [1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12]})

    print(f"-- Defining mask to exclude 0 velocity...")
    mask = (mds.inversion.vel_obs > 0)

    if save:
        print(f"-- Loading inversion parameter sensitivity...")
        # Only mask 101 and 103 -- no mask on regularisation.
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_friction_inv_lcurve',
            run = False,
            load_only = True,
            coeff_masks = {101: mask,
                           103: mask})

        print(f"-- Processing inversion parameter sensitivity...")
        diagnostics = pyissm.inversion.sensitivity.compute_sensitivity_diagnostics(manifest, output_dir=f'{model_dir}/AIS3_ssa_friction_inv_lcurve/')

        fig, ax = pyissm.inversion.plot.plot_lcurve(diagnostics)
        ax.set_title('Grounded ice friction inversion - L-curve analysis')
        plt.savefig(f'{model_dir}/AIS3_ssa_friction_inv_lcurve/lcurve.png')

    else:
        print(f"-- Running inversion parameter sensitivity...")
        # Only mask 101 and 103 -- no mask on regularisation.
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_friction_inv_lcurve',
            run = True,
            load_only = False,
            coeff_masks = {101: mask,
                           103: mask})


## ------------------------------------
## Assemble inverted fields and solve stress balance - Full domain
## ------------------------------------
if 'ssa_inverted_solve' in steps:

    print("-------------------------------------------------------------")
    print(f" ASSEMBLING INVERTED MODEL AND SOLVING STRESS BALANCE"       )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Updating rheology B from floating-ice rheology L-curve ({rheology_lcurve_run})...")
    mdr = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/{rheology_lcurve_run}/{rheology_lcurve_run}.nc')
    md.materials.rheology_B[mdr.mesh.extractedvertices - 1] = mdr.results.StressbalanceSolution.MaterialsRheologyBbar # Note: -1 for zero-based indexing

    print(f"-- Updating friction field from grounded-ice friction L-curve ({friction_lcurve_run})...")
    mdf = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_friction_inv_lcurve/{friction_lcurve_run}/{friction_lcurve_run}.nc')
    fric_control, fric_field, _, _ = friction_law_info(md)  # Schoof: FrictionC / C ; Budd: FrictionCoefficient / coefficient
    _fld = getattr(md.friction, fric_field).astype(float)
    _fld[mdf.mesh.extractedvertices - 1] = getattr(mdf.results.StressbalanceSolution, fric_control) # -1 for zero-based indexing
    setattr(md.friction, fric_field, _fld)

    # Fix negative effective pressure (consistent with the inversion setup)
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = 0.07  # match the N floor set in param

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print(f"-- Disabling inversion (forward solve only)...")
    md.inversion.iscontrol = 0
    md.verbose.solution = 1

    print(f"-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_inverted'
    md.cluster = cluster
    md.settings.waitonlock = 0

    if save:
        print(f"-- Loading stress balance solution...")
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only = True, runtime_name = False)

        if diagnostics:
            vel = md.results.StressbalanceSolution.Vel
            vel_obs = md.inversion.vel_obs
            residual = vel - vel_obs
            print(f"\nFORWARD SOLVE DIAGNOSTICS:")
            print(f"   Max modelled velocity: {np.nanmax(vel):.2f} m/yr")
            print(f"   Velocity RMSE vs obs:  {np.sqrt(np.nanmean(residual**2)):.2f} m/yr")

        if plot:
            fig, ax = pyissm.plot.plot_model_field(md, md.results.StressbalanceSolution.Vel,
                                                   show_cbar = True,
                                                   cmap = 'PuOr',
                                                   cbar_kwargs = {'label': 'Modelled velocity (m/a)'})
            ax.set_title('Inverted model - SSA stress balance velocity')
            plt.savefig(f'{model_dir}/AIS3_inverted_velocity.png')

        print(f"\nSaving inverted model to {model_dir}/AIS3_inverted.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_inverted.nc')

    else:
        print(f"-- Submitting stress balance solve...")
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only = False, runtime_name = False)


## ------------------------------------
## Transient relaxation - Full domain
## ------------------------------------
if 'ssa_relaxation' in steps:

    print("-------------------------------------------------------------")
    print(f" TRANSIENT RELAXATION"                                       )
    print("-------------------------------------------------------------")

    print(f"-- Loading inverted model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_inverted.nc')

    print(f"-- Configuring transient relaxation...")
    md.inversion.iscontrol = 0
    md.verbose.solution = 1

    # Relax the dynamics + free surface to damp initialisation shock.
    # Thermal and SMB are held fixed; grounding line is allowed to migrate.
    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.isthermal = 0
    md.transient.isgroundingline = 1
    md.groundingline.migration = 'SubelementMigration'
    md.transient.requested_outputs = ['default', 'Vel', 'Thickness', 'Surface', 'Base', 'MaskOceanLevelset']

    # Short relaxation window -- TODO: tune final_time / time_step for your domain.
    md.timestepping.start_time = 0
    md.timestepping.final_time = 20    # years
    md.timestepping.time_step  = 0.05  # years

    print(f"-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_relaxed'
    md.cluster = cluster
    md.settings.waitonlock = 0

    if save:
        print(f"-- Loading transient solution...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only = True, runtime_name = False)

        if diagnostics:
            dH = md.results.TransientSolution.Thickness[-1] - md.geometry.thickness
            print(f"\nRELAXATION DIAGNOSTICS:")
            print(f"   Max |dH| over relaxation: {np.nanmax(np.abs(dH)):.2f} m")
            print(f"   Mean |dH| over relaxation: {np.nanmean(np.abs(dH)):.2f} m")

        print(f"\nSaving relaxed model to {model_dir}/AIS3_relaxed.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_relaxed.nc')

    else:
        print(f"-- Submitting transient relaxation...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only = False, runtime_name = False)
