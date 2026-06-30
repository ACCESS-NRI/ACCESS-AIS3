import pyissm
import ccdtools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import os

## ------------------------------------
## Configure options
## ------------------------------------

# Change directory to gdata to prevent storage limits in $HOME
os.chdir('/g/data/au88/jh7060/ACCESS-AIS3/')
os.environ['ISSM_DIR'] = '/g/data/vk83/apps/spack/1.1/release/linux-x86_64/issm-git.2026.05.18_2026.05.18-kgta35igm37z4qnqnul7rcmgx2inftqd'

# Should plots be generated?
plot = True
diagnostics = True
save = True
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
    'ssa_friction_inv_sensit'
]

# Define steps to run
# steps = ['process_domain', 'mesh', 'param']
# steps = ['mesh', 'param']
# steps = ['ssa_rheology_floating_inv_sensit']
# steps = ['ssa_rheology_floating_inv_sensit']
steps = ['ssa_rheology_floating_inv_lcurve']
# steps = ['ssa_rheology_floating_inv']
# steps = ['ssa_friction_inv_sensit']


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
if 'ssa_friction_inv_sensit' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA FRICTION INVERSION SENSITIVITY - GROUNDED ICE"          )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Loading SSA floating rheology inversion results...")
    mds = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/run_0003_1_10_1e-17/run_0003_1_10_1e-17.nc')

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
    md.friction.effective_pressure_limit = 0.01

    print(f"-- Extracting ice only (including ice-front)...")
    ice_levelset_elements = pyissm.tools.interp.vertex_to_element(md, md.mask.ice_levelset)
    mds = md.extract(ice_levelset_elements < 1)

    # Set only Neumann BCs on the floating ice-front (Neumann with Dirichlet constraint elsewhere)
    iceFront = (mds.mask.ice_levelset >= 0) & (mds.mask.ocean_levelset < 0)
    mds.stressbalance.spcvx[iceFront] = np.nan
    mds.stressbalance.spcvy[iceFront] = np.nan
    mds.stressbalance.spcvz[iceFront] = np.nan

    print(f"-- Defining inversion parameters...")
    mds.inversion = pyissm.model.classes.inversion.m1qn3(mds.inversion)
    mds.inversion.control_parameters = ['FrictionC']
    mds.inversion.min_parameters = np.full(mds.mesh.numberofvertices, 0.05)
    mds.inversion.max_parameters = np.full(mds.mesh.numberofvertices, 250**2) ## TODO: Set upper bound once better constrained
    mds.inversion.maxsteps = 500
    mds.inversion.maxiter = 200

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    # No friction on PURELY floating ice elements
    # TODO: Initialise the friction field as a float in param to prevent the need to convert it here to avoid >0 consistency issue
    ocean_elements = mds.mask.ocean_levelset[mds.mesh.elements - 1] # -1 for zero-based indexing
    pos_e = np.where(np.min(ocean_elements, axis=1) < 0)[0]
    flags = np.zeros(mds.mesh.numberofvertices, dtype=bool)
    flags[mds.mesh.elements[pos_e, :] - 1] = True # -1 for zero-based indexing
    mds.friction.C = mds.friction.C.astype(float)
    mds.friction.C[flags] = 0.05
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
