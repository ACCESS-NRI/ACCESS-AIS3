import pyissm
import ccdtools as ccdtools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import os


def friction_law_info(md):
    """Return (control_parameter, field_attr, min_bound, max_bound) for md's friction law.

    Schoof (regularized Coulomb) inverts 'FrictionC' (field md.friction.C); Budd/Weertman (the
    'default' class) inverts 'FrictionCoefficient' (field md.friction.coefficient). The saved
    friction class (set by friction_law in ais_0.1_param.py) is the single source of truth.
    """
    if type(md.friction).__name__ == 'default':   # Budd / Weertman power law
        # VALIDATED bounds [0.05, 900] for p=q=1 (grounded RMSE 61.4 unregularised / 60.4 with
        # cf501=0.0001, see ais_0.1_param.py). The earlier [0.1, 10] bounds were tuned for the
        # superseded p=q=3 law (u ~ C^-6, `friconly_nfix`, RMSE 98.9) -- under p=q=1 (u ~ C^-2)
        # fast ice needs a much larger C for the same resisting stress, and that ceiling pinned
        # 41% of the domain at C=10 the first time p=1 was tried with it.
        return 'FrictionCoefficient', 'coefficient', 0.05, 900
    # Schoof (regularized Coulomb): tested directly for the Siple Coast trunk deficit and ruled
    # out on physics grounds, not just numerics -- see docs/inversion_worklog.md section 5.4.
    # Coupled-domain adjoint inversion of FrictionC is unstable near the Coulomb cap regardless
    # of solver settings, and a forward-only sweep across the full documented Cmax range
    # (0.17-0.84) left the Siple Coast trunk ratio completely unchanged from Budd's. Kept here
    # only so friction_law='schoof' remains loadable; not the recommended path.
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
save = True
inversion_sensitivity = False

# Define execution directory
execution_dir = '/g/data/au88/jh7060/ACCESS-AIS3/execution'

# Define location to save final models
model_dir = '/g/data/au88/jh7060/ACCESS-AIS3/models'

# Define domain_file
domain_file = ('/g/data/au88/jh7060/ACCESS-AIS3/assets/ais_domain.exp')

# Define param_file
param_file = ('/g/data/au88/jh7060/ACCESS-AIS3/config/ais_0.1_param.py')

# Define cluster requirements
cluster = pyissm.model.classes.cluster.gadi()
cluster.codepath = os.environ['ISSM_DIR']+'/bin'
cluster.executionpath = execution_dir
cluster.storage = 'gdata/au88+gdata/vk83'
cluster.moduleuse = ['/g/data/vk83/modules/']
cluster.moduleload = ['access-issm_ad/2026.05.0']  # was access-issm/2025.11.0: executing a
# 2026.05.18 binary under a 2025.11.0 module load -- a stale-module mismatch caught and fixed
# across every scratchpad script this session; production had not been updated to match.
# np/memory: 32 cores / 100GB is under-provisioned -- this mesh needs ~130GB minimum even at
# 32 ranks (see docs/inversion_worklog.md section 8), which is the likely real cause of the
# OOM history noted below on the maxsteps line, not maxsteps itself. 48 cores / 190GB is the
# configuration validated as SU-optimal this session (>96 cores was actively worse).
cluster.np = 48
cluster.memory = 190
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
    'ssa_friction_inv_reg_lcurve',
    'ssa_inverted_solve',
    'ssa_relaxation',
    'ho_thermal_steadystate',
    'ho_friction_inv',
    'melt_gamma_tuning',
    'ho_relaxation',
    'historical_dhdt_tuning',
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
steps = ['param']
# steps = ['ssa_friction_inv_lcurve']
# steps = ['ssa_friction_inv_sensit']

## ------------------------------------
## Chosen inversion runs (update after inspecting sensit / lcurve diagnostics)
## ------------------------------------
# Floating-ice rheology B field taken from the rheology L-curve (cf502 regularisation).
rheology_lcurve_run = 'run_004_1_10_1e-17'

# Preferred 101/103 cost-function coefficients for the friction inversion.
# The cf101=1000/cf103=0.1 choice below (run_021, vel_rmse=960.5) came from a sensit sweep run
# against the C_init=10 dead-zone bug (see ais_0.1_param.py): with u ~ C^-6 and the model stuck
# at zero velocity everywhere, that sweep's vel_rmse was never measuring model skill (it was
# ~equal to RMS(v_obs) itself, i.e. the null model). Every cell in that grid is void.
# VALIDATED instead (grounded RMSE 98.9, `friconly_nfix`): cf101=10, cf103=100 -- log-weighted,
# so the slow interior (which absolute weighting like 1000/0.1 effectively ignores) contributes
# to the fit. 10/100 was carried through every successful run this pipeline is based on.
friction_cf101 = 10
friction_cf103 = 100

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

# m1qn3's relative gradient-norm stopping tolerance (default 1e-4). ROOT CAUSE of every earlier
# p=q=1 "convergence" that silently never fit anything: under p=1's much gentler cost-function
# landscape than p=3's, ||g(X)||/||g(X0)|| falls below 1e-4 by iteration ~16, while the cost is
# still falling fast (not flattening) and the fit is nowhere near done -- a false stop, not a
# real one. Tightened well below anything that can trigger this early, so every successful run
# in this pipeline instead stops on dxmin (step-size), the genuine convergence criterion.
friction_inv_gttol = 1e-8

# Grounded-ice friction C field, in two stages (see ssa_friction_inv_lcurve /
# ssa_friction_inv_reg_lcurve below):
#   1. `friction_baseline_run` -- the UNREGULARISED (cf501 effectively off) p=q=1 baseline,
#      grounded RMSE 61.4, used only as the warm-start for stage 2 below (its own C field is
#      usable but ~5x rougher, C-field roughness 0.82 vs the p=q=3 baseline's 0.17).
#   2. `friction_lcurve_run` -- warm-started from (1), light DragCoefficientAbsGradient
#      regularisation (cf501). cf501=0.0001 is the validated corner: it drops the C-field
#      roughness to 0.18 (matching p=q=3) while the RMSE *improves* further, to 60.4 -- not a
#      tradeoff, both axes move the same direction. This is the field `ssa_inverted_solve` uses.
friction_baseline_run = f'run_001_{friction_cf101}_{friction_cf103}_1e-08'
friction_lcurve_run = f'run_001_{friction_cf101}_{friction_cf103}_0.0001'


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
    mds.inversion.max_parameters = np.full(mds.mesh.numberofvertices, fric_max)  # bounds fixed above (friction_law_info)
    # The "converges in ~15 steps" premise here was measured against the C_init=10 dead-zone bug
    # (see ais_0.1_param.py): with the model stuck at zero velocity, m1qn3 had nothing to do and
    # "converged" immediately by never moving. VALIDATED (grounded RMSE 98.9, `friconly_nfix`):
    # the real fit needs 192 m1qn3 iterations to reach dxmin, with the cost still improving as
    # late as iteration ~150. 30/50 would cut this run off before the fit has developed at all.
    # The OOM history this comment used to cite is addressed above (cluster.np/memory) -- 100GB
    # was under the ~130GB this mesh needs even at 32 ranks, not a consequence of maxsteps itself.
    mds.inversion.maxsteps = 500
    mds.inversion.maxiter = 500

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
    # VALIDATED FIX: restrict the 101/103 velocity misfit to GROUNDED observed ice. This control
    # (FrictionCoefficient) is frozen at ~0 on floating ice, so shelf model-obs mismatch could
    # never be corrected where it arises -- it pushed through the grounding line and was absorbed
    # by grounded friction instead (in `friconly_nfix`, floating vertices were 16% of the observed
    # misfit before this fix). Shelves stay in the domain and still buttress; they just stop
    # driving the grounded control.
    print(f"-- Defining mask to exclude 0 velocity and restrict to grounded ice...")
    mask = (mds.inversion.vel_obs > 0) & (mds.mask.ocean_levelset >= 0)

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
    print(f" SSA FRICTION INVERSION - UNREGULARISED BASELINE (p=q=1)"    )
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Loading SSA floating rheology inversion results...")
    mds = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/{rheology_lcurve_run}/{rheology_lcurve_run}.nc')

    print(f"-- Updating rheology field from inversion results...")
    md.materials.rheology_B[mds.mesh.extractedvertices - 1] = mds.results.StressbalanceSolution.MaterialsRheologyBbar # Note: -1 for zero-based indexing

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print(f"-- Flooring thin ice at 100m (numerical stability), preserving observed surface...")
    # Production param.py already floors thickness at 10m (ais_0.1_param.py:62), with N
    # consistently floored against that same 10m value at param time -- internally
    # consistent, but 10m proved numerically fragile during this project's own testing (thin,
    # steep-terrain vertices at that thickness were the source of repeated velocity
    # runaways). VALIDATED fix: raise the floor to 100m, but absorb the extra thickness into
    # the BASE only, leaving the OBSERVED surface (hence grad(surface), hence driving
    # stress) untouched. An earlier attempt that instead rebuilt surface as bed+H moved the
    # surface by up to +90m and fabricated driving stress (worst case RMSE 269.6, Transantarctic
    # Mountains). Also re-floor N against the NEW 100m thickness: N was already floored once in
    # param.py, but against the stale 10m value -- left unrefloored, thin-ice vertices carry
    # ~10x too little basal drag for their now-larger overburden, with no C value able to
    # compensate (this was the exact mechanism behind a 14.56x-too-fast runaway group found
    # earlier this project).
    _ri = md.materials.rho_ice; _rw = md.materials.rho_water
    _H = np.asarray(md.geometry.thickness).ravel().copy()
    _ol = np.asarray(md.mask.ocean_levelset).ravel()
    _surf0 = np.asarray(md.geometry.surface).ravel().copy()
    _nfl = int((_H < 100).sum())
    _H = np.maximum(_H, 100.0)
    _flt = _ol < 0
    _base = np.empty_like(_H); _surf = np.empty_like(_H)
    _base[_flt] = -_H[_flt] * _ri / _rw; _surf[_flt] = _H[_flt] * (1.0 - _ri / _rw)
    _surf[~_flt] = _surf0[~_flt]
    _base[~_flt] = _surf0[~_flt] - _H[~_flt]
    md.geometry.thickness = _H; md.geometry.base = _base; md.geometry.surface = _surf
    print(f"   {_nfl} verts floored to 100m; max surface change on grounded ice = "
          f"{np.nanmax(np.abs(_surf[~_flt] - _surf0[~_flt])):.3g} m (should be 0)")

    print(f"-- Re-flooring effective pressure against the updated (100m-floored) thickness...")
    _lim = 0.07
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    _Nfloor = _lim * md.materials.rho_ice * md.constants.g * _H
    _nbad = int(np.sum(N < _Nfloor * 0.999))
    N = np.maximum(N, _Nfloor)
    print(f"   {_nbad} verts raised to the updated N floor")
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = _lim

    print(f"-- Define general control parameters...")
    md.inversion.iscontrol = 1
    md.verbose.solution = 0
    md.verbose.qmu = 0
    md.verbose.control = 1

    print(f"-- Extracting friction-inversion domain (floating/grounded ice-front BC)...")
    mds = extract_friction_inversion_domain(md)

    print(f"-- Overriding friction.coupling = {friction_coupling}...")
    mds.friction.coupling = friction_coupling

    print(f"-- Re-flooring N on the extracted domain (sanity check)...")
    _Hs = np.asarray(mds.geometry.thickness).ravel()
    _Ns = mds.friction.effective_pressure.copy(); _Ns[_Ns < 0] = 0
    _Nfs = 0.07 * mds.materials.rho_ice * mds.constants.g * _Hs
    mds.friction.effective_pressure = np.maximum(_Ns, _Nfs)
    mds.friction.effective_pressure_limit = 0.07
    _gr_check = (np.asarray(mds.mask.ice_levelset).ravel() < 0) & (np.asarray(mds.mask.ocean_levelset).ravel() > 0)
    _frac = mds.friction.effective_pressure / np.maximum(mds.materials.rho_ice * mds.constants.g * _Hs, 1e-9)
    assert np.nanmin(_frac[_gr_check]) >= 0.0699, "N still below floor after re-flooring"

    print(f"-- Defining inversion parameters...")
    mds.inversion = pyissm.model.classes.inversion.m1qn3(mds.inversion)
    fric_control, fric_field, fric_min, fric_max = friction_law_info(mds)  # Schoof or Budd
    mds.inversion.control_parameters = [fric_control]
    mds.inversion.min_parameters = np.full(mds.mesh.numberofvertices, fric_min)
    mds.inversion.max_parameters = np.full(mds.mesh.numberofvertices, fric_max)  # bounds fixed above (friction_law_info)
    # See the updated note in ssa_friction_inv_sensit: the "keep budget small" premise was
    # measured against the C_init=10 dead-zone bug and the OOM history is addressed by the
    # cluster.np/memory fix above, not by capping maxsteps. VALIDATED: 500/500 (p=q=1 converges
    # to grounded RMSE 61.4 at 198 iterations, on dxmin -- see friction_inv_gttol below for why
    # that's the criterion that actually has to fire).
    mds.inversion.maxsteps = 500
    mds.inversion.maxiter = 500
    # See friction_inv_gttol definition above: without this, m1qn3's default gttol=1e-4 stops
    # every p=q=1 run by iteration ~16 on a false "converged" signal, before the fit has done
    # anything -- this is the single fix that made p=q=1 (and its RMSE win over p=q=3) visible
    # at all; every earlier p=1 attempt silently never tested it.
    mds.inversion.gttol = friction_inv_gttol

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

    print(f"-- Setting-up coefficient grid (single unregularised baseline run)...")
    # This step is now ONLY the unregularised p=q=1 baseline (cf501 negligible, effectively
    # off) -- its sole purpose is to be the warm-start source for ssa_friction_inv_reg_lcurve
    # below, which does the actual cf501 (DragCoefficientAbsGradient) L-curve sweep. Folding a
    # 501 grid directly into this step (as the old p=q=3-era code did) doesn't reproduce the
    # validated methodology: each grid point there cold-starts from C_init=1.8 independently,
    # never warm-started from a converged state, which is a different (and unvalidated)
    # experiment. See ssa_friction_inv_reg_lcurve for the real sweep and the p=q=3 -> p=q=1
    # 501-scale note (C's units/magnitude differ completely between the two laws, so the old
    # p=3-tuned 1e-3..3e-1 range is meaningless here -- validated p=1 range is ~1e-4..1e-2).
    param_grid = pyissm.inversion.sensitivity.build_parameter_grid(
        {101: [friction_cf101],
         103: [friction_cf103],
         501: [1e-8]})

    # VALIDATED FIX: restrict 101/103 to grounded observed ice (see the matching fix and
    # rationale in ssa_friction_inv_sensit above).
    print(f"-- Defining mask to exclude 0 velocity and restrict to grounded ice...")
    mask = (mds.inversion.vel_obs > 0) & (mds.mask.ocean_levelset >= 0)

    # VALIDATED FIX: exclude grounding-line-adjacent elements from the 501 regularisation mask.
    # C steps from its grounded value to ~0 across the grounding line (floating C is pinned to
    # a near-zero bound), and coeff_masks are per-vertex weights ISSM integrates element-wise,
    # so a plain grounded mask still weights elements straddling that discontinuity -- and since
    # floating C cannot move, that part of the penalty is irreducible. See section 3.1.
    grounded_mask = np.asarray(mds.mask.ocean_levelset) >= 0
    _elx = np.asarray(mds.mesh.elements).astype(int) - 1
    _float_v = np.asarray(mds.mask.ocean_levelset).ravel() < 0
    _touch_f = _float_v[_elx].any(axis = 1)
    _gladj = np.zeros(mds.mesh.numberofvertices, dtype = bool)
    _gladj[_elx[_touch_f].ravel()] = True
    reg_mask = grounded_mask & ~_gladj

    if save:
        # Single-point baseline, not a sweep -- no L-curve to plot here (see
        # ssa_friction_inv_reg_lcurve for the actual cf501 L-curve). Just load and report
        # grounded RMSE against the validated reference (61.4).
        print(f"-- Loading baseline friction inversion result...")
        mdb = pyissm.model.io.load_model(
            f'{model_dir}/AIS3_ssa_friction_inv_lcurve/{friction_baseline_run}/{friction_baseline_run}.nc')
        vel = np.asarray(mdb.results.StressbalanceSolution.Vel).ravel()
        vo = np.asarray(mdb.inversion.vel_obs).ravel()
        gr = (np.asarray(mdb.mask.ice_levelset).ravel() < 0) & (np.asarray(mdb.mask.ocean_levelset).ravel() > 0)
        rmse = np.sqrt(np.nanmean((vel[gr] - vo[gr]) ** 2))
        print(f"   grounded RMSE = {rmse:.1f} m/yr (validated reference: 61.4)")

    else:
        print(f"-- Running unregularised p=q=1 baseline inversion...")
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_friction_inv_lcurve',
            run = True,
            load_only = False,
            coeff_masks = {101: mask,
                           103: mask,
                           501: reg_mask})


## ------------------------------------
## Friction regularisation L-curve, warm-started from the unregularised baseline
## ------------------------------------
if 'ssa_friction_inv_reg_lcurve' in steps:

    print("-------------------------------------------------------------")
    print(f" SSA FRICTION INVERSION L-CURVE - cf501 REGULARISATION (p=q=1)")
    print("-------------------------------------------------------------")

    print(f"-- Loading parameterized model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_param.nc')

    print(f"-- Loading SSA floating rheology inversion results...")
    mds = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_rheology_floating_inv_lcurve/{rheology_lcurve_run}/{rheology_lcurve_run}.nc')

    print(f"-- Updating rheology field from inversion results...")
    md.materials.rheology_B[mds.mesh.extractedvertices - 1] = mds.results.StressbalanceSolution.MaterialsRheologyBbar

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print(f"-- Flooring thin ice at 100m (numerical stability), preserving observed surface...")
    # Same fix as ssa_friction_inv_lcurve above -- must be applied identically here so this
    # step's geometry matches what friction_baseline_run was actually solved against; a
    # mismatch would mean the warm-started C field below gets paired with different driving
    # stress than it was tuned for. See ssa_friction_inv_lcurve for the full rationale.
    _ri = md.materials.rho_ice; _rw = md.materials.rho_water
    _H = np.asarray(md.geometry.thickness).ravel().copy()
    _ol = np.asarray(md.mask.ocean_levelset).ravel()
    _surf0 = np.asarray(md.geometry.surface).ravel().copy()
    _H = np.maximum(_H, 100.0)
    _flt = _ol < 0
    _base = np.empty_like(_H); _surf = np.empty_like(_H)
    _base[_flt] = -_H[_flt] * _ri / _rw; _surf[_flt] = _H[_flt] * (1.0 - _ri / _rw)
    _surf[~_flt] = _surf0[~_flt]
    _base[~_flt] = _surf0[~_flt] - _H[~_flt]
    md.geometry.thickness = _H; md.geometry.base = _base; md.geometry.surface = _surf

    _lim = 0.07
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    _Nfloor = _lim * md.materials.rho_ice * md.constants.g * _H
    N = np.maximum(N, _Nfloor)
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = _lim

    print(f"-- Extracting friction-inversion domain (floating/grounded ice-front BC)...")
    mds = extract_friction_inversion_domain(md)

    print(f"-- Overriding friction.coupling = {friction_coupling}...")
    mds.friction.coupling = friction_coupling

    print(f"-- Re-flooring N on the extracted domain (sanity check)...")
    _Hs = np.asarray(mds.geometry.thickness).ravel()
    _Ns = mds.friction.effective_pressure.copy(); _Ns[_Ns < 0] = 0
    _Nfs = 0.07 * mds.materials.rho_ice * mds.constants.g * _Hs
    mds.friction.effective_pressure = np.maximum(_Ns, _Nfs)
    mds.friction.effective_pressure_limit = 0.07

    print(f"-- Warm-starting C from the unregularised baseline ({friction_baseline_run})...")
    # Warm-starting (rather than cold-starting each 501 grid point from C_init=1.8, as the
    # p=q=3-era code did) is the validated methodology: the regularisation sweep only needs
    # to locally smooth an already-converged C field, not re-solve the whole continent per
    # grid point. This also matches how every successful 501 sweep in this project has
    # actually been run.
    mdb = pyissm.model.io.load_model(
        f'{model_dir}/AIS3_ssa_friction_inv_lcurve/{friction_baseline_run}/{friction_baseline_run}.nc')
    C_baseline = np.asarray(mdb.results.StressbalanceSolution.FrictionCoefficient).ravel()

    print(f"-- Defining inversion parameters...")
    mds.inversion = pyissm.model.classes.inversion.m1qn3(mds.inversion)
    fric_control, fric_field, fric_min, fric_max = friction_law_info(mds)
    mds.inversion.control_parameters = [fric_control]
    mds.inversion.min_parameters = np.full(mds.mesh.numberofvertices, fric_min)
    mds.inversion.max_parameters = np.full(mds.mesh.numberofvertices, fric_max)
    mds.inversion.maxsteps = 500
    mds.inversion.maxiter = 500
    mds.inversion.gttol = friction_inv_gttol

    setattr(mds.friction, fric_field, C_baseline.copy())

    print(f"-- Assigning cluster and updating settings...")
    mds.cluster = cluster
    mds.settings.waitonlock = 0

    # No friction on PURELY floating ice elements
    ocean_elements = mds.mask.ocean_levelset[mds.mesh.elements - 1]
    pos_e = np.where(np.min(ocean_elements, axis=1) < 0)[0]
    flags = np.zeros(mds.mesh.numberofvertices, dtype=bool)
    flags[mds.mesh.elements[pos_e, :] - 1] = True
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
    # VALIDATED range for p=q=1 (found by direct sweep warm-started from the RMSE-61.4
    # baseline): cf501=0.0001 drops C-field roughness from 0.82 to 0.18 (matching the p=q=3
    # baseline's own 0.17) while RMSE *improves* to 60.4 -- both axes better, not a tradeoff.
    # RMSE degrades fast past that point (70.5 at 0.0003, 307.8 by 0.016), so 0.0001 is the
    # chosen corner (`friction_lcurve_run` above), not just a swept value. This range is NOT
    # comparable to the old p=q=3 1e-3..3e-1 grid -- C's units/magnitude differ completely
    # between the two laws (u ~ C^-6 vs u ~ C^-2), so the two sweeps can't share a scale.
    param_grid = pyissm.inversion.sensitivity.build_parameter_grid(
        {101: [friction_cf101],
         103: [friction_cf103],
         501: [0.0001, 0.0003, 0.001, 0.002, 0.0032]})

    print(f"-- Defining mask to exclude 0 velocity and restrict to grounded ice...")
    mask = (mds.inversion.vel_obs > 0) & (mds.mask.ocean_levelset >= 0)

    # Exclude grounding-line-adjacent elements from the 501 regularisation mask -- see the
    # matching note in ssa_friction_inv_lcurve / docs/inversion_worklog.md section 3.1.
    grounded_mask = np.asarray(mds.mask.ocean_levelset) >= 0
    _elx = np.asarray(mds.mesh.elements).astype(int) - 1
    _float_v = np.asarray(mds.mask.ocean_levelset).ravel() < 0
    _touch_f = _float_v[_elx].any(axis = 1)
    _gladj = np.zeros(mds.mesh.numberofvertices, dtype = bool)
    _gladj[_elx[_touch_f].ravel()] = True
    reg_mask = grounded_mask & ~_gladj

    if save:
        print(f"-- Loading inversion parameter sensitivity...")
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_friction_inv_reg_lcurve',
            run = False,
            load_only = True,
            coeff_masks = {101: mask,
                           103: mask,
                           501: reg_mask})

        print(f"-- Processing inversion parameter sensitivity...")
        diagnostics = pyissm.inversion.sensitivity.compute_sensitivity_diagnostics(manifest, output_dir=f'{model_dir}/AIS3_ssa_friction_inv_reg_lcurve/')

        fig, ax = pyissm.inversion.plot.plot_lcurve(diagnostics)
        ax.set_title('Grounded ice friction inversion - cf501 L-curve (p=q=1, warm-started)')
        plt.savefig(f'{model_dir}/AIS3_ssa_friction_inv_reg_lcurve/lcurve.png')

    else:
        print(f"-- Running warm-started cf501 regularisation sweep...")
        manifest = pyissm.inversion.sensitivity.parameter_sensitivity(
            mds,
            param_grid,
            output_dir = f'{model_dir}/AIS3_ssa_friction_inv_reg_lcurve',
            run = True,
            load_only = False,
            coeff_masks = {101: mask,
                           103: mask,
                           501: reg_mask})


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
    mdf = pyissm.model.io.load_model(f'{model_dir}/AIS3_ssa_friction_inv_reg_lcurve/{friction_lcurve_run}/{friction_lcurve_run}.nc')
    fric_control, fric_field, _, _ = friction_law_info(md)  # Schoof: FrictionC / C ; Budd: FrictionCoefficient / coefficient
    _fld = getattr(md.friction, fric_field).astype(float)
    _fld[mdf.mesh.extractedvertices - 1] = getattr(mdf.results.StressbalanceSolution, fric_control) # -1 for zero-based indexing
    setattr(md.friction, fric_field, _fld)

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print(f"-- Flooring thin ice at 100m (numerical stability), preserving observed surface...")
    # Must match the geometry ssa_friction_inv_lcurve/ssa_friction_inv_reg_lcurve actually
    # solved against -- otherwise this assembled forward solve pairs the inverted friction
    # field with different driving stress than it was tuned for. See ssa_friction_inv_lcurve
    # for the full rationale.
    _ri = md.materials.rho_ice; _rw = md.materials.rho_water
    _H = np.asarray(md.geometry.thickness).ravel().copy()
    _ol = np.asarray(md.mask.ocean_levelset).ravel()
    _surf0 = np.asarray(md.geometry.surface).ravel().copy()
    _H = np.maximum(_H, 100.0)
    _flt = _ol < 0
    _base = np.empty_like(_H); _surf = np.empty_like(_H)
    _base[_flt] = -_H[_flt] * _ri / _rw; _surf[_flt] = _H[_flt] * (1.0 - _ri / _rw)
    _surf[~_flt] = _surf0[~_flt]
    _base[~_flt] = _surf0[~_flt] - _H[~_flt]
    md.geometry.thickness = _H; md.geometry.base = _base; md.geometry.surface = _surf

    # Fix negative effective pressure (consistent with the inversion setup)
    _lim = 0.07
    N = md.friction.effective_pressure.copy()
    N[N < 0] = 0
    _Nfloor = _lim * md.materials.rho_ice * md.constants.g * _H
    N = np.maximum(N, _Nfloor)
    md.friction.effective_pressure = N
    md.friction.effective_pressure_limit = _lim

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


## ==================================================================================
## POST-INVERSION EXTENSIONS (new, 2026-08) -- see docs/inversion_worklog.md and
## /home/120/jh7060/.claude/plans/cuddly-finding-creek.md for the full investigation.
##
## Standard practice after a friction inversion: higher-order thermal spin-up (SSA has
## no vertical shear physics, so temperature/rheology can't be solved self-consistently
## -- ais_0.1_param.py's surface-temperature-as-proxy approach is a known, flagged
## caveat), a friction re-inversion under that higher-order physics (SSA-tuned friction
## isn't valid once vertical shear resistance is added), ocean melt-rate calibration, a
## short post-inversion relaxation, and a historical run tuned against observed dH/dt.
##
## Stage 1 (ho_thermal_steadystate) is implemented and ready to test. Stages 2-5 are
## scaffolds: correct model loading / solver setup / save-submit structure, each with an
## explicit TODO marking the science decision that still needs iteration. Do not treat
## their output as validated the way AIS3_inverted.nc / AIS3_relaxed.nc are once stage 1
## has actually been run and checked.
## ==================================================================================

## ------------------------------------
## Stage 1: Higher-order (MOLHO) thermal steady state
## ------------------------------------
# Couples stress balance + thermal (+ melting + enthalpy) to convergence on an extruded
# mesh, replacing ais_0.1_param.py's surface-temperature-as-proxy rheology_B with a real
# depth-resolved one. MOLHO chosen over full HO/FS: it's the standard cost-effective
# higher-order approximation for this purpose and (unlike HO/FS) doesn't require the mesh
# to be pre-extruded for the stress-balance part itself -- pyissm/model/param.py's 2D-mesh
# guard only fires for HO/FS. The thermal/enthalpy part is inherently 3D regardless, so
# the mesh is extruded here either way.
if 'ho_thermal_steadystate' in steps:

    print("-------------------------------------------------------------")
    print(f" HIGHER-ORDER (MOLHO) THERMAL STEADY STATE"                   )
    print("-------------------------------------------------------------")

    print(f"-- Loading inverted model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_inverted.nc')

    print(f"-- Extruding mesh (15 layers)...")
    # TODO (verify at implementation/test time, see plan verification section): confirm
    # MOLHO solves cleanly on an extruded mesh -- the param.py 2D-mesh guard only blocks
    # HO/FS, not MOLHO, meaning MOLHO is DESIGNED to skip extrusion for stress-balance-only
    # solves. Whether it behaves correctly when thermal coupling forces extrusion anyway is
    # the one real unknown in this stage; test on a small region (e.g. the PIG/Thwaites or
    # Siple Coast bounding boxes already used for diagnostics this session) before running
    # continent-wide.
    md = md.extrude(num_layers = 15, extrusion_exponent = 1.3)

    print(f"-- Setting flow equation to MOLHO...")
    md = pyissm.model.param.set_flow_equation(md, MOLHO = 'all')
    md = pyissm.model.bc.set_molho_bc(md)

    print(f"-- Configuring thermal solve...")
    md.thermal.isenthalpy = 1  # temperate-ice handling near the bed, standard at this scale
    md.thermal.maxiter = 100
    md.thermal.reltol = 0.01

    print('-- Removing icebergs from ice levelset...')
    md.mask.ice_levelset = pyissm.model.param.kill_icebergs(md)

    print(f"-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_thermal_steadystate'
    md.cluster = cluster
    md.settings.waitonlock = 0

    if save:
        print(f"-- Loading steady-state solution...")
        md = pyissm.model.execute.solve(md, 'SteadystateSolution', load_only = True, runtime_name = False)

        if diagnostics:
            T = md.results.SteadystateSolution.Temperature
            print(f"\nTHERMAL STEADY-STATE DIAGNOSTICS:")
            print(f"   Min temperature: {np.nanmin(T):.2f} K")
            print(f"   Max temperature: {np.nanmax(T):.2f} K")
            # Sanity range: should stay within [220, 273.15] K -- outside that indicates a
            # non-physical result (too cold: bad BC/units; at/above 273.15 K widely: the
            # enthalpy formulation should be handling temperate ice, not raw temperature
            # exceeding the pressure-melting point).

        print(f"-- Recomputing rheology_B from the depth-resolved temperature...")
        md.materials.rheology_B = pyissm.tools.materials.cuffey(md.initialization.temperature)

        print(f"\nSaving thermal steady-state model to {model_dir}/AIS3_thermal_steadystate.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_thermal_steadystate.nc')

    else:
        print(f"-- Submitting steady-state solve...")
        md = pyissm.model.execute.solve(md, 'SteadystateSolution', load_only = False, runtime_name = False)


## ------------------------------------
## Stage 2 (SCAFFOLD): Higher-order friction re-inversion
## ------------------------------------
# SSA-tuned friction (from ssa_friction_inv_reg_lcurve) isn't physically valid once MOLHO
# adds vertical-shear resistance to the force balance -- the friction coefficient has to
# absorb a different share of the driving stress. Warm-start from the SSA-inverted C
# (same warm-start pattern validated for every mesh/geometry change this session) rather
# than cold-starting from C_init, and reuse the existing sensitivity-sweep infrastructure,
# which is confirmed to just call a generic Stressbalance solve (respects whatever flow
# equation is already set on md, not SSA-hardcoded) -- but this combination (MOLHO +
# m1qn3 inversion via parameter_sensitivity) has not been tested. Verify on the same small
# test region as stage 1 before trusting continent-wide.
if 'ho_friction_inv' in steps:

    print("-------------------------------------------------------------")
    print(f" HIGHER-ORDER (MOLHO) FRICTION RE-INVERSION"                  )
    print("-------------------------------------------------------------")

    print(f"-- Loading thermal steady-state model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_thermal_steadystate.nc')

    # Warm start is ALREADY carried through, not a separate step: AIS3_inverted.nc has the
    # correctly SSA-solved FrictionCoefficient, ho_thermal_steadystate extrudes it, and
    # Model.extrude() projects every existing 2D field (including friction, via
    # md.friction._extrude(md), confirmed in pyISSM/model/Model.py) onto the new 3D mesh
    # automatically -- so md.friction.coefficient here is already the warm-started field,
    # just replicated up every vertical column rather than defined only at the base.
    fric_control, fric_field, fric_min, fric_max = friction_law_info(md)

    print(f"-- Restricting friction control to base-layer vertices...")
    # Friction is a basal boundary condition -- only vertexonbase vertices are physically
    # meaningful control points. Pin every non-base vertex's bounds to its current
    # (replicated) value so m1qn3 doesn't spend gradient steps on physically meaningless
    # upper-layer copies of the same nominal field. vertexonbase / numberofvertices2d are
    # both set by Model.extrude() (pyISSM/model/Model.py:819,832).
    nv3d = md.mesh.numberofvertices
    on_base = np.asarray(md.mesh.vertexonbase).astype(bool).ravel()
    print(f"   {int(on_base.sum())} of {nv3d} vertices are on the base layer "
          f"(expect == numberofvertices2d = {md.mesh.numberofvertices2d})")

    _fld = getattr(md.friction, fric_field).astype(float)
    md.inversion = pyissm.model.classes.inversion.m1qn3(md.inversion)
    md.inversion.iscontrol = 1
    md.inversion.control_parameters = [fric_control]
    md.inversion.min_parameters = np.where(on_base, fric_min, _fld)
    md.inversion.max_parameters = np.where(on_base, fric_max, _fld)
    md.inversion.maxsteps = 500
    md.inversion.maxiter = 500
    md.inversion.gttol = friction_inv_gttol

    print(f"-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_ho_friction_inv'
    md.cluster = cluster
    md.settings.waitonlock = 0

    print(f"-- Setting-up cost function coefficients/masks...")
    # Start from the validated SSA values (cf101=10/cf103=100, grounded+observed mask,
    # grounding-line-excluded 501 mask) -- same physical misfit/regularisation this project
    # has used throughout, just re-evaluated under MOLHO's own velocity field. cf501 is kept
    # at the SSA-chosen 0.0001 as a starting point; MOLHO's different force balance could
    # shift the misfit-vs-regularisation tradeoff, so treat this as a prior, not a final
    # value -- worth a small re-sweep (mirroring ssa_friction_inv_reg_lcurve's methodology)
    # once this converges once and a baseline RMSE is in hand.
    vo = np.asarray(md.inversion.vel_obs).ravel()
    ol = np.asarray(md.mask.ocean_levelset).ravel()
    il = np.asarray(md.mask.ice_levelset).ravel()
    mask = (vo > 0) & (ol >= 0) & on_base
    grounded_mask = (ol >= 0) & on_base
    _elx = np.asarray(md.mesh.elements).astype(int) - 1
    _float_v = ol < 0
    _touch_f = _float_v[_elx].any(axis = 1)
    _gladj = np.zeros(nv3d, dtype = bool)
    _gladj[_elx[_touch_f].ravel()] = True
    reg_mask = grounded_mask & ~_gladj

    cf = np.zeros((nv3d, 3))
    cf[mask, 0] = friction_cf101
    cf[mask, 1] = friction_cf103
    cf[reg_mask, 2] = 0.0001
    md.inversion.cost_functions = [101, 103, 501]
    md.inversion.cost_functions_coefficients = cf

    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.stressbalance.restol = 0.01
    md.stressbalance.reltol = 0.1
    md.stressbalance.abstol = np.nan
    md.settings.solver_residue_threshold = 1e-3

    if save:
        print(f"-- Loading MOLHO friction inversion result...")
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only = True, runtime_name = False)

        if diagnostics:
            vel = md.results.StressbalanceSolution.Vel
            gr = (il < 0) & (ol > 0) & on_base
            print(f"\nHO FRICTION INVERSION DIAGNOSTICS:")
            print(f"   Grounded (base-layer) RMSE: {np.sqrt(np.nanmean((vel[gr]-vo[gr])**2)):.2f} m/yr")

        print(f"\nSaving to {model_dir}/AIS3_ho_friction_inv.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_ho_friction_inv.nc')
    else:
        print(f"-- Submitting MOLHO friction inversion...")
        md = pyissm.model.execute.solve(md, 'Stressbalance', load_only = False, runtime_name = False)


## ------------------------------------
## Stage 3 (SCAFFOLD): Ocean basal-melt gamma calibration
## ------------------------------------
# Data plan (see plan doc for the full survey): ocean thermal forcing from the Zhou et
# al. observational climatology (real observed conditions, not a CMIP scenario --
# /g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/zhou_annual_06_nov/
# tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc, tf(z,y,x), 8km
# horizontal / 60m vertical, 30 levels -30 to -1770m, static). gamma_0 seeded from the
# published ISMIP6 coefficient
# (/g/data/au88/ismip6/2300/forcings/parameterizations/coeff_gamma0_DeltaT_quadratic_local_median.nc,
# gamma0=11075.45 m/yr, deltaT_basin on the same 8km grid, 16 distinct basin values) as an
# informed prior, then fine-tuned against ccdtools's
# measures_its_live_antarctic_quarterly_ice_shelf_height_change 'melt' field (1992-2017
# observed basal melt rates).
if 'melt_gamma_tuning' in steps:

    print("-------------------------------------------------------------")
    print(f" OCEAN BASAL-MELT GAMMA CALIBRATION"                          )
    print("-------------------------------------------------------------")

    from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

    ZHOU_TF_FILE = ('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/ocean/climatology/'
                     'zhou_annual_06_nov/tf/v3/tf_AIS_obs_ocean_climatology_zhou_annual_06_nov_v3_1972-2024.nc')
    GAMMA0_FILE = ('/g/data/au88/ismip6/2300/forcings/parameterizations/'
                    'coeff_gamma0_DeltaT_quadratic_local_median.nc')
    TIME_SENTINEL = 1e9  # ISSM timeseries convention for a constant (non-time-varying) field

    print(f"-- Loading MOLHO friction-inverted model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_ho_friction_inv.nc')

    print(f"-- Configuring ISMIP6 basal melt parameterisation...")
    md.basalforcings = pyissm.model.classes.basalforcings.ismip6(md.basalforcings)

    print(f"-- Loading published gamma0/deltaT_basin (informed prior, {GAMMA0_FILE})...")
    ds_gamma = xr.open_dataset(GAMMA0_FILE)
    gamma0_prior = float(ds_gamma['gamma0'].values)
    deltaT_grid = ds_gamma['deltaT_basin'].values      # (y, x), 16 distinct values
    gx, gy = ds_gamma['x'].values, ds_gamma['y'].values
    print(f"   gamma0 prior = {gamma0_prior:.2f} m/yr")

    print(f"-- Deriving basin_id from deltaT_basin (avoids the Mouginot-vs-Rignot basin-set")
    print(f"   compatibility question entirely: basin_id and delta_t both come from the SAME")
    print(f"   file/grid the published gamma0 was calibrated against, self-consistent by")
    print(f"   construction, rather than trying to remap MIPKIT's mouginot_basins onto it)...")
    basin_vals = np.unique(deltaT_grid[np.isfinite(deltaT_grid)])
    num_basins = int(basin_vals.size)
    print(f"   {num_basins} basins found (expect 16)")
    basin_id_grid = np.searchsorted(basin_vals, deltaT_grid) + 1  # 1-indexed basin IDs
    delta_t_per_basin = basin_vals.copy()   # delta_t[k] corresponds to basin k+1

    # basin_id is per-ELEMENT (pyISSM basalforcings.ismip6 docstring); nearest-neighbour
    # lookup on element centroids against the 8km ocean grid.
    elx = np.asarray(md.mesh.elements2d if hasattr(md.mesh, 'elements2d') else md.mesh.elements).astype(int) - 1
    vx2d = np.asarray(md.mesh.x2d if hasattr(md.mesh, 'x2d') else md.mesh.x).ravel()
    vy2d = np.asarray(md.mesh.y2d if hasattr(md.mesh, 'y2d') else md.mesh.y).ravel()
    # NOTE: elx may reference the 3D element list (mds.mesh.elements) if elements2d isn't
    # populated the way expected -- verify at run time; the 2D-mesh-only fields
    # (x2d/y2d/elements2d) are documented as preserved by Model.extrude() but exact naming
    # should be double-checked against this pyissm version's actual Model.extrude() output.
    if elx.shape[1] >= 3:
        ecx = vx2d[elx[:, :3]].mean(axis=1)
        ecy = vy2d[elx[:, :3]].mean(axis=1)
    gxx, gyy = np.meshgrid(gx, gy)
    basin_lookup = NearestNDInterpolator(np.column_stack([gxx.ravel(), gyy.ravel()]), basin_id_grid.ravel())
    md.basalforcings.basin_id = basin_lookup(np.column_stack([ecx, ecy])).astype(float)
    md.basalforcings.num_basins = num_basins
    md.basalforcings.delta_t = delta_t_per_basin
    md.basalforcings.islocal = 1  # local quadratic parameterisation, matching the gamma0 prior source

    print(f"-- Loading Zhou ocean thermal-forcing climatology and interpolating onto the mesh...")
    ds_tf = xr.open_dataset(ZHOU_TF_FILE)
    tfx, tfy, tfz = ds_tf['x'].values, ds_tf['y'].values, ds_tf['z'].values
    tf_full = ds_tf['tf'].values  # (z, y, x)
    nv = md.mesh.numberofvertices
    mesh_x = np.asarray(md.mesh.x).ravel()
    mesh_y = np.asarray(md.mesh.y).ravel()
    tf_list = []
    for k in range(tfz.size):
        interp_k = RegularGridInterpolator((tfy, tfx), tf_full[k], method='linear',
                                            bounds_error=False, fill_value=np.nan)
        vals = interp_k(np.column_stack([mesh_y, mesh_x]))
        vals = np.nan_to_num(vals, nan=0.0)  # 0 degC TF outside the ocean grid (grounded interior)
        col = np.append(vals, TIME_SENTINEL)
        tf_list.append(col.reshape(-1, 1))
    md.basalforcings.tf = tf_list
    md.basalforcings.tf_depths = tfz.copy()
    print(f"   {tfz.size} depth layers ({tfz.min():.0f} to {tfz.max():.0f} m), interpolated onto {nv} mesh vertices")

    print(f"-- Loading ITS_LIVE observed ice-shelf melt rate (calibration target)...")
    catalog = ccdtools.catalog.DataCatalog()
    melt_obs_ds = catalog.load_dataset('measures_its_live_antarctic_quarterly_ice_shelf_height_change')
    # NOTE: variable name confirmed present as 'melt' by the earlier data survey (see plan
    # doc) but its exact spelling/units in THIS specific loaded object should be checked
    # against melt_obs_ds.data_vars at run time before trusting the interpolation below.
    melt_obs_grid = melt_obs_ds['melt']
    melt_obs_on_mesh = pyissm.data.interp.xr_to_mesh(melt_obs_ds, 'melt', md.mesh.x, md.mesh.y)

    print(f"-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_melt_gamma_tuning'
    md.cluster = cluster
    md.settings.waitonlock = 0

    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.ismasstransport = 1   # melt only enters the solve as a masstransport BC flux
    md.timestepping.start_time = 0
    md.timestepping.final_time = 0.01  # yr -- deliberately tiny: evaluate melt, don't evolve geometry
    md.timestepping.time_step = 0.01
    md.transient.requested_outputs = ['default', 'BasalforcingsFloatingiceMeltingRate']

    print(f"-- Setting-up gamma_0 sweep grid (published prior +/- a validation range)...")
    # build_parameter_grid is generic (just a dict->DataFrame cartesian-product builder, see
    # pyissm/inversion/sensitivity.py) and reusable here even though the rest of that module
    # (assign_cost_functions/parameter_sensitivity) is hardcoded to inversion cost functions
    # and a Stressbalance solve -- neither applies to a scalar basalforcings field, so the
    # actual submit/compare loop below is hand-written rather than reusing those.
    gamma_grid = pyissm.inversion.sensitivity.build_parameter_grid(
        {0: [gamma0_prior * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)]})

    if save:
        print(f"-- Loading melt calibration sweep results and comparing to observations...")
        best = None
        for _, row in gamma_grid.iterrows():
            gname = f"AIS3_melt_gamma_tuning_g{row['run_id']}"
            mdi = pyissm.model.io.load_model(f'{model_dir}/AIS3_ho_friction_inv.nc')  # cheap reload for field shapes
            mdi.miscellaneous.name = gname
            mdi.cluster = cluster
            try:
                mdi = pyissm.model.execute.solve(mdi, 'Transient', load_only = True, runtime_name = False)
            except Exception as e:
                print(f"   run {row['run_id']} (gamma_0={row['cf0']:.1f}): FAILED to load ({e})")
                continue
            melt_sim = np.asarray(mdi.results.TransientSolution[-1].BasalforcingsFloatingiceMeltingRate).ravel()
            floating = np.asarray(mdi.mask.ocean_levelset).ravel() < 0
            rmse = np.sqrt(np.nanmean((melt_sim[floating] - melt_obs_on_mesh[floating]) ** 2))
            print(f"   gamma_0={row['cf0']:.1f}: melt RMSE vs ITS_LIVE = {rmse:.2f} m/yr")
            if best is None or rmse < best[1]:
                best = (row['cf0'], rmse, mdi)

        if best is not None:
            print(f"\nBest gamma_0 = {best[0]:.2f} m/yr (melt RMSE = {best[1]:.2f} m/yr)")
            md = best[2]
            md.basalforcings.gamma_0 = best[0]
            print(f"Saving to {model_dir}/AIS3_melt_gamma_tuning.nc")
            pyissm.model.io.save_model(md, f'{model_dir}/AIS3_melt_gamma_tuning.nc')
        else:
            print(f"   No sweep runs loaded successfully -- nothing to save.")

    else:
        print(f"-- Submitting gamma_0 sweep ({len(gamma_grid)} runs)...")
        for _, row in gamma_grid.iterrows():
            mdi = md.extract(np.ones(md.mesh.numberofvertices, dtype=bool))  # cheap full copy per run
            mdi.basalforcings.gamma_0 = float(row['cf0'])
            mdi.miscellaneous.name = f"AIS3_melt_gamma_tuning_g{row['run_id']}"
            mdi.cluster = cluster
            print(f"   run {row['run_id']}: gamma_0={row['cf0']:.1f}")
            pyissm.model.execute.solve(mdi, 'Transient', load_only = False, runtime_name = False)


## ------------------------------------
## Stage 4 (SCAFFOLD): Post-calibration relaxation (~1 year)
## ------------------------------------
# Same transient structure as ssa_relaxation above (deactivate_all then re-enable
# isstressbalance/ismasstransport/issmb/isgroundingline), but loading the stage 2/3
# output and with final_time set to ~1 year -- per the user's stated intent, this is
# meant only to damp diagnostic-to-prognostic shock after the MOLHO re-inversion +
# melt-parameter change, NOT to do the historical spin-up itself (that's stage 5). Kept
# as a separate step from ssa_relaxation rather than overwriting it, so the working SSA
# baseline (AIS3_relaxed.nc) stays available for comparison.
if 'ho_relaxation' in steps:

    print("-------------------------------------------------------------")
    print(f" POST-CALIBRATION RELAXATION (~1 YEAR)"                       )
    print("-------------------------------------------------------------")

    print(f"-- Loading melt-calibrated model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_melt_gamma_tuning.nc')

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
    md.timestepping.final_time = 1     # years -- short shock-damping only, see note above
    md.timestepping.time_step  = 0.02  # years

    print(f"-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_ho_relaxed'
    md.cluster = cluster
    md.settings.waitonlock = 0

    if save:
        print(f"-- Loading transient solution...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only = True, runtime_name = False)
        print(f"\nSaving to {model_dir}/AIS3_ho_relaxed.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_ho_relaxed.nc')
    else:
        print(f"-- Submitting post-calibration relaxation...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only = False, runtime_name = False)


## ------------------------------------
## Stage 5 (SCAFFOLD): Historical run tuned against observed dH/dt
## ------------------------------------
# Data caveat (state plainly, do not silently substitute): the best available dH/dt
# observational record on this system is MIPKIT's dhdt_cpom
# (/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc),
# annual, 1993-2019 -- there is no dataset here reaching 2025. dhdt_smith in the same file
# is a single static ~2019 epoch, usable as an independent cross-check only. Run
# 1995-2019, not 1995-2025.
if 'historical_dhdt_tuning' in steps:

    print("-------------------------------------------------------------")
    print(f" HISTORICAL RUN (1995-2019) TUNED AGAINST OBSERVED dH/dt"     )
    print("-------------------------------------------------------------")

    print(f"-- Loading post-calibration relaxed model...")
    md = pyissm.model.io.load_model(f'{model_dir}/AIS3_ho_relaxed.nc')

    md.inversion.iscontrol = 0
    md.verbose.solution = 1

    md.transient = pyissm.model.classes.transient.deactivate_all(md.transient)
    md.transient.isstressbalance = 1
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.isthermal = 0  # TODO: consider re-enabling once stage 1/2 are validated
    md.transient.isgroundingline = 1
    md.groundingline.migration = 'SubelementMigration'

    md.timestepping.start_time = 1995
    md.timestepping.final_time = 2019  # bounded by dhdt_cpom coverage, see note above
    md.timestepping.time_step  = 0.1   # years

    print(f"-- Building time-varying SMB from RACMO 1995-2019 annual means...")
    # smb.arma (pyissm/model/classes/smb.py) is the more sophisticated option (basin-wise
    # piecewise-polynomial trend + AR/MA lag structure) but fitting one honestly requires a
    # real statistical fitting procedure against historical SMB, not hand-picked
    # coefficients -- out of scope to fabricate here. Using smb.default's plain timeseries
    # support instead (confirmed available, smb.py:114) with REAL annual-mean RACMO smbgl
    # data for every year 1995-2019: a completely standard choice (this project's own param
    # step already uses an annual RACMO mean for surface temperature, same convention), just
    # not statistically modelled/extrapolated beyond the observed record.
    smb_years = np.arange(1995, 2020)
    racmo_smb_data = catalog.load_dataset('racmo2.4p1_monthly_11km_1979-2023')
    smbgl = racmo_smb_data['smbgl']  # kg/m^2 per month (glaciated-area surface mass balance)
    [racmo_x, racmo_y] = pyissm.tools.general.ll_to_xy(racmo_smb_data['lat'].values, racmo_smb_data['lon'].values, -1)

    nv = md.mesh.numberofvertices
    mb_arr = np.empty((nv + 1, smb_years.size))
    for i, yr in enumerate(smb_years):
        smb_yr = smbgl.sel(time = smbgl['time.year'] == yr)
        # kg/m^2/month water-equiv mass -> m ice-equiv thickness/yr: divide by rho_ice, sum
        # the 12 months (mass balance accumulates additively), no /12 (already summing
        # monthly totals into one annual total, not averaging a monthly rate).
        smb_yr_myr = (smb_yr.sum('time') / md.materials.rho_ice).to_numpy()
        mb_arr[:nv, i] = pyissm.data.interp.points_to_mesh(racmo_x, racmo_y, smb_yr_myr, md.mesh.x, md.mesh.y)
        print(f"   {yr}: mesh-mean SMB = {np.nanmean(mb_arr[:nv, i]):.3f} m ice eq/yr")
    mb_arr[nv, :] = smb_years.astype(float)  # ISSM timeseries convention: last row = time (yr)
    md.smb = pyissm.model.classes.smb.default(md.smb)
    md.smb.mass_balance = mb_arr

    print(f"-- Assigning cluster and updating settings...")
    md.miscellaneous.name = 'AIS3_historical_1995_2019'
    md.cluster = cluster
    md.settings.waitonlock = 0

    if save:
        print(f"-- Loading historical transient solution...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only = True, runtime_name = False)

        print(f"-- Comparing simulated dH/dt against MIPKIT's dhdt_cpom (1993-2019)...")
        mipkit = xr.open_dataset('/g/data/au88/ismip6/2300/forcings/ISMIP7/AIS/obs/mipkit/AntarcticaObsISMIP7-v1.2.nc')
        # dhdt_cpom is annual 1993-2019 on the 1km grid (y1km, x1km) -- use its LAST available
        # year as the closest match to our run's 2019 endpoint rather than trying to
        # reconstruct a full matching time series from a 27-step coordinate whose exact
        # date alignment to the 0.1yr model timestep should be checked at run time.
        dhdt_obs_grid = mipkit['dhdt_cpom'].isel(cpom_dhdt_time = -1)
        dhdt_obs_mesh = pyissm.data.interp.xr_to_mesh(
            mipkit, 'dhdt_cpom', md.mesh.x, md.mesh.y, x_var = 'x1km', y_var = 'y1km')

        thick_ts = md.results.TransientSolution
        H0 = np.asarray(thick_ts[0].Thickness).ravel()
        H1 = np.asarray(thick_ts[-1].Thickness).ravel()
        years_elapsed = md.timestepping.final_time - md.timestepping.start_time
        dhdt_sim = (H1 - H0) / years_elapsed

        gr = (np.asarray(md.mask.ice_levelset).ravel() < 0) & (np.asarray(md.mask.ocean_levelset).ravel() > 0)
        mismatch_rmse = np.sqrt(np.nanmean((dhdt_sim[gr] - dhdt_obs_mesh[gr]) ** 2))
        print(f"   Grounded dH/dt mismatch RMSE vs dhdt_cpom = {mismatch_rmse:.3f} m/yr")

        # TODO: actual tuning loop. This step reports the mismatch for ONE (gamma_0,
        # cf501) configuration; turning this into a real calibration means re-running
        # stage 3 (melt_gamma_tuning) and/or ssa_friction_inv_reg_lcurve at a few
        # different values and repeating this comparison, keeping whichever configuration
        # minimises mismatch_rmse -- mirroring the L-curve/sensitivity pattern already used
        # elsewhere in this pipeline (build_parameter_grid over the candidate values,
        # rerun this dH/dt comparison per candidate) rather than a new methodology. Not
        # done automatically here since each candidate requires re-running the full
        # multi-stage chain (friction/melt/relaxation/historical), which is a substantial
        # compute cost to automate blindly rather than direct.

        print(f"\nSaving to {model_dir}/AIS3_historical_1995_2019.nc")
        pyissm.model.io.save_model(md, f'{model_dir}/AIS3_historical_1995_2019.nc')
    else:
        print(f"-- Submitting historical transient run...")
        md = pyissm.model.execute.solve(md, 'Transient', load_only = False, runtime_name = False)
