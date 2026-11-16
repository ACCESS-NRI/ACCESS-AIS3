# LARMIP contribution: ACCESS-AIS3

**ACCESS-AIS3.** The simulations use the Ice-sheet and Sea-level System Model (ISSM;
@larour2012) through its Python interface, pyISSM, in a continental Antarctic configuration
developed for ACCESS. Ice flow is solved with the shallow-shelf approximation [@macayeal1989] on an
unstructured triangular mesh of 1.59 million vertices, refined by observed speed: element edges are
at most 1.5 km on fast-flowing grounded ice, 0.5–5 km on ice shelves, and the median edge length
over the ice sheet is 1.5 km. Bed topography and initial ice geometry are from BedMachine Antarctica
v3 [@morlighem2020]. Basal sliding follows a Budd-type law [@budd1979],
τ_b = C² N^(1/3) |u|^(−2/3) u (p = 3, q = 1). The effective pressure N is the ice overburden minus the
ocean pressure at the bed (full hydraulic connection to the ocean), bounded below by 7% of the
overburden, so basal drag falls as grounded ice thins towards flotation. The flow law uses n = 3. On
grounded ice the rate factor follows from RACMO2.4p1 surface temperature [@vandalum2025] via the
temperature dependence of @cuffey2010. On floating ice it is obtained by inversion. The grounding
line evolves with sub-element parameterisations of its position and of basal friction
[@seroussi2014]; no basal melt is applied to partially floating elements. The calving-front position
is held fixed, so the experiments isolate the response to melt. Sub-shelf melt uses the ISMIP6 local
quadratic parameterisation [@jourdain2020] driven by an observational thermal-forcing climatology
[@zhou2024]. Its coefficient (γ0 = 300 m yr⁻¹) and per-basin temperature corrections were calibrated,
following the ISMIP7 ocean protocol [@reese2026], to basin-integrated present-day melt
[@paolo2023; @adusumilli2020] in IMBIE2 basins [@mouginot2017], with buttressing-weighted skill
[@reese2018] used to choose γ0.

**Initialisation and spin-up.** The model is initialised by data assimilation rather than a long
thermal spin-up. The basal friction coefficient is inferred by minimising the misfit to MEaSUREs
InSAR-based surface velocities [@rignot2011; @mouginot2019] with an adjoint method [@morlighem2010]
and the M1QN3 optimiser [@gilbert1989], using a linear Budd law (p = q = 1) with effective pressure
prescribed from the subglacial hydrology estimate of @ehrenfeucht2025. The cost function combines
absolute and logarithmic velocity misfit with a Tikhonov regularisation of the friction gradient;
the resulting grounded-ice velocity RMSE is 66 m yr⁻¹ in a full-continent forward solve. From this
state the model is run from 1995 to 2019 with annual RACMO2.4p1 surface mass balance and the
calibrated sub-shelf melt; modelled elevation change over that period was compared with
CryoSat-2-era altimetry. The 2019 end state is the starting point for all LARMIP experiments. For the
experiments, C is rescaled once so that the basal drag of this state is unchanged under the
nonlinear, flotation-coupled law above; initial grounded speeds agree with the inverted state to
within 5% at 90% of vertices. All experiments hold surface mass balance at its 1995–2014 RACMO2.4p1
mean and the ocean forcing constant. Over the 200 years the control gains ice at a slowly decreasing
rate, from 0.39 to 0.23 mm yr⁻¹ of sea-level equivalent (−57 mm in total), with mass loss only in
the Amundsen Sea sector. All results are reported relative to the control. In the LARMIP experiments
the prescribed additional basal melt is added to the parameterised melt under every fully floating
element in the respective region, so the forced area follows the grounding line as it migrates.

**Sliding-law sensitivity.** We also provide a second set of experiments that keeps the inverted law
unchanged: linear sliding (p = q = 1) with the prescribed effective pressure held fixed in time. Its
control drifts by −89 mm over 200 years, and its responses after 200 years are 2.0–2.6 times smaller
than those of the main set. In both sets the ramp experiments agree with the linear-response
prediction from the step experiments to within 13% (mostly within 7%).
