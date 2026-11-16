# LARMIP contribution: ACCESS-ISSM (AIS3 configuration)

<!-- DRAFT 2026-10-01. Fill in before sending: model/version name as it should appear in the
paper, the CTRL drift numbers once the 200-yr control has run, and the three references marked
TODO in larmip_refs.bib. Citation keys refer to larmip_refs.bib. -->

**ACCESS-ISSM (AIS3).** The simulations use the Ice-sheet and Sea-level System Model (ISSM;
@larour2012) through its Python interface, pyISSM, in a continental Antarctic configuration
developed for ACCESS. Ice flow is solved with the shallow-shelf approximation [@macayeal1989] on an
unstructured triangular mesh of 1.59 million vertices, refined by observed speed: element edges are
at most 1.5 km on fast-flowing grounded ice, 0.5–5 km on ice shelves, and the median edge length
over the ice sheet is 1.5 km. Bed topography and initial ice geometry are from BedMachine Antarctica
v3 [@morlighem2020]. Basal sliding follows a Budd-type law [@budd1979] with linear dependence on
velocity and effective pressure (p = q = 1). Effective pressure is taken from the subglacial
hydrology estimate of @ehrenfeucht2024 and bounded below by 7% of the ice overburden pressure. The
flow law uses n = 3. On grounded ice the rate factor follows from RACMO2.4p1 surface temperature
[@vandalum2024] via the temperature dependence of @cuffey2010. On floating ice it is obtained by
inversion. The grounding line evolves with sub-element parameterisations of its position and of basal
friction [@seroussi2014]; no basal melt is applied to partially floating elements. The
calving-front position is held fixed, so the experiments isolate the response to melt. Sub-shelf melt uses the ISMIP6 local quadratic
parameterisation [@jourdain2020] driven by an observational thermal-forcing climatology
[@zhou_tf]. Its coefficient (γ0 = 300 m yr⁻¹) and per-basin temperature corrections were
calibrated, following the ISMIP7 ocean protocol, to basin-integrated present-day melt
[@paolo2023; @adusumilli2020] in IMBIE2 basins [@mouginot2017], with buttressing-weighted
skill [@reese2018] used to choose γ0.

**Initialisation and spin-up.** The model is initialised by data assimilation rather than a
long thermal spin-up. The basal friction coefficient is inferred by minimising the misfit to MEaSUREs
InSAR-based surface velocities [@rignot2011; @mouginot2019] with an adjoint method [@morlighem2010]
and the M1QN3 optimiser [@gilbert1989]. The cost function combines absolute and logarithmic
velocity misfit with a Tikhonov regularisation of the friction gradient; the resulting grounded-ice
velocity RMSE is 66 m yr⁻¹ in a full-continent forward solve. From this state the model is run
from 1995 to 2019 with annual RACMO2.4p1 surface mass balance and the calibrated sub-shelf melt;
modelled elevation change over that period was compared with CryoSat-2-era altimetry. The 2019
end state is the starting point for all LARMIP experiments. All experiments hold surface mass balance
at its 1995–2014 RACMO2.4p1 mean and the ocean forcing constant. Over the 200 years the control
gains ice at a slowly decreasing rate, from 0.56 to 0.39 mm yr⁻¹ of sea-level equivalent
(−89 mm in total), with mass loss only in the Amundsen Sea sector. All results are reported relative to the control.
In the LARMIP experiments the prescribed additional basal melt is added to the parameterised melt
under every fully floating element in the respective region, so the forced area follows the
grounding line as it migrates.
