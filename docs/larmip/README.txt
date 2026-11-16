ACCESS-AIS3 -- LARMIP-3 contribution
====================================

Contact: Justin Kin Jun Hew (justinkinjun.hew@anu.edu.au)

Files
-----
AIS3_ssa_larmip_sle_coupledN_p3.csv   Main set: 1 control, 5 step and 5 ramp experiments
AIS3_ssa_larmip_sle.csv               Sliding-law sensitivity set: the same 11 experiments
model_description.tex / .md           Model description (half a page) for the paper
larmip_refs.bib                       References cited in the model description
authors.md                            Contributing authors and affiliations

The two sets differ only in the basal sliding law (see the model description):
  main (coupledN_p3)   Budd law, p = 3, q = 1. Effective pressure N = overburden minus ocean
                       pressure at the bed, floored at 7% of overburden, so it evolves with
                       the geometry. The friction coefficient is rescaled once at 2019 so that
                       the initial basal drag equals that of the inverted state.
  sensitivity          The inverted law unchanged: linear Budd law (p = q = 1) with effective
                       pressure prescribed from a subglacial hydrology model, fixed in time.

Format
------
Comma-separated text. The first line is a comment (starting with "#"), the second is the
column header, then 201 rows.

  year     Model year, 0 to 200. Year 0 is the initial state (2019.0); year 200 is 2219.0.
           Values are snapshots at the start of each model year, so year 0 is zero.
  ctrl     Control: no additional melt.
  larmip_r<n>_step    8 m/yr additional melt under the ice shelves of sector n from year 0.
  larmip_r<n>_ramp    Additional melt rising linearly from 0 to 8 m/yr over years 0-100,
                      then held at 8 m/yr for years 100-200 (the optional experiments).

In the main-set file every column name carries the suffix "_coupledN_p3".

Sector numbers n follow LARMIP3_sectors-1km.nc:
  1 Weddell, 2 East Antarctica, 3 Ross, 4 Amundsen, 5 Peninsula.

Units and sign
--------------
Sea-level contribution in metres, positive = sea-level rise.

Values are NOT control-subtracted: each column is that run's own change since year 0. The
response to the additional melt is (experiment - ctrl) from the same file.

Sea-level contribution is computed from the change in ice volume above flotation (VAF):
  SLE = -dVAF * rho_ice / rho_ocean / A_ocean,
  rho_ice = 917 kg m^-3, rho_ocean = 1028 kg m^-3, A_ocean = 3.625e14 m^2.
VAF follows the ISMIP convention: vertex-area weighted, grounded vertices only, height above
flotation clipped at zero. It is computed from the annual thickness and grounded-ice mask.

Experiment set-up
-----------------
- Additional melt is added to the parameterised (ISMIP6 local quadratic) melt under every
  fully floating element in the sector. The forced area therefore follows the grounding line
  as it migrates. Partially floating elements receive no melt.
- Calving front fixed.
- Surface mass balance held at its 1995-2014 RACMO2.4p1 mean.
- Ocean thermal forcing held constant.
- Time step 0.1 yr.
- Mesh: 1.59 million vertices.
- Ice flow: shallow-shelf approximation (ISSM, via pyISSM).

Control drift over 200 years:
  main set           -56.7 mm (the ice sheet gains mass)
  sensitivity set    -89 mm
