To: anders.levermann@pik-potsdam.de
Cc: Felicity McCormack, Mike Tetley
Subject: LARMIP-3 contribution: ACCESS-AIS3

Dear Anders,

Please find attached our LARMIP-3 contribution from ACCESS-AIS3, a continental Antarctic
configuration of ISSM developed for ACCESS.

We have run the control, the five 8 m/yr step experiments and the five optional ramp
experiments. Each run covers 200 years with yearly output.

We submit two sets of runs that differ only in the basal sliding law:

- Main set (AIS3_ssa_larmip_sle_coupledN_p3.csv): a Budd law with p = 3, with effective
  pressure that evolves as grounded ice thins towards flotation.
- Sensitivity set (AIS3_ssa_larmip_sle.csv): the linear law with fixed effective pressure that
  the model was initialised with.

The main set's responses after 200 years are 2.0-2.6 times larger than the sensitivity set's. We
would suggest using the main set in the paper, with the second set as a sliding-law sensitivity
if that is useful. We are happy to provide only one set if you prefer.

The attachments are:

1. Sea-level time series: one CSV per set, each with the control, step and ramp runs. They are
   in metres, positive = sea-level rise, from volume above flotation, and not
   control-subtracted. README.txt describes the columns and conventions.
2. (a) Model description, including the initialisation (model_description.tex, also as .md).
   (b) BibTeX file with the cited references (larmip_refs.bib).
   (c) Authors and affiliations (also in authors.md):

       Justin Kin Jun Hew (1), Felicity McCormack (2, 3), Mike Tetley (1)

       (1) ACCESS-NRI (Australia’s Climate Simulator), Australian National University,
           Canberra, ACT 2601, Australia
       (2) Securing Antarctica’s Environmental Future, Monash University, Clayton, Victoria,
           Australia
       (3) School of Earth, Atmosphere and Environment, Monash University, Clayton,
           Kulin Nations, Victoria, Australia

Two things to check:

- We read the sector numbers in LARMIP3_sectors-1km.nc as 1 Weddell, 2 East Antarctica, 3 Ross,
  4 Amundsen, 5 Peninsula. Please let us know if that is not the intended order.
- The additional melt is applied only to fully floating elements, so the forced area follows
  the grounding line as it retreats. Please tell us if you intended the forced area to stay
  fixed at the initial ice-shelf extent.

Please let us know if you need the data in a different format.

Best regards,
Justin Hew
(on behalf of Felicity McCormack and Mike Tetley)
