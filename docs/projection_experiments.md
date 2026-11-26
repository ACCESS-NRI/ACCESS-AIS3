# ACCESS-AIS3 projection experiments (SSA track)

How the Antarctic projection design (control, SSP scenarios, ocean/melt uncertainty,
attribution, dynamical sensitivities) is implemented for this configuration, what forcing
exists for it, and what has been run. Code: `projection_ssp_ssa` in
[`config/ais_0.1_SSA.py`](../config/ais_0.1_SSA.py), launched per experiment by
[`config/launch_projection_ssa.pbs`](../config/launch_projection_ssa.pbs). Pipeline history:
[`post_inversion_pipeline_worklog.md`](post_inversion_pipeline_worklog.md) §6.

## Choices made

| Item | Choice | Why |
|---|---|---|
| Track | SSA only | Only track with a checked historical run (worklog §4q–§4t). HO flows 10–15% too fast (§5a) and would cost >100k SU per run to 2300. |
| Start | 2019.0, end state of `AIS3_historical_1995_2019_SSA.nc` | Continues the historical run. |
| End | 2300; 2100 for ssp370 | ISMIP7 ssp370 forcing stops at 2100 for every GCM. |
| Ocean forcing | `TF = TF0 + [TF_gcm(t) − TF_gcm(1995–2014)]`, TF0 = Zhou climatology | `deltaT_basin` was fitted against Zhou; absolute GCM TF would carry each GCM's bias into that fit. |
| SMB forcing | `SMB = SMB0 + [SMB_gcm(t) − SMB_gcm(1995–2014)]`, SMB0 = RACMO 1995–2014 mean | Keeps RACMO's spatial pattern, adds the GCM trend. |
| GCM SMB product | `dEBM2-8000m` | The only product available for all seven ISMIP7 GCMs. |
| Reference period | 1995–2014 | Overlaps the historical run; GCM historical files end in 2014. |
| Melt | ref = gamma_0 300 + refit deltaT (`AIS3_melt_final_ssa.nc`); high = gamma_0 15000 + its refit deltaT (`AIS3_PROJ_MELT=high`, candidate g7); no low member | Decided 2026-09-29 from J3 (worklog §6.1): 300 is kept as reference (ISMIP7-style weighting still picks it) but gives ~4% of the ocean models' warming response; 15000 is the J1+J2+J3 95th percentile (~33%). Nothing converged lies below 300. **Superseded 2026-11-19** (worklog §6.29-6.31): with the real J2 target and J4, 300 is ruled out; the new reference is gamma_0 = 4,000 (`AIS3_SSA_CALIB=_hm4000`), range ~1,000-8,000. The October set below still uses 300; the v2 set uses `_hm4000`. |
| Shelf collapse | Optional, ISMIP7 fracture path C (`AIS3_PROJ_COLLAPSE=1`), see below | One run so far (`ssp585_cesm2_ref_ref_collapse`, complete to 2300). |
| Not included | SMB elevation feedback, calving (front fixed), GIA, thermal evolution | No implementation yet. ISMIP7 ships `dacabfdz` for the feedback (dEBM2-8000m: historical 1850-2014 and ssp126/534-over/585, both GCMs); planned in the v2 set, stage B. |

Anomalies are annual (placed at mid-year), zero where a source grid has no data, and a GCM's
last year is held if a run goes past it (ssp585 TF ends 2299).

## Available ISMIP7 forcing

| GCM | Scenarios | To 2300? |
|---|---|---|
| CESM2-WACCM | ctrl, ssp126, ssp534-over, ssp585 (ssp245: dEBM2 SMB only, 2015-2100; no ocean, `dacabfdz` or collapse mask, checked 2026-11-20) | yes (ssp370: 2100) |
| MRI-ESM2-0 | ctrl, ssp126, ssp534-over, ssp585 | yes (ssp370: 2100) |
| ACCESS-CM2, CanESM5, GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR | ssp370 only | no, 2100 |

So the SSP spread to 2300 uses two GCMs (ssp126/ssp585), and GCM spread is only possible at
ssp370 (seven GCMs, to 2100).

## Experiment matrix

Names follow `<scenario>_<gcm>_<melt>_<forcing>`; `ctrl` has no forcing anomaly.

| Stage | Experiment | `qsub -v` settings (plus `AIS3_STEPS=projection_ssp_ssa`) | Status |
|---|---|---|---|
| 1 | `ctrl` | `AIS3_PROJ_SCENARIO=ctrl` | **done** (2026-10-09; results in post_inversion_pipeline_worklog.md §6.11) |
| 1 | `ssp126_cesm2_ref_ref`, `ssp585_cesm2_ref_ref` | `SCENARIO=ssp126/ssp585, GCM=CESM2-WACCM` | **done** (2026-10-09; results in post_inversion_pipeline_worklog.md §6.11) |
| 1 | `ssp370_cesm2_ref_ref` (to 2100) | `SCENARIO=ssp370, GCM=CESM2-WACCM` | **done** (2026-10-09; results in post_inversion_pipeline_worklog.md §6.11) |
| 2 | `ssp585_cesm2_high_ref`, `ssp370_cesm2_high_ref` | `AIS3_PROJ_MELT=high` | **done** (2026-10-09; results in post_inversion_pipeline_worklog.md §6.11) |
| 2 | `ssp585_cesm2_ref_{oceanonly,smbonly}` | `AIS3_PROJ_FORCING=ocean/smb` | **done** (2026-10-09; results in post_inversion_pipeline_worklog.md §6.11) |
| 2 | `ssp{126,370,585}_mri_ref_ref`, `ssp370_{access,canesm5,gfdl,ipsl,mpi}_ref_ref` | `GCM=...` | **done** (2026-10-09; results in post_inversion_pipeline_worklog.md §6.11) |
| 3 | `<scenario>_<gcm>_ref_ref_collapse` | `AIS3_PROJ_COLLAPSE=1` | `ssp585_cesm2_ref_ref_collapse` **done** to 2300 (2026-10-27; worklog 6.12) |
| 3 | sliding law | `AIS3_PROJ_FRICTION=coupledN` / `coupledN_p3` | used for LARMIP only (`ctrl_coupledN_p3` to 2219); no SSP run yet, see v2 set |
| 3 | GIA | not implemented | — |

## Next set (v2): plan, 2026-11-20

The October set's three headlines each rest on a choice a reviewer can attack:
1. "Snowfall wins to 2100."
2. "The CESM2 ssp585 reversal after ~2185 is SMB-driven."
3. "Collapse adds ~315 mm by 2300."

The choices:
- sliding is linear with fixed N, while LARMIP-3 went in with `coupledN_p3`;
- the melt calibration is gamma_0 = 300, now ruled out (worklog 6.29-6.31), and it gives the
  ocean-only +31 mm by 2300;
- there is no SMB-elevation feedback, although claim 2 is entirely SMB-driven and the CESM2
  ssp585 anomaly reaches -3.8 m ice eq/yr as a mesh mean by 2275;
- there is no melt on partly floating elements (x1.35 in the LARMIP test, worklog 6.18).

v2 moves the reference to the defensible physics and then adds the two missing processes as a 2x2.

**v2 reference:** `AIS3_SSA_CALIB=_hm4000` (gamma_0 = 4,000, worklog 6.31) +
`AIS3_PROJ_FRICTION=coupledN_p3`, the LARMIP-3 main set.
- C is rescaled once at 2019 from the `_hm4000` historical state, which is Budd p=1 with fixed N.
  This is the same method as the LARMIP-3 runs, which started from the production historical.
- The `_hm4000_p3_flotN` historical (worklog 6.32) is not used as the start: `init_path` does not
  carry `AIS3_SSA_FRICTION`, and the 1995-2019 record cannot tell the friction laws apart
  (worklog 6.33).
- Names end `_coupledN_p3_hm4000`. In every table below the `qsub -v` settings also include
  `AIS3_STEPS=projection_ssp_ssa,AIS3_SSA_CALIB=_hm4000,AIS3_PROJ_FRICTION=coupledN_p3`.

SU estimates scale the measured October totals (Running, below) by 1.5 for `coupledN_p3`. The
LARMIP `coupledN_p3` runs cost 1.1-2x their standard twins.

### Stage A: v2 core set (no code changes)

| Experiment | `qsub -v` settings | End | Est. SU | Tests |
|---|---|---|---|---|
| `ctrl_coupledN_p3_hm4000` | `AIS3_PROJ_SCENARIO=ctrl` | 2300 | ~10k | The 2026-11-20 smoke test (184750273) built `AIS3_proj_ssa_init_hm4000.nc` and `_baseline_hm4000.npz`, so the whole set can start together. |
| `ssp585_cesm2_ref_ref_coupledN_p3_hm4000` | `SCENARIO=ssp585, GCM=CESM2-WACCM` | 2300 | ~25k | How far claims 1-2 and the ocean-only term move under v2 |
| `ssp585_mri_ref_ref_coupledN_p3_hm4000` | `SCENARIO=ssp585, GCM=MRI-ESM2-0` | 2300 | ~13k | as above |
| `ssp126_{cesm2,mri}_ref_ref_coupledN_p3_hm4000` | `SCENARIO=ssp126, GCM=...` | 2300 | ~15k + 12k | Low end of the scenario range |
| `ssp534-over_{cesm2,mri}_ref_ref_coupledN_p3_hm4000` | `SCENARIO=ssp534-over, GCM=...` | 2300 | ~2 x 13k | Overshoot: does the reversal depend on the path, or only on the warming reached? |
| `ssp585_cesm2_ref_ref_collapse_coupledN_p3_hm4000` | `SCENARIO=ssp585, GCM=CESM2-WACCM, AIS3_PROJ_COLLAPSE=1` | 2300 | ~25k | Claim 3 under v2. The residue 1e-2 and 480-min inner floor are now defaults. |

Plus one offline check of the Ross 2214 removal, with no solves. The 2,357 masked + 9,911
unanchored vertices (~209k km²) removed in one year are the largest single step in claim 3
(worklog 6.12).
- Ice mask at 2213 = the `ice_levelset` of `AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2200.nc`,
  minus vertices with `collapsed_at` < 2214 in the 2200-2225 history.
- Rerun apply_collapse's connectivity test on that mask with the 2213 `ocean_levelset`.
- Confirm whether the remnant really had no edge connection to grounded ice.

Not in v2: ssp245 CESM2 (SMB only, to 2100) and more ssp370 GCMs. Pinned HO ctrl and ssp585
are already complete to 2300 (worklog 6.22); no further HO projection runs are planned.

### Stage B: SMB-elevation feedback (`AIS3_PROJ_SMBFB=1`, implemented 2026-11-20)

How it works (`projection_ssp_ssa`; header comment in `ais_0.1_SSA.py`):
- `md.smb` is ISSM's `SMBgradients`, SMB = `smbref` + `b` (s - `href`), evaluated every time step,
  so there are no extra sub-solves. ISSM's formula has no clamping and no unit conversion
  (`SmbGradientsx`); pyissm divides `smbref` and `b` by yts when writing.
  - `smbref` = the annual SMB columns already built (SMB0 + anomaly; SMB0 without SMB forcing);
  - `href` = the 2019 surface of the init state, so the feedback starts at zero (cached as
    `models/AIS3_proj_ssa_href<calib>.npy`);
  - `b_pos = b_neg` = ISMIP7 `dacabfdz` (dEBM2-8000m, same 761x761 grid as `acabf`), interpolated
    like the SMB anomaly and converted from kg m-2 s-1 m-1 to m ice eq/yr per m (x yts / rho_ice).
    NaN becomes 0. With SMB forcing it is the scenario's annual field (last year held). Without SMB
    forcing (ctrl, ocean only) it is the GCM's historical 1995-2014 mean, held fixed.
- Size, CESM2, m ice eq/yr per km, p1 / p50 / p99 over the grid: historical 2000 -0.59 / 0.00 /
  +0.53; ssp585 2100 -0.76 / 0.00 / +1.54; 2275 -0.49 / 0.00 / +3.71.
- ctrl takes the gradient's GCM from `AIS3_PROJ_SMBFB_GCM`, so each GCM gets its own feedback
  ctrl. Names: `_smbfb` (ctrl `_smbfb-<gcm>`). History files also hold `surface`. Both variables
  are passed down the chain.
- Smoke check: in the ctrl smoke history, SMB - SMB0 must equal `b` (s - `href`) vertex by vertex.

| Experiment | `qsub -v` settings | End | Est. SU |
|---|---|---|---|
| `ctrl_coupledN_p3_smbfb-cesm2_hm4000` | `SCENARIO=ctrl, AIS3_PROJ_SMBFB=1, AIS3_PROJ_SMBFB_GCM=CESM2-WACCM` | 2300 | ~10k |
| `ctrl_coupledN_p3_smbfb-mri_hm4000` | `SCENARIO=ctrl, AIS3_PROJ_SMBFB=1, AIS3_PROJ_SMBFB_GCM=MRI-ESM2-0` | 2300 | ~10k |
| `ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000` | `SCENARIO=ssp585, GCM=CESM2-WACCM, AIS3_PROJ_SMBFB=1` | 2300 | ~25k |
| `ssp585_mri_ref_ref_coupledN_p3_smbfb_hm4000` | `SCENARIO=ssp585, GCM=MRI-ESM2-0, AIS3_PROJ_SMBFB=1` | 2300 | ~13k |

Tests claim 2: the feedback should make the reversal larger and earlier.

### Stage C: melt on partly floating elements (SEM1) and the "best physics" member

Set up 2026-11-21; the four runs below launched 2026-11-23 (184910466-69) once the v2 MRI ocean-only term reached
+38.8 mm by 2175 (~4x October, doubling every ~25 yr). SEM1 is a calibration tag, `AIS3_SSA_CALIB=_hm4000sem1`, so
it covers the whole chain (melt step, historical, every projection chunk):
- **ISSM build:** `/g/data/au88/jh7060/ISSM-40c764c7-sem1/install` is the production commit `40c764c7` (spack
  issm-git.2026.05.18), rebuilt by `config/build_issm_sem1.pbs` with the recorded spack build environment and
  configure arguments, plus one change. `Element::Ismip6FloatingiceMeltingRate` now skips only fully grounded
  elements (`this->IsAllGrounded()`), not every element that isn't fully floating (`!this->IsAllFloating()`).
  - Its `issm.exe` RPATH points at its own `lib/`, so the production library on `LD_LIBRARY_PATH` from the
    inner jobs' module cannot replace it.
  - Marker file `SEM1_PATCH` in the prefix. Pass `AIS3_ISSM_DIR=<prefix>`; it is passed down the chain.
- **Calibration:** `config/melt_calibration_hm_refit_sem1.py` writes `models/melt_calibration_hm_sem1.json`,
  member `hm4000sem1`.
  - ISSM's SubelementMelt1 applies `(1 - phi) fmb + phi gmb` (phi = grounded fraction), so each element
    contributes mean melt x area x floating fraction.
  - deltaT is refit so that each basin's SEM1 total equals its no-SEM1 total at `hm4000` (same element-integral
    form). Present-day melt per basin is unchanged; SEM1 only moves it towards the grounding line.
- **Driver:**
  - a `_hm<g>sem1` tag needs the marker (refuses otherwise) and sets `melt_interpolation = SubelementMelt1` in
    the historical run and every projection chunk;
  - calibration members are looked up in every `models/melt_calibration_hm*.json`;
  - any Subelement melt interpolation with ismip6 melt on an unpatched build is refused (it would change
    nothing).
- **Order:**
  1. melt step and historical run with `AIS3_SSA_CALIB=_hm4000sem1, AIS3_ISSM_DIR=<prefix>` (~1.5k SU);
  2. a one-chunk check against the LARMIP SEM1 test (x1.34-1.39);
  3. the runs below, **only if** the v2 ocean term turns out large enough to matter (MRI ocean-only, v2 vs
     October CESM2).

| Experiment | `qsub -v` settings | End | Est. SU |
|---|---|---|---|
| `ctrl_coupledN_p3_hm4000sem1`, `ssp585_cesm2_ref_ref_coupledN_p3_hm4000sem1` | `AIS3_SSA_CALIB=_hm4000sem1, AIS3_ISSM_DIR=/g/data/au88/jh7060/ISSM-40c764c7-sem1/install` (+ `SCENARIO`, `GCM`) | 2300 | ~10k + 25k |
| `ctrl_coupledN_p3_smbfb-cesm2_hm4000sem1`, `ssp585_cesm2_ref_ref_coupledN_p3_smbfb_hm4000sem1` | as above + `AIS3_PROJ_SMBFB=1` (ctrl also `AIS3_PROJ_SMBFB_GCM=CESM2-WACCM`) | 2300 | ~10k + 25k |

With stages A and B, this gives a full 2x2 (SEM1 x SMB feedback) at v2 for ssp585 CESM2, each
cell with its own ctrl. That attributes each process and their interaction from the
configuration we would defend. Removing `coupledN_p3` and `_hm4000` one at a time from the best
member would add 4 more runs (~70k). Deferred: the October set vs stage A already gives their
combined effect.

### Stage D: robustness of the method

| Experiment | `qsub -v` settings | End | Est. SU | Tests |
|---|---|---|---|---|
| `ctrl_coupledN_p3_driftcorr_hm4000`, `ssp585_cesm2_ref_ref_coupledN_p3_driftcorr_hm4000` | `AIS3_PROJ_DRIFTCORR=ctrl_coupledN_p3_hm4000, AIS3_PROJ_END=2150` (+ `SCENARIO`, `GCM`). Implemented 2026-11-21: fixed SMB flux correction = minus that ctrl's (H(2020.0) - H(2019.1)) / 0.9 yr on every ice vertex; for v2 that is -114 Gt/yr on grounded ice and +243 Gt/yr on floating ice. | 2150 | ~5k + 8k | Is subtracting the control valid? The v2 ctrl gains +111 Gt/yr of ice above flotation in its first year and +134 Gt/yr over 2020-2030, while Antarctica is losing mass. A >10% change in forced minus ctrl means the 2100 numbers need a caveat. |
| `ssp585_mri_ref_{oceanonly,smbonly}_coupledN_p3_hm4000` | `SCENARIO=ssp585, GCM=MRI-ESM2-0, AIS3_PROJ_FORCING=ocean/smb` | 2300 | ~2 x 13k | Is the reversal SMB-driven for MRI too? MRI barely reverses (-80 -> -39 mm). Submitted 2026-11-21 (184792079 / 184792083). |
| `ctrl_coupledN_p3_hm8000`, `ssp585_cesm2_ref_ref_coupledN_p3_hm8000` | `AIS3_SSA_CALIB=_hm8000` (+ `SCENARIO`, `GCM`) | 2300 | ~35k + ~1.5k for its melt step and historical | Upper end of the melt range. Run the historical submit-only, then load-only, as in worklog 6.32/6.34. Melt step 184792047 -> historical submit-only 184792049 (2026-11-21); load-only, then ctrl and ssp585, still to do. |

### Stage E: no SU (Python)

Linear-response emulator for the ocean term:
- Inputs: the per-sector LARMIP step responses (both sets; ramp vs step prediction 0.96-1.13,
  worklog 6.15), convolved with each GCM's sector melt anomaly.
- The melt anomaly is computed from the TF anomaly with the hm1000 / hm4000 / hm8000
  calibrations (Python melt as `melt_history_match.py`).
- Gives the ocean contribution for all seven GCMs and the lower melt member without runs.
- Caveat: the step forcing is uniform per sector, so the emulator ignores melt pattern, collapse
  and SEM1.

### Later, not costed here

- Non-local quadratic (`islocal=0`) and PICO: each needs its own J1-J4 history-matching
  calibration before any projection run. The quadratic law's structural J3 misfit (worklog 6.30)
  is the reason to try them.
- Initialisation ensemble: 2-3 friction inversions along the L-curve, and Ehrenfeucht vs
  flotation N, each through relaxation, ctrl and ssp585 to 2150.
- Grounded B from the HO thermal steady state, with friction re-inverted on it. This replaces B
  rather than inverting it, so the grounded-B gradient problem of worklog 6.25 does not block it.
- Resolution: the Amundsen LARMIP step with a ~500 m grounding-line mesh (now 1.33 km median).

### Budget and order

| Stage | Est. SU |
|---|---|
| A (7 SSA runs + collapse + Ross check) | ~126k |
| B (feedback: 2 ctrls, CESM2, MRI) | ~58k |
| C (SEM1 2x2 cells) | ~70k |
| D (drift test, MRI attribution, hm8000) | ~76k |
| Total | ~330k |

- au88 had 684k SU available on 2026-11-20, for the whole project, until the end of 2026.q4.
- If budget or time runs short: ctrl + ssp585 CESM2/MRI (stage A), stage B and the Ross check,
  ~105k, is the minimum that tests all three headlines.
- A submit-only projection driver, like `AIS3_HIST_SUBMIT_ONLY`, would cut every estimate by
  40-50%: the idle 128 GB driver is that share of the measured totals.

## Running

```bash
cd config
qsub -v AIS3_STEPS=projection_ssp_ssa,AIS3_PROJ_SCENARIO=ctrl \
     -o logs/proj_ssa_ctrl_2019.out -e logs/proj_ssa_ctrl_2019.err launch_projection_ssa.pbs
```

- Each chunk (2019→2050→2100→…→2300) runs in one submission and then queues the next itself.
- To stop a chain after its current chunk: `touch models/AIS3_proj_ssa_<exp>.STOP`.
- A 0.2-yr smoke test: add `AIS3_PROJ_SMOKE=1` (writes `*_smoke` names only).
- The first chunk of the first experiment builds `models/AIS3_proj_ssa_init.nc` and
  `AIS3_proj_ssa_baseline.npz` (TF0, SMB0). Do not start two experiments until those exist.

Outputs per experiment:
- a checkpoint per chunk end: `models/AIS3_proj_ssa_<exp>_<year>.nc`;
- annual history per chunk: `models/AIS3_proj_ssa_<exp>_<t0>-<t1>_history.npz` (thickness,
  ocean levelset, speed, basal melt, SMB, and scalar series: VAF, grounded/floating area, total
  SMB and BMB, grounding-line and ice-front flux).

Cost: about 12.6 min of the 48-cpu solve per model year (historical run), so ~60 h of solve
per run to 2300. Measured totals (inner solves + driver jobs, retries included; 2 SU per cpu-hour):
~2.5-3k SU to 2100; to 2300 ctrl 6.4k, ssp585 MRI 8.5k, ssp126 CESM2 10.6k, ssp585 CESM2 16.6k,
high melt 18.2k, collapse 17.3k. Stronger forcing slows the solve after ~2150. The 32-cpu / 128 GB
driver waiting on the solve is 40-50% of every total (~64 SU/h idle).

## Shelf collapse (ISMIP7 fracture path C)

`AIS3_PROJ_COLLAPSE=1` (SSA and HO steps) reads the GCM/scenario's annual collapse mask,
`<GCM>/<scenario>/fracture/<ver>/ice_shelf_collapse_mask_*_ismip7_8km*.nc` (0/1 `mask`, years
1950-2299, ssp370 to 2100, on the TF grid; "ice shelves should collapse on January 1st", floating
ice only). The newest version is used unless `AIS3_PROJ_COLLAPSE_VER` says otherwise:

| GCM | Scenarios | Versions (default **bold**) |
|---|---|---|
| CESM2-WACCM | ssp126, ssp370, ssp534-over | v2, **v2.1** |
| CESM2-WACCM | ssp585 | v1, v2, **v2.1** |
| MRI-ESM2-0 | ssp126, ssp370 | **v1** |
| MRI-ESM2-0 | ssp534-over, ssp585 | v1, **v2** |

No mask exists for ssp245, ctrl or the five ssp370-only GCMs.

How it is applied (SSA step only; the HO step refuses the option for now):
- Each chunk is split into sub-solves at every 1 January on which the mask changes on the mesh
  (linear interpolation, then >= 0.5).
- Before each sub-solve, every masked vertex that holds floating ice is made ice-free
  (`ice_levelset` = +1, thickness `min_thickness`). So are floating pieces that this cuts off
  from grounded ice, which would otherwise move as rigid bodies.
- Only floating ice collapses. Ice that ungrounds inside the mask goes at the next mask change
  or chunk start.
- Everywhere else the front stays fixed (`ismovingfront=0`). The new edge is an ordinary
  calving front with ocean pressure.
- Removed shelf mass leaves no flux term; it shows as a drop in floating ice volume. History
  files get `collapsed_at`, the removal time for each vertex.

A first version applied the mask as a 1e4 m/yr extra melt (`melt_anomaly`), which left
collapsed shelves at 1 m. Its smoke test blew up: detached shelf remnants reached 4e9 m/yr
(worklog 6.12).

Sampled collapsed area on the 8 km grid, CESM2 ssp585 v2.1: 2k km² (2019), 25k (2050),
300k (2100), 0.9M (2150), 2.0M (2200), 3.35M (2299). CESM2 ssp126 v2.1: 137k km² by 2299. The
mask also covers open ocean and grounded ice; the floating-only rule handles both.

```bash
cd config
qsub -v AIS3_STEPS=projection_ssp_ssa,AIS3_PROJ_SCENARIO=ssp585,AIS3_PROJ_GCM=CESM2-WACCM,AIS3_PROJ_COLLAPSE=1 \
     -o logs/proj_ssa_ssp585_cesm2_ref_ref_collapse_2019.out -e logs/proj_ssa_ssp585_cesm2_ref_ref_collapse_2019.err \
     launch_projection_ssa.pbs
```

## HO runs (check on the SSA ensemble)

Plan (decided 2026-09-30): at most three HO runs, all to 2100 with CESM2-WACCM forcing identical
to the SSA runs, reported as forced minus HO CTRL and compared with the SSA equivalents:

| Experiment | `qsub -v` (plus `AIS3_STEPS=projection_ssp`) | Status |
|---|---|---|
| `ho_relaxfix/ctrl` | `AIS3_PROJ_SCENARIO=ctrl` | waits on prerequisites |
| `ho_relaxfix/ssp585_cesm2_ref_ref` | `SCENARIO=ssp585, GCM=CESM2-WACCM` | waits on prerequisites |
| `ho_relaxfix/ssp585_cesm2_high_ref` (optional) | as above plus `AIS3_PROJ_MELT=high` | waits on prerequisites |

Prerequisites: fix HO's 10-15% too-fast flow (worklog §5a/5c; ~20-40k SU), then rerun the HO
historical run from the relaxed state with `AIS3_HIST_TAG=_relaxfix` (~11k SU with annual
output). The projection step refuses to start until `AIS3_historical_1995_2019_relaxfix.nc`
reaches 2019.

Code: `projection_ssp` in [`config/ais_0.1.py`](../config/ais_0.1.py), launched by
[`config/launch_projection_ho.pbs`](../config/launch_projection_ho.pbs). Same options and names as
the SSA step; files are `AIS3_proj_ho_relaxfix_<exp>_*`. Differences from SSA: 10-yr chunks (the
96-cpu hugemem solve is capped at 24 h; ~82 min per model year), forcing built on the base layer
with the SSA interpolation and copied up each column (the HO base layer is the SSA mesh, checked
at run time, and the SSA climatology caches are reused), history files hold base-layer fields plus
surface speed. The driver job takes 24 cpu / 720 GB of a hugemem node instead of all 48 cpu /
1450 GB, which annual output makes possible.

Cost estimate, from the HO historical run's logs: ~395 SU per model year for the solve plus
~75 SU for the driver, so ~38k SU per run to 2100, ~76k SU for ctrl + ssp585 ref, ~115k SU with
the high-melt run; 2-4 weeks of wall clock per run (8-9 chunks, hugemem queue waits). Faster
flow under ssp585 could shorten the timestep and raise this 1.5-2x after ~2080.

## LARMIP (due 30 Oct 2026)

5+1 core runs, 200 yr from the 2019 state (2019.0-2219.0), SSA track only:
- control: the existing `ctrl` chain (runs past 2219);
- `larmip_r<1-5>_step`: CTRL forcing plus 8 m/yr extra melt (`md.basalforcings.melt_anomaly`) on
  the region's vertices; ISSM applies it only under fully floating elements, so it follows the
  grounding line;
- optional `larmip_r<1-5>_ramp`: 0 -> 8 m/yr linearly over 100 yr, then 8 m/yr for 100 yr.

Settings agreed for all runs: calving front fixed; no melt on partially floating elements;
sub-element grounding-line migration and friction; SMB fixed at the RACMO 1995-2014 mean.

```bash
cd config
for r in 1 2 3 4 5; do
  qsub -v AIS3_STEPS=projection_ssp_ssa,AIS3_PROJ_LARMIP_REGION=$r \
       -N larmip_r$r -o logs/proj_ssa_larmip_r${r}_step_2019.out -e logs/proj_ssa_larmip_r${r}_step_2019.err \
       launch_projection_ssa.pbs        # add AIS3_PROJ_LARMIP_MODE=ramp for the ramp set
done
python export_larmip.py [--ramp]       # -> docs/larmip/AIS3_ssa_larmip_sle.csv
```

Region mask: LARMIP-3's own `LARMIP3_sectors-1km.nc` (from https://www.pik-potsdam.de/~anders/larmip3/,
copied to `assets/larmip3/` with `LARMIP3_Infos.txt` and the 2 km / 4 km versions), the default.
Variable `z` on the ISMIP 8 km polar-stereographic extent; numbering checked against known shelves
on our mesh:

| Region | Sector | Floating area on the AIS3 mesh |
|---|---|---|
| 1 | Weddell (Filchner-Ronne) | 537,000 km^2 |
| 2 | East Antarctica (Amery ...) | 205,000 km^2 |
| 3 | Ross | 459,000 km^2 |
| 4 | Amundsen | 91,000 km^2 |
| 5 | Peninsula (Larsen C ...) | 92,000 km^2 |

Region 0 (no sector) holds ~1,100 km^2 of floating ice. The protocol text gives 8 m/yr only for
the ramp's final value; the core step runs use 8 m/yr too (to confirm with A. Levermann).
`models/larmip_region_mask_PROVISIONAL_amundsen_only.nc` predates the real mask and is superseded
(its region 3 is Amundsen, which is region 4 in LARMIP-3 numbering). Submission text: `docs/larmip/model_description.tex` and
`larmip_refs.bib`. Cost: ~6k SU per 200-yr run, ~36k SU core, ~66k SU with the ramps.
