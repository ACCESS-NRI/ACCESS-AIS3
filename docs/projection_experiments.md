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
| Melt | ref = gamma_0 300 + refit deltaT (`AIS3_melt_final_ssa.nc`); high = gamma_0 15000 + its refit deltaT (`AIS3_PROJ_MELT=high`, candidate g7); no low member | Decided 2026-09-29 from J3 (worklog §6.1): 300 is kept as reference (ISMIP7-style weighting still picks it) but gives ~4% of the ocean models' warming response; 15000 is the J1+J2+J3 95th percentile (~33%). Nothing converged lies below 300. |
| Shelf collapse | Optional, ISMIP7 fracture path C (`AIS3_PROJ_COLLAPSE=1`), see below | Off in all runs so far. |
| Not included | SMB elevation feedback (`dacabfdz`), calving (front fixed), GIA, thermal evolution | No implementation yet. |

Anomalies are annual (placed at mid-year), zero where a source grid has no data, and a GCM's
last year is held if a run goes past it (ssp585 TF ends 2299).

## Available ISMIP7 forcing

| GCM | Scenarios | To 2300? |
|---|---|---|
| CESM2-WACCM | ctrl, ssp126, ssp245 (folder present, not checked), ssp534-over, ssp585 | yes (ssp370: 2100) |
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
| 3 | `<scenario>_<gcm>_ref_ref_collapse` | `AIS3_PROJ_COLLAPSE=1` | `ssp585_cesm2_ref_ref_collapse` launched 2026-10-13 (driver 181109462); smoke tests in worklog 6.12 |
| 3 | GIA, sliding law | not implemented | — |

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

Cost: about 12.6 min of the 48-cpu solve per model year (historical run), so ~60 h and ~3k SU
of compute per run to 2300, plus the 8-cpu driver jobs. Stronger forcing may slow the solve.

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
