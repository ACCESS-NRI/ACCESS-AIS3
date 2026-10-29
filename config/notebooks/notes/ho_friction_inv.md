### What this step does
Inverts the HO basal friction (Schoof `C`) with m1qn3:
- cost functions 101/103 (velocity misfit) on **surface** vertices
- cost function 501 (smoothness of C) on **base** vertices

Each submission runs `maxsteps=5` iterations and saves `AIS3_ho_friction_inv.nc`. The next
submission resumes from that file, copying `results.StressbalanceSolution.FrictionC` into
`md.friction.C`. Without a checkpoint it starts from `AIS3_thermal_steadystate.nc`.

### Status
- Paused after chunk 3. Chunk 4 failed five times (hangs at iteration 2, one SIGBUS;
  worklog §3.4, §4h, §4j).
- Grounded RMSE is 94–99 m/yr.
- The file on disk is chunk 3 with targeted smoothing. The unsmoothed chunk 3 is
  `AIS3_ho_friction_inv_chunk3_backup_presmoothing.nc`.
- This field was inverted with HO's old, too-soft B, from before the D2 rescale (worklog
  §5c/§5d).

### Before you run
- Copy the current checkpoint to a chunk-tagged name first. A completed chunk overwrites it,
  and a walltime-killed chunk saves nothing.
- The inner solve runs 96 hugemem cpus, which is capped at 24 h.

### What to check afterwards
- The m1qn3 cost table in `execution/AIS3_ho_friction_inv/*.outlog`:
  - Iteration 1 should continue the previous chunk's final cost. If it shows `3.0065e7`
    again, the chunk restarted from scratch (the resume bug).
  - The cf101/cf103 contributions should not be 0/0. If they are, the misfit mask is wrong.
- The printed grounded (surface) RMSE.

### Known pitfalls
- After a restart, iteration 2 can jump 12–19× in cost (m1qn3 loses its curvature estimate
  between submissions). `dfmin_frac=0.3` targets this.
- Tightening `restol` to 0.001 made single iterations run past 21 h.
