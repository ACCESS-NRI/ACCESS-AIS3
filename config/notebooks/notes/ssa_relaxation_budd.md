### What this step does
A 20-year SSA transient with zero SMB and zero melt, time step 0.05 yr. It saves
`AIS3_SSA_relaxed.nc`.

**The relaxed geometry is deliberately not carried forward.** The saved file keeps the
inverted geometry in `md.geometry`; the relaxation itself is only in `md.results`. Later SSA
steps use the file as their model shell. This is by design (worklog §4z correction): only the
HO track adopts its relaxed state.

### Before you run
- `AIS3_SSA_inverted.nc` must exist, with its thickness already restored.
- It takes about 9.5 h. Archive the existing output first (`mv -n`).

### What to check afterwards
- Mean and max |dH|. With the corrected thickness: mean 26.19 m, max about 1.5 km.

### Known pitfalls
- The 400-step history makes the file large. Later steps strip `TransientSolution` right
  after loading it to avoid running out of memory.
