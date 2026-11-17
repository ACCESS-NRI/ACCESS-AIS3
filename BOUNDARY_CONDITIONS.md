# Boundary Conditions — AIS3 Workflow

## Parameterisation (full continental model)

The mesh includes an ocean buffer surrounding the entire continent. All `spcvx`/`spcvy`/`spcvz` fields are initialised to `NaN` (Neumann). The code then attempts to impose Dirichlet BCs only at ice vertices coinciding with the mesh boundary (`vertexonboundary`), using observed initialisation velocities. With the ocean buffer in place, no ice vertices lie on the mesh boundary, so this step sets no Dirichlet BCs on ice. The calving front is a material interface interior to the mesh (flanked by ice on one side and ocean buffer elements on the other) and receives Neumann BCs by default. The outer ocean mesh boundary nominally gets Dirichlet constraints, but since those vertices carry near-zero initialisation velocities this is inconsequential.

## Rheology inversion — floating ice extraction

The extraction mask is vertex-based: `(ocean_levelset < 0) & (ice_levelset < 0)`. This selects strictly interior floating-ice vertices, so the calving front (where `ice_levelset ≈ 0`) becomes the boundary of the extracted subdomain. By default, `extract()` imposes Dirichlet constraints on all vertices of the newly-created boundary, using the model's existing observed velocity fields. Consequently, the calving front receives Dirichlet BCs (observed velocities) rather than Neumann in the extracted model. This does not materially affect the rheology inversion because the interior of the ice shelf has sufficient observational signal that a velocity constraint at the calving-front strip does not distort the recovered B field significantly.

## Friction inversion — `AIS3_ssa_friction_inv_sensit` (current approach)

The extraction mask is element-based: the vertex-level `ice_levelset` is interpolated to elements (`vertex_to_element`), and elements satisfying `ice_levelset_elements < 1` are extracted. The threshold `< 1` (rather than `< 0`) includes ice-front elements — those straddling the calving-front contour where the element-averaged levelset is between 0 and 1. Including these elements means the calving front itself is an interior interface in the extracted mesh, and naturally receives Neumann BCs. The outer seaward edge of the ice-front element strip becomes the extracted-mesh boundary and would nominally receive Dirichlet constraints from `extract()`.

After extraction, Dirichlet BCs are explicitly cleared on floating ice-front vertices:

```python
iceFront = (mds.mask.ice_levelset >= 0) & (mds.mask.ocean_levelset < 0)
mds.stressbalance.spcvx[iceFront] = np.nan
mds.stressbalance.spcvy[iceFront] = np.nan
mds.stressbalance.spcvz[iceFront] = np.nan
```

This targets vertices on the outer edge of floating ice-front elements (where `ice_levelset >= 0` but `ocean_levelset < 0`), converting those from Dirichlet back to Neumann. Grounded ice-front vertices retain the Dirichlet BCs inherited from `extract()`. The net result is: Neumann on the floating calving front (physically correct — the boundary condition there is ocean back-pressure, not a prescribed velocity), and Dirichlet on the grounded ice boundary and lateral margins.

## Why Neumann matters more for friction than rheology

The friction coefficient C governs momentum transfer at the bed beneath grounded and lightly-floating ice. The friction inversion is sensitive to the velocity field near the grounding line, which is directly influenced by the calving-front BC. Imposing an incorrect Dirichlet constraint at the calving front propagates a velocity error inward through the momentum equation, biasing C near the grounding line. The rheology field B, being an ice-column material property, is less sensitive to this because the floating-ice velocity field is dominated by lateral shear and longitudinal stress gradients across the full shelf width, not the calving-front boundary value alone.

## Alternative approaches tested

### `AIS3_ssa_friction_inv_sensit_2`

Replaced the manual BC clearing with `pyissm.model.bc.marine_ice_sheet_bc()`. This function identifies the ice front using a different connectivity criterion that does not guarantee a contiguous ring of elements along the calving margin. The result is inconsistent Neumann/Dirichlet patches on the ice front, producing a poorly-posed problem — confirmed by solver residual divergence. This behaviour is consistent with MATLAB ISSM (i.e. the function is working as designed; it is simply not suited to this mesh and domain).

### `AIS3_ssa_friction_inv_sensit_3`

Excluded ice-front elements entirely by using `ice_levelset < 0` at element level, so the calving front became the extracted-mesh boundary and received uniform Dirichlet BCs from `extract()`. The model converges, but the friction inversion cannot adjust C near the grounding line to compensate for any calving-front velocity mismatch introduced by the Dirichlet constraint. This formulation is numerically stable but physically inferior to `AIS3_ssa_friction_inv_sensit`.
