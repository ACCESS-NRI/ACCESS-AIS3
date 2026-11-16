"""Collapse run, deterministic NaN at the 2089 sub-solve (worklog 6.15): how is the ice held
together after that year's collapse? Pieces = fully-ice elements joined across shared edges;
anchor elements = fully grounded with friction coefficient > 0 at all vertices (as in
apply_collapse). Compares the 2075 checkpoint with the 2089 one (state the failing solve started
from). Read-only."""
import numpy as np, netCDF4
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
M = '/g/data/au88/jh7060/ACCESS-AIS3/models'


def pieces(path):
    d = netCDF4.Dataset(path)
    x = np.asarray(d['mesh']['x'][:]).ravel(); y = np.asarray(d['mesh']['y'][:]).ravel()
    el = np.asarray(d['mesh']['elements'][:]).astype(int) - 1; nv = x.size
    il = np.asarray(d['mask']['ice_levelset'][:]).ravel(); ol = np.asarray(d['mask']['ocean_levelset'][:]).ravel()
    fc = np.asarray(d['friction']['coefficient'][:]).ravel(); H = np.asarray(d['geometry']['thickness'][:]).ravel()
    v = np.asarray(d['initialization']['vel'][:]).ravel()
    ice = il < 0
    ie = np.where(np.all(ice[el], axis=1))[0]; e = el[ie]
    ed = np.sort(np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]]), axis=1)
    key = ed[:, 0].astype(np.int64) * nv + ed[:, 1]
    own = np.tile(np.arange(ie.size), 3); o = np.argsort(key, kind='stable'); same = key[o][1:] == key[o][:-1]
    n, lab = connected_components(coo_matrix((np.ones(same.sum()), (own[o][:-1][same], own[o][1:][same])),
                                             shape=(ie.size, ie.size)), directed=False)
    anch = np.all((ol[e] > 0) & (fc[e] > 0), axis=1)
    size = np.bincount(lab); nanc = np.bincount(lab, weights=anch.astype(float)).astype(int)
    nfl = np.bincount(lab, weights=np.all(ol[e] < 0, axis=1).astype(float)).astype(int)
    # element-edge count per element inside its piece: elements with only one ice neighbour = slivers/tips
    nb = np.bincount(np.r_[own[o][:-1][same], own[o][1:][same]], minlength=ie.size)
    infull = np.zeros(nv, bool); infull[e.ravel()] = True
    lone = ice & ~infull
    print(f'{path.split("/")[-1]}: pieces {n}; largest {size.max()} el; no anchor {(nanc == 0).sum()} pieces ({size[nanc == 0].sum()} el); '
          f'1-5 anchor el {((nanc > 0) & (nanc <= 5)).sum()} pieces ({size[(nanc > 0) & (nanc <= 5)].sum()} el, floating {nfl[(nanc > 0) & (nanc <= 5)].sum()}); '
          f'ice elements with <=1 ice neighbour {(nb <= 1).sum()}; ice vertices in no full-ice element {lone.sum()} (floating {(lone & (ol < 0)).sum()}); '
          f'max init vel {v[ice].max():.0f} m/yr', flush=True)
    return dict(x=x, y=y, e=e, lab=lab, size=size, nanc=nanc, nfl=nfl, nb=nb, ol=ol, v=v, H=H)


a = pieces(f'{M}/AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2075.nc')
b = pieces(f'{M}/AIS3_proj_ssa_ssp585_cesm2_ref_ref_collapse_2089.nc')
for tag, r in (('2075', a), ('2089', b)):
    w = np.where((r['nanc'] > 0) & (r['nanc'] <= 5) & (r['nfl'] > 0))[0]
    w = w[np.argsort(-r['size'][w])][:10]
    print(f'{tag}: weakly anchored pieces holding floating ice (largest first): elements, anchor el, floating el, centre x,y km, max vel')
    for k in w:
        c = np.unique(r['e'][r['lab'] == k].ravel())
        print(f'   {r["size"][k]:6d} {r["nanc"][k]:3d} {r["nfl"][k]:6d}   {r["x"][c].mean() / 1e3:8.1f} {r["y"][c].mean() / 1e3:8.1f}  {r["v"][c].max():8.0f}')
# where are the fastest ice vertices in the 2089 starting state (velocity from the 2088 sub-solve's end)?
ice_v = np.unique(b['e'].ravel())
top = ice_v[np.argsort(-b['v'][ice_v])][:10]
print('2089 fastest ice vertices: x,y km, vel, H, grounded')
for i in top:
    print(f'   {b["x"][i] / 1e3:8.1f} {b["y"][i] / 1e3:8.1f} {b["v"][i]:9.0f} {b["H"][i]:7.1f} {b["ol"][i] > 0}')
print('DONE')
