#!/usr/bin/env python

"""...

"""

import numpy as np
from numba import njit, prange
from ipcoal.smc.src.ms_smc_interval_prob import (
    _get_fast_sum_pb1, _get_fast_sum_pb2
)


@njit
def get_fast_prob_topo_unchanged_given_b(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    sidx: int,
    pidx: int,
) -> float:
    """Return probability of tree-change that does not change topology.

    Parameters
    ----------
    arr: np.ndarray
        Genealogy embedding table for a single genealogy.
    branch: int
        A selected focal branch selected by Node index.
    sibling: int
        Node index of the sibling of 'branch'.
    parent: int
        Node index of the parent of 'branch'.
    """
    # get all intervals on branch b
    benc = genc[:, bidx]
    idxs = benc.nonzero()[0]
    btab = gemb[idxs, :]

    # get intervals containing both b and b' (above t_m)
    senc = genc[:, sidx]
    midxs = (benc & senc).nonzero()[0]
    mtab = gemb[midxs, :]

    # get intervals containing either b or c
    penc = genc[:, pidx]
    fidxs = (benc | penc).nonzero()[0]
    ftab = gemb[fidxs, :]

    # get lower and upper bounds of this gtree edge
    t_lb, t_ub = btab[:, 0].min(), btab[:, 1].max()

    # get sum pb1 from intervals 0 to m
    pb1 = _get_fast_sum_pb1(btab, ftab, mtab)
    # logger.info(f"branch {branch}, sum-pb1={pb1:.3f}")

    # get sum pb2 from m to end of b
    pb2 = _get_fast_sum_pb2(btab, ftab, mtab)
    # logger.info(f"branch {branch}, sum-pb2={pb2:.3f}")
    return (1 / (t_ub - t_lb)) * (pb1 + pb2)


@njit
def get_fast_prob_topo_changed_given_b(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    sidx: int,
    pidx: int,
) -> float:
    """Return probability of tree-change that does not change topology.

    Parameters
    ----------
    arr: np.ndarray
        Genealogy embedding table for a single genealogy.
    branch: int
        A selected focal branch selected by Node index.
    sibling: int
        Node index of the sibling of 'branch'.
    parent: int
        Node index of the parent of 'branch'.
    """
    return 1 - get_fast_prob_topo_changed_given_b(gemb, genc, bidx, sidx, pidx)

####################################################################


@njit
def get_fast_prob_topo_unchanged(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
    rarr: np.ndarray,
) -> float:
    """Return probability that recombination causes a topology-change.

    """
    total_prob = 0
    for bidx, blen in enumerate(barr):
        # get relationships
        sidx = rarr[bidx, 1]
        pidx = rarr[bidx, 2]

        # get P(tree-unchanged | S, G, b) for every genealogy
        prob = get_fast_prob_topo_unchanged_given_b(
            gemb=gemb,
            genc=genc,
            bidx=bidx,
            sidx=sidx,
            pidx=pidx,
        )

        # get Prob scaled by the proportion of this branch on each tree.
        total_prob += (blen / sumlen) * prob
    return total_prob


@njit
def get_fast_prob_topo_changed(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
    rarr: np.ndarray,
) -> float:
    """Return probability that recombination causes a topology-change.

    """
    return 1 - get_fast_prob_topo_unchanged(gemb, genc, barr, sumlen, rarr)

####################################################################


@njit(parallel=True)
def get_fast_topo_unchanged_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
) -> np.ndarray:
    """return LAMBDA rate parameters for waiting distance prob density.

    """
    lambdas = np.zeros(emb.shape[0])
    for gidx in prange(emb.shape[0]):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        relate = rarr[gidx]
        # probability is a float in [0-1]
        prob_topo = get_fast_prob_topo_unchanged(gemb, genc, blens, sumlen, relate)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_topo * recombination_rate
    return lambdas


@njit(parallel=True)
def get_fast_topo_changed_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
) -> np.ndarray:
    """return LAMBDA rate parameters for waiting distance prob density.

    """
    lambdas = np.zeros(emb.shape[0])
    for gidx in prange(emb.shape[0]):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        relate = rarr[gidx]
        # probability is a float in [0-1]
        prob_topo = get_fast_prob_topo_changed(gemb, genc, blens, sumlen, relate)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_topo * recombination_rate
    return lambdas


if __name__ == "__main__":

    import toytree
    import ipcoal
    from ipcoal.smc.src.utils import (
        iter_topos_and_spans_from_model,
        iter_topos_from_trees,
        get_test_data,
        get_test_model,
    )
    from ipcoal.smc.src.embedding import TreeEmbedding

    # setup species tree model
    # sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    # model = ipcoal.Model(sptree, Ne=1e5, nsamples=2, seed_trees=123)
    # imap = model.get_imap_dict()
    # model.sim_trees(1, 1e5)

    sptree, gtree, imap = get_test_data()
    E = TreeEmbedding(sptree, gtree, imap, nproc=1)

    # model = ipcoal.Model(sptree)
    # model = get_test_model()
    # model.sim_trees(1, 1e5)
    # imap = model.get_imap_dict()
    # topos = [i[0] for i in iter_topos_and_spans_from_model(model, False)]
    # E = TreeEmbedding(model.tree, topos, imap)
    # print(E.rarr[0])
    # p = get_fast_prob_topo_unchanged_given_b(
    #     E.emb[0], E.enc[0], 0, 1, 8,
    # )
    # print(p)

    args = (E.emb[0], E.enc[0], E.barr[0], E.sarr[0], E.rarr[0])
    p = get_fast_prob_topo_unchanged(*args)
    print(p)
    print(1 - p)
    # p = get_fast_prob_topo_changed_given_b(*args)

    # get expected waiting distance parameters
    # print(get_fast_topo_changed_lambdas(table.earr, table.barr, table.sarr, table.rarr, 2e-9))
