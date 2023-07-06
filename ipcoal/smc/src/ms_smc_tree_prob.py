#!/usr/bin/env python

"""Tree-unchanged and tree-changed probabilities from Embeddings.

This module contains functions optimized for calculating tree-change
and tree-unchanged probabilities and waiting distances from data
that has been pre-organized into arrays using a TreeEmbedding class
object. The functions are jit-compiled and so run much much faster
than the didactic functions that take ToyTrees as input.

Note: Ne in TreeEmbedding emb arrays are stored as 2 * diploid Ne.
"""

import numpy as np
from loguru import logger
from numba import njit, prange
from ipcoal.smc.src.ms_smc_interval_prob import _get_fast_pij

logger = logger.bind(name="ipcoal")

# __all__ = [
#     "get_tree_change_probability",
#     "get_tree_change_probability_given_b",
#     "get_tree_change_probability_given_b_and_t",
#     "get_tree_change_waiting_distance",
#     "get_tree_change_rate",
#     "get_tree_change_rvs",
# ]


@njit
def get_fast_prob_tree_unchanged_given_b_and_tr(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
    time: float,
) -> float:
    """Return prob tree-unchanged given recomb on branch b at time t.

    This function is mainly for didactic purposes.

    Note: this function assumes garr[:, 3] is (2 * diploid Ne), which
    is how Ne is stored in the TreeEmbedding array. When using this
    function with a normal table from `get_genealogy_embedding_table()`
    the garr[:, 3] is (1 * diploid Ne) and should be multiplied by 2.
    This is automatically done in the functions in `ipcoal.smc` that
    do not have `_fast_` in their names.

    Parameters
    ----------
    garr: np.ndarray
        A genealogy embedding table for a single genealogy.
    bidx: int
        Index of a genealogy branch on which recombination occurs.
    """
    # subselect array intervals for this genealogy branch
    idxs = np.nonzero(genc[:, bidx])[0]
    arr = gemb[idxs, :]

    # get interval containing time tr
    tidx = np.nonzero((arr[:, 0] <= time) & (arr[:, 1] >= time))[0]
    if not tidx.size:
        raise ValueError(f"No interval exists on branch {bidx} at time {time}.")
    tidx = tidx[0]

    # (nedges / neff) * time
    inner = (arr[tidx, 4] / arr[tidx, 3]) * time
    inner = np.exp(inner) if inner < 100 else 1e15

    # (1 / nedges) + pij * inner
    term1 = (1 / arr[tidx, 4]) + _get_fast_pij(arr, tidx, tidx) * inner

    # iterate over all intervals from idx to end of b and get pij
    term2 = 0
    for jdx in range(tidx + 1, arr.shape[0]):
        term2 += _get_fast_pij(arr, tidx, jdx) * inner
    return term1 + term2

#####################################################################


@njit
def get_fast_prob_tree_unchanged_given_b(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
) -> float:
    """Return prob tree-unchanged given recomb on branch b.

    Parameters
    ----------
    gemb: np.ndarray
        A genealogy embedding table for a single genealogy.
    genc: np.ndarray
        A Node encoding table for a single genealogy.
    bidx: int
        Index of a genealogy branch on which recombination occurs.
    """
    # subselect array intervals for this genealogy branch
    idxs = np.nonzero(genc[:, bidx])[0]
    arr = gemb[idxs, :]

    # get top and bottom times on branch b
    tbl = arr[:, 0].min()
    tbu = arr[:, 1].max()

    # sum over the intervals on b where recomb could occur
    sumval = 0
    for idx in range(arr.shape[0]):
        term1 = (1 / arr[idx, 4]) * arr[idx, 5]
        term2_outer = arr[idx, 3] / arr[idx, 4]

        # Avoid overflow when inner value here is too large. Simply
        # setting it to a very large value seems asymptotically OK.
        estop = (arr[idx, 4] / arr[idx, 3]) * arr[idx, 1]
        estart = (arr[idx, 4] / arr[idx, 3]) * arr[idx, 0]
        if estop > 100:
            term2_inner = 1e15
            # logger.warning("overflow")  # no-jit
        else:
            term2_inner = np.exp(estop) - np.exp(estart)

        # pij component
        term3 = 0
        for jdx in range(idx, arr.shape[0]):
            term3 += _get_fast_pij(arr, idx, jdx)
        sumval += term1 + (term2_inner * term2_outer * term3)
    return (1 / (tbu - tbl)) * sumval


def get_fast_prob_tree_changed_given_b(
    gemb: np.ndarray,
    genc: np.ndarray,
    bidx: int,
) -> float:
    """Return prob tree-changed given recomb on branch b.

    Parameters
    ----------
    garr: np.ndarray
        A genealogy embedding table for a single genealogy.
    bidx: int
        Index of a genealogy branch on which recombination occurs.
    """
    return 1 - get_fast_prob_tree_unchanged_given_b(gemb, genc, bidx)

#####################################################################


@njit
def get_fast_prob_tree_unchanged(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.ndarray,
    sumlen: float,
) -> float:
    """Return probability recombination does not cause tree-change.

    Returns the probability that recombination occurring on this
    genealogy embedded in this parameterized species tree causes a
    tree change, under the MS-SMC'. A tree-change is defined as the
    opposite of a no-change event, and includes any change to
    coalescent times (whether or not it changes the topology).

    This is used within `get_fast_waiting_distance_to_tree_change_rates`

    This probability is 1 - P(no-change | S,G), where S is the
    species tree and G is the genealogy.

    Parameters
    ----------
    garr: np.ndarray
        A genealogy embedding table for a single genealogy.
    barr: np.ndarray
        Branch lengths for each branch on the genealogy.
    sumlen: int
        Sum of branch lengths on the genealogy.
    """
    # traverse over all edges of the genealogy
    total_prob = 0
    for bidx, blen in enumerate(barr):
        # get P(tree-unchanged | S, G, b)
        prob = get_fast_prob_tree_unchanged_given_b(gemb, genc, bidx=bidx)
        # contribute to total probability normalized by prop edge len
        total_prob += (blen / sumlen) * prob
    return total_prob


@njit
def get_fast_prob_tree_changed(
    gemb: np.ndarray,
    genc: np.ndarray,
    barr: np.array,
    sumlen: float,
) -> float:
    """Return probability recombination causes a tree-change.

    A tree-change is defined as the opposite of a tree-unchanged event
    (no-change event), and includes any change to coalescent times
    (whether or not it changes the topology).

    This probability is 1 - P(no-change | S,G), where S is the
    species tree and G is the genealogy.

    Parameters
    ----------
    garr: np.ndarray
        A genealogy embedding table for a single genealogy.
    barr:
        Array of branch lengths for each branch on the genealogy.
    sumlen:
        Sum branch lengths of the genealogy.
    """
    return 1 - get_fast_prob_tree_unchanged(gemb, genc, barr, sumlen)

#####################################################################


@njit(parallel=True)
def get_fast_tree_changed_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
) -> np.ndarray:
    """Return LAMBDA rate parameters for waiting distance prob. density.

    Note: earr stores neff as 2Ne. No NaN allowed in earr. The arrays
    used as input here come from a TreeEmbedding or TopologyEmbedding
    object.

    Parameters
    ----------
    earr: np.ndarray
        Embedding array of (ngenealogies * nintervals, 7 + nnodes)
        containing one or more genealogy embedding arrays each with
        a unique genealogy index (gidx) in column 6.
    barr: np.ndarray
        Array of shape (ngenealogies, nnodes - 1) containing branch
        lengths for each branch on each genealogy.
    sarr: np.ndarray
        Array of (ngenealogies,) w/ sum branch lengths of each tree.
    recombination_rate: float
        The per-site per-generation recombination rate.
    """
    lambdas = np.zeros(emb.shape[0], dtype=np.float64)

    # use numba parallel to iterate over genealogies
    # pylint-disable: not-an-iterable
    for gidx in prange(emb.shape[0]):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        # probability is a float in [0-1]
        prob_tree = get_fast_prob_tree_changed(gemb, genc, blens, sumlen)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_tree * recombination_rate
    return lambdas


@njit(parallel=True)
def get_fast_tree_unchanged_lambdas(
    emb: np.ndarray,
    enc: np.ndarray,
    barr: np.ndarray,
    sarr: np.ndarray,
    rarr: np.ndarray,  # placeholder
    recombination_rate: float,
) -> np.ndarray:
    """Return LAMBDA rate parameters for waiting distance prob. density.

    Note: earr stores neff as 2Ne. No NaN allowed in earr. The arrays
    used as input here come from a TreeEmbedding or TopologyEmbedding
    object.

    Parameters
    ----------
    earr: np.ndarray
        Embedding array of (ngenealogies * nintervals, 7 + nnodes)
        containing one or more genealogy embedding arrays each with
        a unique genealogy index (gidx) in column 6.
    barr: np.ndarray
        Array of shape (ngenealogies, nnodes - 1) containing branch
        lengths for each branch on each genealogy.
    sarr: np.ndarray
        Array of (ngenealogies,) w/ sum branch lengths of each tree.
    recombination_rate: float
        The per-site per-generation recombination rate.
    """
    lambdas = np.zeros(emb.shape[0], dtype=np.float64)

    # use numba parallel to iterate over genealogies
    # pylint-disable: not-an-iterable
    for gidx in prange(emb.shape[0]):
        gemb = emb[gidx]
        genc = enc[gidx]
        blens = barr[gidx]
        sumlen = sarr[gidx]
        # probability is a float in [0-1]
        prob_tree = get_fast_prob_tree_unchanged(gemb, genc, blens, sumlen)
        # lambda is a rate > 0
        lambdas[gidx] = sumlen * prob_tree * recombination_rate
    return lambdas

#####################################################################


if __name__ == "__main__":

    import ipcoal
    import pandas as pd
    from ipcoal.smc.src.embedding import TreeEmbedding
    from ipcoal.smc.src.utils import get_test_data

    ipcoal.set_log_level("WARNING")
    pd.options.display.max_columns = 14
    pd.options.display.width = 1000

    # SPTREE, GTREE, IMAP = get_test_data()

    # # Select a branch to plot and get its relations
    # BIDX = 2
    # BRANCH = GTREE[BIDX]
    # SIDX = BRANCH.get_sisters()[0].idx
    # PIDX = BRANCH.up.idx

    # # Get genealogy embedding table
    # ETABLE = ipcoal.msc.get_genealogy_embedding_table(SPTREE, GTREE, IMAP, encode=False)
    # print(f"Full genealogy embedding table\n{ETABLE}\n")

    SPTREE, GTREE, IMAP = get_test_data(1, 1000)
    E = TreeEmbedding(SPTREE, GTREE, IMAP)

    p_no = get_fast_prob_tree_unchanged(E.emb[0], E.enc[0], E.barr[0], E.sarr[0])
    print(f"Probability of no-change\n{p_no:.3f}\n")

    p_tree = get_fast_prob_tree_changed(E.emb[0], E.enc[0], E.barr[0], E.sarr[0])
    print(f"Probability of tree-change\n{p_tree:.3f}\n")

    lambdas_ = get_fast_tree_changed_lambdas(E.emb, E.enc, E.barr, E.sarr, None, 2e-9)
    print(lambdas_)

    # # p_tree = get_probability_tree_change(SPTREE, GTREE, IMAP)
    # # p_topo = get_probability_topology_change(SPTREE, GTREE, IMAP)

    # # print(f"Probability of tree-change\n{p_tree:.3f}\n")
    # # print(f"Probability of topology-change\n{p_topo:.3f}\n")





    # # setup species tree model
    # sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    # model = ipcoal.Model(sptree, Ne=1e5, nsamples=2, seed_trees=123)
    # imap = model.get_imap_dict()

    # # simulate ARGs
    # model.sim_trees(100)
    # gtrees = model.df.genealogy

    # # get embedding tables
    # table = TreeEmbedding(model.tree, model.df.genealogy, imap, table=True)
    # # print(table.table.iloc[:, :8])
    # # print(table.barr.shape, table.genealogies[0].nnodes)

    # # get expected waiting distance parameters
    # print(get_fast_tree_changed_lambda(table.earr, table.barr, table.sarr, 2e-9))

    # get observed waiting distances from sims


    # print likelihood of genealogies




    # bidx = 0
    # blen = table.barr[0][0]
    # print(np.nonzero(table.earr[:, 7 + bidx])[0])

    # p = get_prob_tree_unchanged_given_b(table.earr)
    # print(p)
