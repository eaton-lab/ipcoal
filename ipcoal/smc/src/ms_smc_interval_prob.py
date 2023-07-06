#!/usr/bin/env python

"""Much faster tree-change probs.

This contains modified versions of the functions in ms_smc.py that
are written to be faster (using jit compilation) and to reduce some
redundancy that would arise when the same functions are run
repeatedly without changing the
"""

import numpy as np
from loguru import logger
from numba import njit

logger = logger.bind(name="ipcoal")

################################################################
################################################################
# Piece-wise constants function
################################################################
################################################################


@njit
def _get_fast_pij(itab: np.ndarray, idx: int, jdx: int) -> float:
    """Return pij value for two intervals.

    This returns a value associated with an integration over the
    possible intervals that a detached subtree could re-attach to
    if it was detached in interval idx and could reconnect in any
    intervals between idx and jdx. The idx interval is on branch
    b (intervals in itab), whereas the jdx interval can occur on
    branches b, b', or c (same, sibling or parent).

    Note
    ----
    This is not really intended to be called directly, since the
    table that is entered needs to be specifically constructed. See
    `get_probability_topology_unchanged_given_b_and_tr` for examples.

    Parameters
    ----------
    table
        Intervals on one or more branches betwen intervals idx and jdx.
        This table should include ONLY these intervals.
    idx:
        Index of an interval in itable.
    jdx:
        Index of an interval in jtable.
    """
    # pii
    if idx == jdx:
        term1 = -(1 / itab[idx, 4])
        term2 = np.exp(-(itab[idx, 4] / itab[idx, 3]) * itab[idx, 1])
        return term1 * term2

    # ignore jdx < idx (speed hack so we don't need to trim tables below t_r)
    if jdx < idx:
        return 0

    # involves connections to jdx interval
    term1 = 1 / itab[jdx, 4]
    term2 = (1 - np.exp(-(itab[jdx, 4] / itab[jdx, 3]) * itab[jdx, 5]))

    # involves connections to idx interval
    term3_inner_a = -(itab[idx, 4] / (itab[idx, 3])) * itab[idx, 1]

    # involves connections to edges BETWEEN idx and jdx (not including idx or jdx)
    term3_inner_b = 0
    for qdx in range(idx + 1, jdx):
        term3_inner_b += ((itab[qdx, 4] / itab[qdx, 3]) * itab[qdx, 5])
    term3 = np.exp(term3_inner_a - term3_inner_b)
    return term1 * term2 * term3


@njit
def _get_fast_sum_pb1(btab: np.ndarray, ftab: np.ndarray, mtab: np.ndarray) -> float:
    """Return value for the $p_{b,1}$ variable.

    Parameters
    ----------
    btab: np.ndarray
        Array of all intervals on branch b.
    mtab: np.ndarray
        Array of a subset of btab, including intervals on branch b
        shared with b' (its sister lineage). This potentially excludes
        intervals on b below a species divergence separating b and b'.
    ftab: np.ndarray
        Array of a superset of btab, including all intervals on branch
        b or on its parent branch, c.
    """
    pbval = 0

    # iterate over all intervals from 0 to t_m
    t_m = mtab[:, 0].min()
    for idx in range(btab.shape[0]):
        row = btab[idx]

        # if idx interval start is >=tm it doesn't affect pb1 (its in pb2)
        if row[0] >= t_m:
            continue

        # get first term
        estop = (row[4] / row[3]) * row[1]
        estart = (row[4] / row[3]) * row[0]
        if estop > 100:
            first_term = 1e15
        else:
            first_term = row[3] * (np.exp(estop) - np.exp(estart))

        # pij across bc (from this i on b to each j on bc)
        sum1 = 0
        for jidx in range(ftab.shape[0]):
            sum1 += _get_fast_pij(ftab, idx, jidx)
        # logger.info(f"sum1={sum1}")

        # pij across b > tm (from this i on b to each j on b above tm)
        sum2 = 0
        for jidx in range(mtab.shape[0]):
            # which row in mtab corresponds to idx in btab
            midx = np.argmax(btab[:, 0] == mtab[jidx, 0])
            sum2 += _get_fast_pij(btab, idx, midx)
        # logger.info(f"sum2={sum2}")

        second_term = sum1 + sum2
        pbval += (1 / row[4]) * (row[5] + (first_term * second_term))
    return pbval


@njit
def _get_fast_sum_pb2(btab: np.ndarray, ftab: np.ndarray, mtab: np.ndarray) -> float:
    """Return value for the $p_{b,2}$ variable.

    Parameters
    ----------
    btab: np.ndarray
        Array of all intervals on branch b.
    mtab: np.ndarray
        Array of a subset of btab, including intervals on branch b
        shared with b' (its sister lineage). This potentially excludes
        intervals on b below a species divergence separating b and b'.
    ftab: np.ndarray
        Array of a superset of btab, including all intervals on branch
        b or on its parent branch, c.
    """
    pbval = 0

    # iterate over all intervals from m to bu
    for idx in range(mtab.shape[0]):
        row = mtab[idx]

        # get first term
        estop = (row[4] / row[3]) * row[1]
        estart = (row[4] / row[3]) * row[0]
        if estop > 100:
            first_term = 1e15
        else:
            first_term = row[3] * (np.exp(estop) - np.exp(estart))

        # pij across intervals on b
        sum1 = 0
        for jidx in range(idx, mtab.shape[0]):
            sum1 += _get_fast_pij(mtab, idx, jidx)

        # pij across intervals on c
        sum2 = 0
        for pidx in range(ftab.shape[0]):
            if pidx >= btab.shape[0]:
                midx = np.argmax(ftab[:, 0] == mtab[idx, 0])
                sum2 += _get_fast_pij(ftab, midx, pidx)

        second_term = (2 * sum1) + sum2
        pbval += (1 / row[4]) * ((2 * row[5]) + (first_term * second_term))
    return pbval


if __name__ == "__main__":

    import toytree
    import ipcoal
    from ipcoal.smc.src.embedding import TreeEmbedding, TopologyEmbedding

    # generate data
    sptree = toytree.rtree.imbtree(4, treeheight=1e6)
    model = ipcoal.Model(sptree, Ne=1e5, nsamples=2)
    model.sim_trees(100)
    imap = model.get_imap_dict()

    # get embeddings
    table = TreeEmbedding(model.tree, model.df.genealogy, imap)
    print(table.table)
