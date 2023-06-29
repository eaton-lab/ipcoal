#!/usr/bin/env python

"""Calculate likelihood of a gene tree embedded in a species tree.

TODO
----
make faster version using numpy and jit.

References
----------
- Rannala and Yang (...) "Bayes Estimation of Species Divergence
  Times and Ancestral Population Sizes Using DNA Sequences From Multiple Loci
- Degnan and Salter (...) "..."
- ... (...) "STELLS-mod..."
"""

from typing import Dict, Sequence
import itertools
import numpy as np
import pandas as pd
from numba import njit
from loguru import logger
import toytree
import ipcoal


logger = logger.bind(name="ipcoal")


def get_msc_embedded_gene_tree_table(
    species_tree: toytree.ToyTree,
    gene_tree: toytree.ToyTree,
    imap: Dict,
) -> pd.DataFrame:
    """Return a DataFrame with intervals of coal and non-coal events.

    Each row in the dataframe represents an interval of time along
    the selected gene tree edge (idx), where a recomb event could break
    the current gene tree. It returns the information needed for
    calculating the prob it reattaches to another gene tree nedges given
    the length of each interval (dist), the nedges that exist then
    (nedges), and the species tree params (tau and neff).

    Parameters
    ----------
    species_tree: toytree.ToyTree
        Species tree with a "Ne" feature assigned to every node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_tree: toytree.ToyTree
        Gene tree that could be embedded in the species tree. Edge
        lengths are in units of generations.
    imap: Dict
        A dict mapping species tree node idxs to gene tree node idxs.
    """
    # store temporary results in a dict
    data = {}

    # get table of gene tree node heights (TODO: astype float/int?)
    gt_node_heights = gene_tree.get_node_data("height")

    # iterate over species tree nodes from tips to root
    # for nidx in range(species_tree.nnodes - 1)[::-1]: #.treenode.traverse("postorder"):
    for st_node in species_tree.treenode.traverse("postorder"):
        # st_node = species_tree[nidx]

        # get n nedges into the species tree interval, for tips it is
        # nsamples, for internal intervals get from child intervals.
        if st_node.is_leaf():
            gt_tips = set(imap[st_node.name])
            nedges_in = len(gt_tips)
        else:
            child_idxs = [i.idx for i in st_node.children]
            nedges_in = sum(data[idx]['nedges_out'] for idx in child_idxs)
            st_tips = st_node.get_leaf_names()
            gt_tips = set(itertools.chain(*[imap[i] for i in st_tips]))

        # get nodes that occur in the species tree interval (coalescences)
        mask_below = gt_node_heights > st_node.height + 0.0001
        if st_node.is_root():
            mask_above = gt_node_heights > 0
        else:
            mask_above = gt_node_heights < st_node.up.height
        nodes_in_time_slice = gt_node_heights[mask_below & mask_above]

        # get nodes in the appropriate species tree interval
        coal_events = []
        for gidx in nodes_in_time_slice.index:
            gt_node = gene_tree[gidx]
            tmp_tips = set(gt_node.get_leaf_names())
            if tmp_tips.issubset(gt_tips):
                coal_events.append(gt_node)

        # count nedges out of the interval
        nedges_out = nedges_in - len(coal_events)

        # sort coal events by height, and get height above last event
        # which is either st_node, or last gt_coal.
        coal_dists = []
        for node in sorted(coal_events, key=lambda x: x.height):
            if not coal_dists:
                coal_dists.append(node.height - st_node.height)
            else:
                coal_dists.append(node.height - st_node.height - sum(coal_dists))

        # store coalescent times in the interval
        data[st_node.idx] = {
            "dist": st_node.dist if st_node.up else np.inf,
            "neff": st_node.Ne,
            "nedges_in": nedges_in,
            "nedges_out": nedges_out,
            "coals": coal_dists,
        }
    data = pd.DataFrame(data).T.sort_index()
    return data


def get_censored_interval_log_prob(
    neff: float,
    nedges_in: int,
    nedges_out: int,
    interval_dist: float,
    coal_dists: np.ndarray,
) -> float:
    """Return the log probability of a censored species tree interval.

    Parameters
    ----------
    neff: float
        The diploid effective population size (Ne).
    nedges_in: int
        Number of gene tree edges at beginning of interval.
    nedges_in: int
        Number of gene tree edges at end of interval.
    interval_dist: float
        Length of the species tree interval in units of generations.
    coal_dists: np.ndarray
        Array of ordered coal *interval lengths* within a censored
        population interval, representing the dist from one event to
        the next. Events are gene tree coalescent times ordered from
        when the most edges existed to when the fewest existed.

    Notes
    -----
    Compared to the references below, we convert units to use the
    population Ne values and time in units of generations, as opposed
    popuation theta values and time in units of substitutions. This is
    because when working with genealogies alone we do not need the
    mutation rate. This conversion represents the coalescent rate as
    (1 / 2Ne) instead (2 / theta), since theta=4Neu, and we will assume
    that u=1. Similarly, time in units of E(substitutions/site)
    represents a measure of rate x time, where time is in generations,
    and rate (u) is (mutations / site / generation). To get in
    generations, we multiply by 1 / u, which has of effect when u=1.

    References
    ----------
    - Rannala and Yang (2003)
    - Rannala et al. (2020) book chapter.

    Example
    -------
    >>> ...
    """
    # coalescent rate in this interval
    rate = 1 / (2 * neff)

    # length of final subinterval, where no coalesce occurs
    remaining_time = interval_dist - np.sum(coal_dists)

    # The probability density of observing n-m coalescent events.
    # The opportunity for coalescence is described by the number of
    # ways that m lineages can be joined, times the length of time
    # over which m lineages existed. This 'opportunity' is treated as
    # an *exponential waiting time* with coalescence rate lambda, so:
    #         prob_density = rate * np.exp(-lambda * rate)
    # this prob density is 1 if no lineages coalesce in the interval.
    prob_coal = 1.
    for idx, nedges in enumerate(range(nedges_in, nedges_out, -1)):
        npairs = (nedges * (nedges - 1)) / 2
        time = coal_dists[idx]
        prob_coal *= rate * np.exp(-npairs * rate * time)

    # The probability that no coalescent events occur from the last
    # event until the end of the species tree interval. The more
    # edges that remain, and the longer the remaining distance, the
    # higher this probability is. It is 1 if nedges_out=1, bc there
    # is no one left to coalesce with in the interval. The species tree
    # root interval is always 1.
    prob_no_coal = 1.
    if nedges_out > 1:
        npairs_out = (nedges_out * (nedges_out - 1)) / 2
        prob_no_coal = np.exp(-npairs_out * rate * remaining_time)

    # multiply to get joint prob dist of the gt in the pop
    prob = prob_coal * prob_no_coal

    # return log positive results
    if prob > 0:
        return np.log(prob)
    return np.inf


def get_loglik_gene_tree_msc_from_table(table: pd.DataFrame):
    """Return the log probability of a gene tree given a species tree.

    Example
    -------
    >>>
    >>>
    >>>
    """
    # iterate over the species tree intervals to sum of logliks
    loglik = 0
    for interval in table.index:
        dat = table.loc[interval]

        # get log prob of censored coalescent
        args = (dat.neff, dat.nedges_in, dat.nedges_out, dat.dist, dat.coals)
        prob = get_censored_interval_log_prob(*args)
        loglik += prob

    # species tree prob is the product of population probs
    if loglik == np.inf:
        return loglik
    return -loglik


def get_loglik_gene_tree_msc(
    species_tree: toytree.ToyTree,
    gene_trees: Sequence[toytree.ToyTree],
    imap: Dict,
) -> float:
    """Return -log-likelihood of observing gene tree in a species tree.

    Parameters
    ----------
    species_tree: ToyTree
        Species tree with a "Ne" feature assigned to every Node, and
        edge lengths in units of generations. The tree can be non-
        ultrametric, representing differences in generation times.
    gene_trees: ToyTree, MultiTree, or Sequence[ToyTree]
        One or more gene trees that can be embedded in the species
        tree. Edge lengths are in units of generations.
    imap: Dict
        A dict mapping species tree tip Node names to lists of gene
        tree tip Node names.
    """
    if isinstance(gene_trees, toytree.ToyTree):
        gene_trees = [gene_trees]

    loglik = 0.
    for gtree in gene_trees:
        table = get_msc_embedded_gene_tree_table(species_tree, gtree, imap)
        loglik += get_loglik_gene_tree_msc_from_table(table)
    return loglik


##################################################################

@njit
def get_msc_loglik_from_embedding_table(table: np.ndarray) -> float:
    """Return the log probability of a censored species tree interval.

    Parameters
    ----------
    ...
    """
    ntrees = int(table[:, 6].max())
    loglik = np.zeros(ntrees, dtype=np.float64)

    # iterate over gtrees
    for gidx in range(ntrees):
        arr = table[table[:, 6] == gidx]

        # iterate over species tree intervals
        for sval in range(max(arr[:, 2])):
            narr = arr[arr[:, 2] == sval]

            # get coal rate in this interval
            rate = 1 / (2 * narr[0, 3])

            # get probability of each observed coalescence
            prob_coal = 1.
            for ridx in range(narr.shape[0] - 1):
                nedges = narr[ridx, 4]
                npairs = (nedges * (nedges - 1)) / 2
                dist = narr[ridx, 5]
                prob_coal *= rate * np.exp(-npairs * rate * dist)

            # get probability no coal in final interval
            prob_no_coal = 1.
            if narr[-1, 4] > 1:
                nedges = narr[-1, 4]
                dist = narr[-1, 5]
                npairs = (nedges * (nedges - 1)) / 2
                prob_no_coal = np.exp(-npairs * rate * dist)

            # multiply to get joint prob dist of the gt in the pop
            prob = prob_coal * prob_no_coal
            if prob > 0:
                loglik[gidx] = np.log(prob)
    return -loglik.sum()


if __name__ == "__main__":

    ipcoal.set_log_level("INFO")

    from ipcoal.msc.embedding import Embedding
    SPTREE = toytree.rtree.baltree(2, treeheight=1e6)
    MODEL = ipcoal.Model(SPTREE, Ne=200_000, nsamples=4, seed_trees=123)
    MODEL.sim_trees(1, 1e5)
    GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    IMAP = MODEL.get_imap_dict()
    data = Embedding(MODEL.tree, GENEALOGIES, IMAP)
    print(data.table)
    print(data.table.dtypes)
    print(get_msc_loglik_from_embedding_table(data.table.values))

    # # simulate genealogies
    # RECOMB = 1e-9
    # MUT = 1e-9
    # NEFF = 5e5
    # THETA = 4 * NEFF * MUT

    # # setup species tree model
    # SPTREE = toytree.rtree.unittree(ntips=3, treeheight=1e6, seed=123)
    # SPTREE = SPTREE.set_node_data("Ne", default=NEFF, data={0: 1e5})

    # # setup simulation
    # MODEL = ipcoal.Model(SPTREE, seed_trees=123, nsamples=5)
    # MODEL.sim_trees(10)
    # IMAP = MODEL.get_imap_dict()
    # GTREES = toytree.mtree(MODEL.df.genealogy)
    # # GTREE.draw(ts='c', height=400)

    # table = get_msc_embedded_gene_tree_table(SPTREE, GTREES[0], IMAP)
    # print(table)
    # print(get_loglik_gene_tree_msc_from_table(table))
    # print(get_loglik_gene_tree_msc(SPTREE, GTREES, IMAP))

    # TEST_VALUES = np.logspace(np.log10(NEFF) - 1, np.log10(NEFF) + 1, 19)
    # test_logliks = []
    # for idx in MODEL.df.index:
    #     gtree = toytree.tree(MODEL.df.genealogy[idx])
    #     table = get_msc_embedded_gene_tree_table(SPTREE, gtree, IMAP)

    #     logliks = []
    #     for ne in TEST_VALUES:
    #         table.neff = ne
    #         loglik = get_gene_tree_log_prob_msc(table)
    #         logliks.append(loglik)
    #     test_logliks.append(logliks)

    # logliks = np.array(test_logliks).sum(axis=0)

    # import toyplot
    # canvas, axes, mark = toyplot.plot(
    #     TEST_VALUES, logliks,
    #     xscale="log",
    #     height=300, width=400,
    # )
    # axes.vlines([NEFF])
    # toytree.utils.show(canvas)
