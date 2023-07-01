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
import numpy as np
from numba import njit
from loguru import logger
import toytree
from ipcoal.msc.embedding import get_genealogy_embedding_table

logger = logger.bind(name="ipcoal")

__all__ = ["get_msc_loglik", "get_msc_loglik_from_embedding_table"]


def get_msc_loglik(
    species_tree: toytree.ToyTree,
    gene_trees: Sequence[toytree.ToyTree],
    imap: Dict,
) -> float:
    """Return sum -loglik of genealogies embedded in a species tree.

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
    if isinstance(gene_trees, (toytree.ToyTree, str)):
        gene_trees = [gene_trees]
    if not isinstance(gene_trees[0], toytree.ToyTree):
        gene_trees = toytree.mtree(gene_trees).treelist

    etable = get_genealogy_embedding_table(species_tree, gene_trees, imap, df=False)
    loglik = get_msc_loglik_from_embedding_table(etable)
    return loglik


# @njit
def get_msc_loglik_from_embedding_table(table: np.ndarray) -> float:
    """Return sum -loglik of genealogies embedded in a species tree.

    Parameters
    ----------
    table: np.ndarray
        A genealogy embedding table as an np.ndarray generated from
        `ipcoal.smc.get_genealogy_embedding_table()` run with the
        arg `df=False`.

    Examples
    --------
    >>> args = (sptree, gtrees, imap, False)
    >>> etable = ipcoal.msc.get_genealogy_embedding_table(*args)
    >>> loglik = get_msc_loglik_from_embedding_table(etable)
    """
    ntrees = int(table[:, 6].max())
    logliks = np.zeros(ntrees, dtype=np.float64)

    # iterate over gtrees
    for gidx in range(ntrees):
        arr = table[table[:, 6] == gidx]

        # iterate over species tree intervals
        loglik = 0.
        for sval in range(int(max(arr[:, 2]) + 1)):
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
                loglik += np.log(prob)
            else:
                loglik += np.inf
        logliks[gidx] = loglik
    return -logliks.sum()


def test_kingman(neff: float = 1e5, nsamples: int = 10, ntrees: int = 500):
    """Return a plot of the likelihood of Ne in a single population.
    """
    import toyplot
    import toytree

    # get (sptree, gtrees, imap)
    model = ipcoal.Model(None, Ne=neff, nsamples=nsamples)
    model.sim_trees(ntrees)
    imap = model.get_imap_dict()

    # get embedding table
    etable = get_genealogy_embedding_table(model.tree, model.df.genealogy, imap, df=False)

    # get loglik across a range of test values
    test_values = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1, 20)
    logliks = []
    for val in test_values:
        etable[:, 3] = val
        loglik = get_msc_loglik_from_embedding_table(etable)
        logliks.append(loglik)

    canvas, axes, mark = toyplot.plot(
        test_values, logliks,
        xscale="log", height=300, width=400, opacity=0.7, style={'stroke-width': 4}
    )
    toytree.utils.set_axes_ticks_external(axes)
    axes.vlines([neff], style={"stroke": toytree.color.COLORS1[1], "stroke-width": 3})
    toytree.utils.show(canvas)


def test_msc(neff: float = 1e5, nsamples: int = 4, ntrees: int = 500):
    """Return a plot of the likelihood of constant Ne in multipop tree.
    """
    import toyplot
    import toytree

    # get (sptree, gtrees, imap)
    sptree = toytree.rtree.imbtree(ntips=5, treeheight=1e6)
    model = ipcoal.Model(sptree, Ne=neff, nsamples=nsamples)
    model.sim_trees(ntrees)
    imap = model.get_imap_dict()

    # get embedding table
    etable = get_genealogy_embedding_table(model.tree, model.df.genealogy, imap, df=False)

    # get loglik across a range of test values
    test_values = np.logspace(np.log10(neff) - 1, np.log10(neff) + 1, 20)
    logliks = []
    for val in test_values:
        etable[:, 3] = val
        loglik = get_msc_loglik_from_embedding_table(etable)
        logliks.append(loglik)

    canvas, axes, mark = toyplot.plot(
        test_values, logliks,
        xscale="log", height=300, width=400, opacity=0.7, style={'stroke-width': 4}
    )
    toytree.utils.set_axes_ticks_external(axes)
    axes.vlines([neff], style={"stroke": toytree.color.COLORS1[1], "stroke-width": 3})
    toytree.utils.show(canvas)


if __name__ == "__main__":

    import ipcoal
    ipcoal.set_log_level("INFO")
    # test_kingman(neff=1e6, nsamples=10, ntrees=500)
    test_msc(neff=2e6, nsamples=10, ntrees=500)

    # SPTREE = toytree.rtree.baltree(2, treeheight=1e6)
    # MODEL = ipcoal.Model(SPTREE, Ne=200_000, nsamples=4, seed_trees=123)
    # MODEL = ipcoal.Model(None, Ne=200_000, nsamples=4, seed_trees=123)
    # MODEL.sim_trees(10)
    # GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    # IMAP = MODEL.get_imap_dict()
    # data = get_genealogy_embedding_table(MODEL.tree, GENEALOGIES, IMAP, )
    # print(get_msc_loglik_from_embedding_table(data.values))

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
