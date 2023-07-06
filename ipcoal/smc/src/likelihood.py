#!/usr/bin/env python

"""Compute likelihood of interval lengths given gene tree embedded
in a species tree.

"""

from scipy import stats
import numpy as np
from loguru import logger
from numba import njit
from ipcoal.smc.src.embedding import TreeEmbedding
from ipcoal.smc.src.ms_smc_tree_prob import get_fast_tree_changed_lambdas
from ipcoal.smc.src.ms_smc_topo_prob import get_fast_topo_changed_lambdas

logger = logger.bind(name="ipcoal")


def _update_neffs(supertable: np.ndarray, popsizes: np.ndarray) -> None:
    """Updates 2X diploid Ne values in the concatenated embedding array.

    This is used during MCMC proposals to update Ne values. It takes
    Ne values as input, but stores to the array as 2Ne.

    TODO: faster method use stored masks
    """
    if len(set(popsizes)) == 1:
        supertable[:, :, 3] = popsizes[0] * 2
    else:
        for idx, popsize in enumerate(popsizes):
            mask = supertable[:, :, 2] == idx
            supertable[mask, 3] = popsize * 2


# def get_simple_waiting_distance_likelihood(
#     ts: TreeSequence,
#     recombination: float,
# ) -> float:
#     """Return the likelihood of waiting distances in an ARG.

#     This calculates the likelihood of interval lengths in a tree
#     sequence under the assumption that they are drawn from an
#     exponential probability density with rate parameter r x L, where
#     r is the recombination rate and L is the sum branch lengths of the
#     last genealogy.

#     This can only be run on TreeSequences simulated under the following
#     settings and will raise an exception if not.
#     >>> ancestry="hudson"
#     >>> discrete_genome=False
#     """
#     assert ts.discrete_genome == False
#     # assert ts.ancestry_model == "hudson"  # don't know how to check.


# def get_simple_arg_likelihood(
#     ts: TreeSequence,
#     recombination: float,
# ) -> float:
#     """Return the likelihood of an ARG.

#     This calculates (1) the likelihood of interval lengths in a tree
#     sequence under the assumption that they are drawn from an
#     exponential probability density with rate parameter r x L, where
#     r is the recombination rate and L is the sum branch lengths of the
#     last genealogy; and (2) the likelihood of the coalescent times
#     given the demographic model.

#     This can only be run on TreeSequences simulated under the following
#     settings and will raise an exception if not.
#     >>> ancestry="hudson"
#     >>> discrete_genome=False
#     """
#     assert ts.discrete_genome == False
#     # assert ts.ancestry_model == "hudson"  # don't know how to check.

def get_fast_recomb_event_lambdas(
    sarr: np.ndarray,
    recombination_rate: float,
    *args,
    **kwargs,
) -> np.ndarray:
    """Return loglikelihood of observed waiting distances to recomb events given tree lens and r"""
    return recombination_rate * sarr


def get_waiting_distance_loglik(
    embedding: TreeEmbedding,
    recomb: float,
    lengths: np.ndarray,
    event_type: int = 1,
) -> float:
    """Return -loglik of tree-sequence waiting distances between
    tree change events given species tree parameters.

    Here we will assume a fixed known recombination rate.

    Parameters
    ----------
    embedding_arr: TreeEmbedding
        A TreeEmbedding object with genealogy embedding arrays.
    params: np.ndarray
        An array of effective population sizes to apply to each linaege
        in the demographic model, ordered by their idx label in the
        species tree ToyTree object. Diploid Ne values.
    recomb: float
        per site per generation recombination rate.
    lengths: np.ndarray
        An array of observed waiting distances between tree changes
    event_type: int
        0 = any recombination event.
        1 = tree-change event.
        2 = topology-change event.

    Examples
    --------
    >>> embedding = TreeEmbedding(model.tree, model.df.genealogy, imap)
    >>> intervals = model.df.nbps.values
    >>> params = np.array([1e5, 1e5, 1e5])
    >>> get_tree_distance_loglik(embedding, params, 2e-9, intervals)
    """
    # get rates (lambdas) for waiting distances
    if event_type == 0:
        rate_function = get_fast_recomb_event_lambdas
    elif event_type == 1:
        rate_function = get_fast_tree_changed_lambdas
    else:
        rate_function = get_fast_topo_changed_lambdas

    rates = rate_function(
        emb=embedding.emb,
        enc=embedding.enc,
        barr=embedding.barr,
        sarr=embedding.sarr,
        rarr=embedding.rarr,
        recombination_rate=recomb,
    )

    # get logpdf of observed waiting distances given rates (lambdas)
    logliks = stats.expon.logpdf(scale=1 / rates, x=lengths)
    return -np.sum(logliks)


if __name__ == "__main__":

    import toytree
    import ipcoal
    from ipcoal.msc import _get_msc_loglik_from_embedding

    ############################################################
    RECOMB = 2e-9
    SEED = 123
    NEFF = 1e5
    ROOT_HEIGHT = 1e6
    NSPECIES = 2
    NSAMPLES = 8
    NSITES = 1e5
    NLOCI = 100

    sptree = toytree.rtree.baltree(NSPECIES, treeheight=ROOT_HEIGHT)
    sptree.set_node_data("Ne", {0: 1e5, 1: 2e5, 2: 2e5}, inplace=True)
    model = ipcoal.Model(sptree, nsamples=NSAMPLES, recomb=RECOMB, seed_trees=SEED)
    model.sim_trees(NLOCI, NSITES)
    imap = model.get_imap_dict()

    genealogies = toytree.mtree(model.df.genealogy)
    glens = model.df.nbps.values
    G = TreeEmbedding(model.tree, genealogies, imap)
    print(len(genealogies), "gtrees")

    values = np.linspace(10_000, 400_000, 31)
    test_values = np.logspace(np.log10(NEFF) - 1, np.log10(NEFF) + 1, 20)
    for val in values:
        _update_neffs(G.emb, np.array([val, 2e5, 2e5]))
        tloglik = _get_msc_loglik_from_embedding(G.emb)
        wloglik = get_waiting_distance_loglik(G, RECOMB, glens)
        loglik = tloglik + wloglik
        print(f"{val:.2e} {loglik:.2f} {tloglik:.2f} {wloglik:.2f}")
    raise SystemExit(0)


    ############################################################


    raise SystemExit(0)

    sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
    model = ipcoal.Model(sptree, Ne=1e5, nsamples=2, seed_trees=123)
    model.sim_trees(2, 1e5)
    gtrees = model.df.genealogy
    imap = model.get_imap_dict()

    G = TreeEmbedding(model.tree, model.df.genealogy, imap)
    # tree_loglik = _get_msc_loglik_from_embedding_array(G.emb)
    # wait_loglik = get_waiting_distance_loglik(G, 1e-9, model.df.nbps.values)

    values = np.linspace(10_000, 400_000, 31)
    for val in values:
        _update_neffs(G.emb, np.array([val] * sptree.nnodes))
        tree_loglik = 0#_get_msc_loglik_from_embedding_array(G.emb)
        wait_loglik = get_waiting_distance_loglik(G, 1e-8, model.df.nbps.values)
        loglik = tree_loglik + wait_loglik
        print(f"{val:.2e} {loglik:.2f} {tree_loglik:.2f} {wait_loglik:.2f}")
    # print(table.table.iloc[:, :8])
    # print(table.barr.shape, table.genealogies[0].nnodes)
    # print(get_fast_tree_changed_lambda(table.earr, table.barr, table.sarr, 2e-9))

