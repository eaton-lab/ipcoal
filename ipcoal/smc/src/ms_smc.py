#!/usr/bin/env python

"""General user-facing functions for accessing MS-SMC calculations.

This module provides user-friendly functions that accept either
ToyTrees as input, or a genealogy embedding table as a DataFrame.
Within the functions the faster jit-compiled numpy array-based methods
from the smc module are called, which take the more complicated
TreeEmbedding class inputs. The latter are much faster and should be
used by high-level users.
"""

from typing import Mapping, Sequence
from toytree import ToyTree
from ipcoal.msc import get_genealogy_embedding_table
from ipcoal.smc.src.embedding import TreeEmbedding, TopologyEmbedding
from ipcoal.smc.src.ms_smc_tree_prob import (
    get_fast_tree_changed_lambda,
)
from ipcoal.smc.src.ms_smc_topo_prob import (
    get_fast_topo_changed_lambda,
)
from ipcoal.smc.src.likelihood import (
    get_tree_distance_loglik,
    get_topo_distance_loglik,
)


__all__ = [

]


def get_prob_tree_unchanged_given_b_and_tr(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    branch: int,
    time: float,
) -> float:
    """Return probability of tree-unchanged given a gene tree embedded
    in a species and tree and the branch and timing of recombination.

    $P(tree-unchanged | S,G,B,tr)$

    Parameters
    ----------

    Examples
    --------
    >>>
    """
    etable = get_genealogy_embedding_table(species_tree, genealogy, imap, df=False)
    return get_fast_prob_tree_unchanged_given_b(etable, branch)


if __name__ == "__main__":

    pass
