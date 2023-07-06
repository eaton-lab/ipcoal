#!/usr/bin/env python

"""Utility functions for SMC subpackage."""

from typing import Iterator, Tuple, Sequence, Optional
from loguru import logger
import toytree
from toytree import ToyTree
import numpy as np
import ipcoal

logger = logger.bind(name="ipcoal")


def iter_spans_and_trees_from_model(
    model: ipcoal.Model,
    locus: Optional[int] = None,
) -> Iterator[Tuple[int, ToyTree]]:
    """Yield (int, ToyTree) tuples parsed from a model object.

    This is a convenient way to extract ARG data from an ipcoal
    simulation result, and is used within other functions such as
    `iter_spans_and_topologies()` to extract spans and topology
    changes from an ARG.
    """
    data = model.df[model.df.locus == locus] if locus is not None else model.df
    for span, newick in data[["nbps", "genealogy"]].values:
        yield (span, toytree.tree(newick))


def iter_spans_and_topologies_from_model(
    model: ipcoal.Model,
    locus: Optional[int] = None,
    # average_node_heights: bool = False,
) -> Iterator[Tuple[int, ToyTree]]:
    """Return an array of genealogical topology lengths.

    Parameters
    ----------
    model: ipcoal.Model
        An ipcoal.Model object that has called a simulate function
        such as `sim_trees`, `sim_loci`, etc. such that it has data
        in its `.df` attribute.
    average_node_heights: bool
        If True the node heights are re-calibrated to the site weighted
        average across all tree-change intervals that occurred during
        the topology-change waiting distance.
    """
    current = None
    intervals = []
    # forest = []

    for span, gtree in iter_spans_and_trees_from_model(model, locus=locus):
        intervals.append(span)
        # forest.append(gtree.get_node_data())
        if current:
            new = gtree.get_topology_id(include_root=True)
            # if current.distance.get_treedist_rf(gtree, False):
            if current != new:
                yield sum(intervals), gtree
                # current = gtree
                current = new
                intervals = []
        else:
            # current = gtree
            current = gtree.get_topology_id(include_root=True)


# NEW BEST FUNCTION WITH NODE HEIGHT AVERAGING
def iter_topos_and_spans_from_model(
    model: ipcoal.Model,
    weighted_node_heights: bool = False,
) -> Iterator[ToyTree]:
    """...

    weighted_node_heights: bool
        Not currently working, negative branch lengths can occur.
    """
    cnewick = None
    ctree = None
    ctopo = None
    forest = {}

    # iterate over trees
    idata = zip(model.df.genealogy, model.df.nbps)
    for tidx, (newick, length) in enumerate(idata):

        # load first tree
        if cnewick is None:
            cnewick = newick
            ctree = toytree.tree(newick)
            ctopo = ctree.get_topology_id(include_root=True)
            forest[ctree] = length
            # logger.debug(f"first tree {length}")
            continue

        # newick str did not change (invisible recomb) just add length
        # to the last stored tree.
        if newick == cnewick:
            forest[ctree] += length
            # logger.debug(f"no-change {length}")
            continue

        # newick str changed, check if topology changed. If not, store
        # the last tree and its length to forest, and update current.
        else:
            tree = toytree.tree(newick)
            topo = tree.get_topology_id(include_root=True)
            if topo != ctopo:
                # logger.debug(f"*topo-change -> {sum(forest[i] for i in forest)}")
                if not weighted_node_heights:
                    yield ctree, sum(forest[i] for i in forest)
                else:
                    heights = {}
                    w = [forest[i] for i in forest]

                    # cannot use idx labels here, must use anc names...
                    for nidx in range(ctree.ntips, ctree.nnodes):
                        hnodes = [t.get_mrca_node(*ctree[nidx].get_leaf_names()) for t in forest]
                        h = [i.height for i in hnodes]
                        height = np.average(h, weights=w)
                        # do not allow averaging to cause negative height
                        minh = max([heights.get(i._idx, 0) for i in ctree[nidx].children])
                        if height <= minh:
                            height = minh + 1
                        heights[nidx] = height
                    ctree = ctree.mod.edges_set_node_heights(heights)
                    yield ctree, sum(forest[i] for i in forest)

                cnewick = newick
                ctree = tree
                ctopo = topo
                forest = {ctree: length}
                # logger.debug(f"first tree {length}")

            else:
                # logger.debug(f"tree-change {length}")
                forest[tree] = length
                ctree = tree


def iter_topos_from_trees(trees: Sequence[ToyTree]) -> Iterator[ToyTree]:
    """Returns the genealogy at each topology change."""

    # initial tree
    current = trees[0]
    cidx = current.get_topology_id(include_root=True)
    tree_bunch = [current]

    # iterate over genealogies
    for gtree in trees[1:]:
        nidx = gtree.get_topology_id(include_root=True)
        if cidx != nidx:
            yield current
            current = gtree
            cidx = nidx
            tree_bunch = [gtree]
        else:
            tree_bunch.append(gtree)


def get_topology_interval_lengths(
    model: ipcoal.Model,
    locus: Optional[int] = None,
) -> np.ndarray:
    """Return an array of genealogical topology lengths.

    Parameters
    ----------
    model: ipcoal.Model
        A Model object that has simulated data in its `.df` attribute.
    """
    return np.array([i[1] for i in iter_topos_and_spans_from_model(model)])


if __name__ == "__main__":

    ipcoal.set_log_level("DEBUG")
    MODEL = ipcoal.Model(Ne=100_000, nsamples=5, seed_trees=333)
    MODEL.sim_trees(1, 80000)
    print(MODEL.df.head())

    i0 = iter_topos_and_spans_from_model(MODEL, True)
    i1 = iter_topos_from_trees(toytree.mtree(MODEL.df.genealogy))

    for (i, dist), j in zip(i0, i1):
        print(i.distance.get_treedist_rf(j), dist)

    # c0, _, _ = gtrees[0].draw()
    # c1, _, _ = gtrees[1].draw()
    # c2, _, _ = gtrees[2].draw()

    # toytree.utils.show([c0, c1, c2])

    # for x in iter_topos_and_spans_from_model(MODEL, False):
    #     print(x)

    # for SPAN, GTREE in iter_spans_and_topologies_from_model(MODEL):
    #     print(SPAN)
    #     GTREE.treenode.draw_ascii()

    # ivals = get_topology_interval_lengths(MODEL)
    # print(ivals, ivals.sum())

    # GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    # for g in iter_unique_topologies_from_trees(GENEALOGIES):
    #     print(g.get_topology_id(include_root=True), g.get_node_data("height"))
