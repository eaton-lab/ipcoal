#!/usr/bin/env python

"""An embedding table for MSC calculations.

# ['start', 'stop', 'st_node', 'neff', 'nedges', 'dist', 'gidx']

>>> get_genealogy_embedding_table(sptree, gtree, imap)
"""

from typing import Mapping, Sequence, Union
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
import numpy as np
import pandas as pd
import toytree

logger = logger.bind(name="ipcoal")

__all__ = ["get_genealogy_embedding_table"]


def _get_fast_genealogy_embedding_table(
    species_tree: toytree.ToyTree,
    genealogy: toytree.ToyTree,
    imap: Mapping[str, Sequence[str]],
    df: bool = True,
    encode: bool = False,
    gidx: int = 0,
) -> Union[pd.DataFrame, np.ndarray]:
    """Return a genealogy embedding table.

    The embedding table represents intervals in a species tree where
    that have piece-wise constant coalescent rates and which are
    dividied by coalescent events in the gene tree, or speciation
    events in the species tree.

    Parameters
    ----------
    species_tree: ToyTree
        ...
    genealogy: ToyTree
        ...
    imap: Dict[str, List[str]]
        ...
    encode: bool
        If True then the genealogy Node IDs in each interval are one-
        hot-encoded, otherwise they are excluded.
    df: bool
        If True the table is returned as a DataFrame with column labels
        else it is returned as a numpy array.
    gidx: int or None
        If a genealogy index is provided then a column is added that
        is filled with this integer.

    Examples
    --------
    >>> ...
    """
    # store temporary results in a dict
    split_data = []
    edge_encode = []
    nnodes = genealogy.nnodes

    # dict to update tips to their ancestor if already coalesced.
    name_to_node = {i._name: i for i in genealogy[:genealogy.ntips]}

    # get table of gene tree node heights
    gt_node_heights = np.array([i._height for i in genealogy])

    # iterate over stree from tips to root storing stree node data
    for st_node in species_tree:
        # get all gtree names descended from this sptree interval
        gt_tips = set.union(*(set(imap[i]) for i in st_node.get_leaf_names()))

        # get gtree nodes descended from this interval.
        gt_nodes = set(name_to_node[i] for i in gt_tips)

        # get internal nodes in this TIME interval (coalescences)
        lower = st_node.height + 0.0001  # zero-align ipcoal bug
        upper = np.inf if st_node.is_root() else st_node._up._height
        nodes_in_time_slice = (gt_node_heights > lower) & (gt_node_heights < upper)

        # filter internal nodes must be ancestors of gt_nodes (in this interval)
        # get internal nodes in this interval by getting all nodes in this
        # time interval and requiring that all their desc are in gt_nodes.
        inodes = []
        for iidx in nodes_in_time_slice.nonzero()[0]:
            gt_node = genealogy[iidx]
            if not (set(gt_node.get_leaf_names()) - gt_tips):
                inodes.append(gt_node)

        # vars to be updated at coal events
        start = st_node._height
        edges = {i._idx for i in gt_nodes}

        # iterate over internal nodes
        for gt_node in sorted(inodes, key=lambda x: x._height):

            # add interval from start to first coal, or coal to next coal
            split_data.append([
                start,
                gt_node._height,
                st_node._idx,
                st_node.Ne,
                len(edges),
                0,
                gidx,
            ])
            edge_encode.append(sorted(edges))

            # update counters and indexers
            start = gt_node._height
            edges.add(gt_node._idx)
            for child in gt_node._children:
                edges.remove(child._idx)
            for tip in gt_node.get_leaves():
                name_to_node[tip.name] = gt_node

        # add non-coal interval
        split_data.append([
            start,
            st_node._up._height if st_node._up else np.inf,
            st_node._idx,
            st_node.Ne,
            len(edges),
            0,
            gidx,
        ])
        edge_encode.append(sorted(edges))

    # to array
    arr = np.vstack(split_data)

    # format as a sleek float array
    if not df:
        arr[:, 5] = arr[:, 1] - arr[:, 0]

        # return with coal IDs in list
        if not encode:
            return arr
            # return np.c_[arr, dist, np.array(edge_encode, dtype=object)]

        # return with coal IDs one-hot-encoded
        encoding = []
        for nodes in edge_encode:
            e = np.zeros(nnodes)
            e[nodes] = 1
            encoding.append(e)
        return np.c_[arr, encoding]

    # format as a nice (but slow) human readable dataframe
    table = pd.DataFrame(
        data=split_data,
        columns=['start', 'stop', 'st_node', 'neff', 'nedges', 'dist', 'gidx']
    )
    table['dist'] = table.stop - table.start
    if not encode:
        return table
        # table['edges'] = edge_encode

    # one-hot-encode the edges
    encoding = []
    for nodes in edge_encode:
        e = np.zeros(nnodes)
        e[nodes] = 1
        encoding.append(e)
    encoding = pd.DataFrame(
        data=encoding,
        columns=range(nnodes),
        dtype=np.uint8,
    )
    # table = table.drop(columns=["coal", "edges"])
    return pd.concat([table, encoding], axis=1)


def get_genealogy_embedding_table(
    species_tree: toytree.ToyTree,
    genealogies: Sequence[toytree.ToyTree],
    imap: Mapping[str, Sequence[str]],
    df: bool = True,
    encode: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """Return a genealogy embedding table."""
    # get genealogies as Sequence[ToyTree]
    if isinstance(genealogies, (toytree.ToyTree, str)):
        genealogies = [genealogies]
    if isinstance(genealogies[0], str):
        genealogies = toytree.mtree(genealogies).treelist

    # iterate over genealogies
    etables = []
    for gidx, gtree in enumerate(genealogies):
        args = (species_tree, gtree, imap, df, encode, gidx)
        etables.append(_get_fast_genealogy_embedding_table(*args))

    # concatenate
    if isinstance(etables[0], np.ndarray):
        return np.vstack(etables)
    return pd.concat(etables, axis=0)


if __name__ == "__main__":

    import ipcoal
    SPTREE = toytree.rtree.baltree(2, treeheight=1e6)
    MODEL = ipcoal.Model(SPTREE, Ne=200_000, nsamples=4, seed_trees=123)
    MODEL.sim_trees(10)
    GENEALOGIES = toytree.mtree(MODEL.df.genealogy)
    IMAP = MODEL.get_imap_dict()
    print(get_genealogy_embedding_table(MODEL.tree, GENEALOGIES, IMAP, encode=False))
