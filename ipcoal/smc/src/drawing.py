#!/usr/bin/env python

"""Visualizations for the MS-SMC process

"""

from typing import Mapping, Sequence
from toytree import ToyTree
import toyplot
import pandas as pd
import numpy as np
from ipcoal.msc import get_genealogy_embedding_table
# from ipcoal.smc import get_prob_tree_unchanged_given_b_and_tr_from_table
# from ipcoal.smc import get_prob_topo_unchanged_given_b_and_tr_from_table
# get_waiting_distance_to_recomb_event_rv
# get_waiting_distance_to_tree_change_event_rv
# get_waiting_distance_to_topo_change_event_rv


def get_genealogy_embedding_edge_path(table: pd.DataFrame, branch: int) -> pd.DataFrame:
    """Return the gene tree embedding table intervals a gtree edge
    passes through.

    Parameters
    ----------
    table:
        A table returned by the `get_genealogy_embedding_table` func.
    branch:
        An integer index (idx) label to select a genealogy branch.
    """
    return table[table.edges.apply(lambda x: branch in x)]


def plot_edge_probabilities(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    branch: int,
    stack: int = 1,
    **kwargs,
) -> toyplot.canvas.Canvas:
    """Return a toyplot canvas with probabilities along an edge.

    """
    # setup the canvas and axes
    if not stack:
        canvas = toyplot.Canvas(
            height=kwargs.get("height", 250),
            width=kwargs.get("width", 800),
        )
        axstyle = dict(ymin=0, ymax=1, margin=65)
        ax0 = canvas.cartesian(grid=(1, 3, 0), label="Prob(no-change)", **axstyle)
        ax1 = canvas.cartesian(grid=(1, 3, 1), label="Prob(tree-change)", **axstyle)
        ax2 = canvas.cartesian(grid=(1, 3, 2), label="Prob(topo-change)", **axstyle)

    else:
        canvas = toyplot.Canvas(
            height=kwargs.get("height", 750),
            width=kwargs.get("width", 300),
        )
        axstyle = dict(ymin=0, ymax=1, margin=65)
        ax0 = canvas.cartesian(grid=(3, 1, 0), label="Prob(no-change)", **axstyle)
        ax1 = canvas.cartesian(grid=(3, 1, 1), label="Prob(tree-change)", **axstyle)
        ax2 = canvas.cartesian(grid=(3, 1, 2), label="Prob(topo-change)", **axstyle)

    # Select a branch to plot and get its relations
    branch = genealogy[branch]
    bidx = branch.idx
    sidx = branch.get_sisters()[0].idx
    pidx = branch.up.idx

    # Get genealogy embedding table
    etable = get_genealogy_embedding_table(species_tree, genealogy, imap)
    btable = get_genealogy_embedding_edge_path(etable, bidx)

    # Plot probabilities of change types over a single branch
    # Note these are 'unchange' probs and so we plot 1 - Prob here.
    times = np.linspace(branch.height, branch.up.height, 200, endpoint=False)
    pt_nochange_tree = [
        get_probability_tree_unchanged_given_b_and_tr_from_table(
            etable, bidx, itime
        ) for itime in times
    ]
    pt_nochange_topo = [
        get_probability_topology_unchanged_given_b_and_tr_from_table(
            etable, bidx, sidx, pidx, itime
        ) for itime in times
    ]

    # add line and fill for probabilities
    ax0.plot(times, pt_nochange_tree, stroke_width=5)
    ax0.fill(times, pt_nochange_tree, opacity=0.33)
    ax1.plot(times, 1 - np.array(pt_nochange_tree), stroke_width=5)
    ax1.fill(times, 1 - np.array(pt_nochange_tree), opacity=0.33)
    ax2.plot(times, 1 - np.array(pt_nochange_topo), stroke_width=5)
    ax2.fill(times, 1 - np.array(pt_nochange_topo), opacity=0.33)

    # add vertical lines at interval breaks
    style = {"stroke": "black", "stroke-width": 2, "stroke-dasharray": "4,2"}
    intervals = [btable.start.iloc[0]] + list(btable.stop - 0.001)
    for itime in intervals:
        iprob_tree = get_probability_tree_unchanged_given_b_and_tr_from_table(etable, bidx, itime)
        iprob_topo = get_probability_topology_unchanged_given_b_and_tr_from_table(etable, bidx, sidx, pidx, itime)
        ax0.plot([itime, itime], [0, iprob_tree], style=style)
        ax1.plot([itime, itime], [0, 1 - iprob_tree], style=style)
        ax2.plot([itime, itime], [0, 1 - iprob_topo], style=style)

    # style the axes
    for axis in (ax0, ax1, ax2):
        axis.x.ticks.show = axis.y.ticks.show = True
        axis.y.domain.show = False
        axis.x.ticks.near = axis.y.ticks.near = 7.5
        axis.x.ticks.far = axis.y.ticks.far = 0
        axis.x.ticks.labels.offset = axis.y.ticks.labels.offset = 15
        axis.x.label.text = f"Time of recombination on branch {bidx}"
        axis.y.label.text = "Probability"
        axis.x.label.offset = axis.y.label.offset = 35
        axis.x.spine.style['stroke-width'] = axis.y.spine.style['stroke-width'] = 1.5
        axis.x.ticks.style['stroke-width'] = axis.y.ticks.style['stroke-width'] = 1.5
        axis.label.offset = 20
        axis.x.ticks.locator = toyplot.locator.Explicit([btable.start.iloc[0]] + list(btable.stop))
    return canvas


def plot_waiting_distance_distributions(
    species_tree: ToyTree,
    genealogy: ToyTree,
    imap: Mapping[str, Sequence[str]],
    recombination_rate: float,
    **kwargs,
) -> toyplot.canvas.Canvas:
    """Return a toyplot canvas with waiting distance distributions.
    """
    canvas = toyplot.Canvas(
        height=kwargs.get("height", 300),
        width=kwargs.get("width", 450),
    )
    axes = canvas.cartesian(margin=65)
    axes.x.ticks.show = axes.y.ticks.show = True
    axes.y.domain.show = False
    axes.x.ticks.near = axes.y.ticks.near = 7.5
    axes.x.ticks.far = axes.y.ticks.far = 0
    axes.x.ticks.labels.offset = axes.y.ticks.labels.offset = 15
    axes.x.label.text = "Distance to next event"
    axes.y.label.text = "Probability"
    axes.x.label.offset = axes.y.label.offset = 35
    axes.x.spine.style['stroke-width'] = axes.y.spine.style['stroke-width'] = 1.5
    axes.x.ticks.style['stroke-width'] = axes.y.ticks.style['stroke-width'] = 1.5
    axes.label.offset = 20

    rv_recomb = get_waiting_distance_to_recombination_event_rv(genealogy, recombination_rate)
    rv_tree = get_waiting_distance_to_tree_change_rv(species_tree, genealogy, imap, recombination_rate)
    rv_topo = get_waiting_distance_to_topology_change_rv(species_tree, genealogy, imap, recombination_rate)

    for dist in [rv_recomb, rv_tree, rv_topo]:
        xs_ = np.linspace(dist.ppf(0.025), dist.ppf(0.975), 100)
        ys_ = dist.pdf(xs_)
        axes.plot(xs_, ys_, stroke_width=4)
        axes.fill(xs_, ys_, opacity=1 / 3)
    return canvas


if __name__ == "__main__":

    pass
