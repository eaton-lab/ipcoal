#!/usr/bin/env python

"""SMC source code.

Examples
--------

Generate species tree and genealogies
>>> sptree = toytree.rtree.imbtree(ntips=4, treeheight=1e6)
>>> model = ipcoal.Model(sptree, Ne=1e5, nsamples=2)
>>> model.sim_trees(10)
>>> gtrees = model.df.genealogy
>>> imap = model.get_imap_dict()

Generate an embedding table as a dataframe
>>> args = (model.tree, model.df.genealogy, imap)
>>> table = ipcoal.smc.get_genealogy_embedding_table(*args)

Get `Prob(event|S,G,b,t)`
>>> get_prob_tree_unchanged_given_b_and_tr()

Get `Prob(event|S,G,b,t)` faster from embedding table
>>> get_prob_tree_unchanged_given_b_and_tr_from_table(table, branch=2, time=100)


>>> get_tree_change_prob(sptree, gtree, imap, recomb)
>>> get_tree_change_prob_from_table(table, recomb)
>>> get_tree_change_prob_given_b(sptree, gtree, imap, recomb, branch)
>>> get_tree_change_prob_given_b_from_table(table, recomb, branch)
>>> get_tree_change_prob_given_b_and_tr(sptree, gtree, imap, recomb, branch, time)
>>> get_tree_change_prob_given_b_and_tr_from_table(table, branch, time)



"""
