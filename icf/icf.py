#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["ICF"]

from collections import defaultdict

import numpy as np


class ICF(object):

    def __init__(self, K, nusers, nitems, alpha=10.0, l2v=0.1, l2u=0.1):
        self.K = K
        self.nusers = nusers
        self.nitems = nitems
        self.alpha = alpha
        self.l2v = l2v
        self.l2u = l2u

        self.U = np.random.rand(nusers, K)
        self.V = np.random.rand(nitems, K)

    def train(self, training_set, test_set=None, pool=None, N=200):
        M = map if pool is None else pool.map

        user_items = [[] for i in range(self.nusers)]
        item_users = [[] for i in range(self.nitems)]
        [(user_items[u].append(a), item_users[a].append(u))
         for u, a in training_set]

        if test_set is not None:
            test_user_items = defaultdict(list)
            [test_user_items[u].append(a) for u, a in test_set]
            test_args = [(u, t, user_items[u])
                         for u, t in test_user_items.items()]

        for i in range(10):
            print("Updating users")
            vtv = np.dot(self.V.T, self.V)
            self.U = np.vstack(M(_function_wrapper(self,
                                                   "compute_user_update",
                                                   vtv), user_items))

            print("Updating items")
            utu = np.dot(self.U.T, self.U)
            self.V = np.vstack(M(_function_wrapper(self,
                                                   "compute_item_update",
                                                   utu), item_users))

            # Compute the held out recall.
            if test_set is not None:
                print("Computing held out recall")
                yield np.mean(M(_function_wrapper(self, "compute_recall", N=N),
                                test_args))
            else:
                yield 0.0

    def compute_user_update(self, vec, vtv):
        vm = self.V[vec]
        vtcv = vtv + self.alpha * np.dot(vm.T, vm)
        vtcv[np.diag_indices(self.K)] += self.l2u
        b = (1+self.alpha)*np.sum(vm, axis=0)
        return np.linalg.solve(vtcv, b)

    def compute_item_update(self, vec, utu):
        um = self.U[vec]
        utcu = utu + self.alpha * np.dot(um.T, um)
        utcu[np.diag_indices(self.K)] += self.l2v
        b = (1+self.alpha)*np.sum(um, axis=0)
        return np.linalg.solve(utcu, b)

    def compute_recall(self, args, N=200):
        u, items, previous = args
        items += previous

        # Compute the top recommendations.
        r = np.dot(self.V, self.U[u])
        inds = np.argsort(r)[::-1]

        # Compute the recall.
        recall = np.sum([i in items for i in inds[:N]]) / len(items)
        return recall

    # def compute_recall(self, args, N=200):
    #     u, items, previous = args

    #     # Remove items in training data.
    #     m = np.ones(self.nitems, dtype=bool)
    #     m[previous] = False

    #     # Compute the top recommendations.
    #     r = np.dot(self.V[m], self.U[u])
    #     inds = np.arange(self.nitems)[m][np.argsort(r)[::-1]]

    #     # Compute the recall.
    #     recall = np.sum([i in items for i in inds[:N]]) / len(items)
    #     return recall


class _function_wrapper(object):

    def __init__(self, target, attr, *args, **kwargs):
        self.target = target
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def __call__(self, v):
        return getattr(self.target, self.attr)(v, *self.args, **self.kwargs)
