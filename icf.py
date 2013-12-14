#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

from collections import defaultdict
import numpy as np


class ICF(object):

    def __init__(self, K, nusers, nitems, alpha=40.0, l2v=0.01, l2u=0.01):
        self.K = K
        self.nusers = nusers
        self.nitems = nitems
        self.alpha = alpha
        self.l2v = l2v
        self.l2u = l2u

        self.U = np.random.rand(nusers, K)
        self.V = np.random.rand(nitems, K)

    def train(self, training_set):
        user_items = [[] for i in range(self.nusers)]
        item_users = [[] for i in range(self.nitems)]
        [(user_items[u].append(a), item_users[a].append(u))
         for u, a in training_set]

        for i in range(10):
            print("Updating users")
            vtv = np.dot(self.V.T, self.V)
            for i, vec in enumerate(user_items):
                vm = self.V[vec]
                vtcv = vtv + self.alpha * np.dot(vm.T, vm)
                vtcv[np.diag_indices(self.K)] += self.l2u
                b = (1+self.alpha)*np.sum(vm, axis=0)
                self.U[i] = np.linalg.solve(vtcv, b)

            print("Updating items")
            utu = np.dot(self.U.T, self.U)
            for i, vec in enumerate(item_users):
                um = self.U[vec]
                utcu = utu + self.alpha * np.dot(um.T, um)
                utcu[np.diag_indices(self.K)] += self.l2v
                b = (1+self.alpha)*np.sum(um, axis=0)
                self.V[i] = np.linalg.solve(utcu, b)

            yield i

    def test(self, test_set, M=200):
        user_items = defaultdict(list)
        [user_items[u].append(a) for u, a in test_set]

        recall = 0.0
        for u, items in user_items.items():
            r = np.dot(self.V, self.U[u])
            inds = np.argsort(r)[::-1]
            recall += np.sum([i in items for i in inds[:M]]) / len(items)
        return recall / len(user_items)


if __name__ == "__main__":
    import gzip
    training_set = [map(int, l.split())
                    for l in gzip.open("data/train.txt.gz")]
    test_set = [map(int, l.split()) for l in gzip.open("data/test.txt.gz")]

    model = ICF(50, 1000, 90126)
    for i in model.train(training_set):
        print("Testing")
        print(model.test(test_set))
