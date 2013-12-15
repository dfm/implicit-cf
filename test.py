#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
from multiprocessing import Pool

from icf import ICF

training_set = [map(int, l.split())
                for l in gzip.open("data/train.txt.gz")]
test_set = [map(int, l.split()) for l in gzip.open("data/test.txt.gz")]

pool = Pool()
model = ICF(200, 1000, 90126)

for recall in model.train(training_set, test_set=test_set, pool=pool):
    print(recall)
