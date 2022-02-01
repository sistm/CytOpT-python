# Copyright (C) 2022, Kalidou BA <kalidou.ia.mlds@gmail.com>=
#
# License: MIT (see COPYING file)
# !/usr/bin/env python
# coding: utf-8

import sys
from os import path
import numpy as np
import pandas as pd

from CytOpT import CytOpT


def main():
    Stanford1A_values = pd.read_csv('./tests/data/W2_1_values.csv',
                                    usecols=np.arange(1, 8))
    Stanford1A_clust = pd.read_csv('./tests/data/W2_1_clust.csv',
                                   usecols=[1])

    # Target Data
    Stanford3A_values = pd.read_csv('./tests/data/W2_7_values.csv',
                                    usecols=np.arange(1, 8))
    Stanford3A_clust = pd.read_csv('./tests/data/W2_7_clust.csv',
                                   usecols=[1])

    X_source = np.asarray(Stanford1A_values)
    X_target = np.asarray(Stanford3A_values)
    Lab_source = np.asarray(Stanford1A_clust['x'])
    Lab_target = np.asarray(Stanford3A_clust['x'])
    theta_true = np.zeros(10)
    for k in range(10):
        theta_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

    CytOpT(X_source, X_target, Lab_source, Lab_target=None,
           method="both", theta_true=theta_true,n_iter=1000, n_it_grad=1000)


if __name__ == '__main__':
    main()


