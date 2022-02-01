# Copyright (C) 2021, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import pandas as pd
import numpy as np

from CytOpT import CytOpT


class bothMethod:

    def __init__(self, inputs):
        """
        2D projection using two markers.
        The data are divided into two classes: the CD4 cells where the CD4 marker is present and the CD8 cells where the CD8 marker is present.
        """
        self.inputs = inputs

    def comparisonOptTenClasses(self):
        XSource = np.asarray(self.inputs['Stanford1A_values'])
        XTarget = np.asarray(self.inputs['Stanford3A_values'])
        labSource = np.asarray(self.inputs['Stanford1A_clust']['x'])
        labTarget = np.asarray(self.inputs['Stanford3A_clust']['x'])

        thetaTrue = np.zeros(10)
        for k in range(10):
            thetaTrue[k] = np.sum(labTarget == k + 1) / len(labTarget)

        nIter = 1000
        stepGrad = 5
        power = 0.99

        Results_Minmax = CytOpT(XSource, XTarget, labSource, eps=0.0001, lbd=0.0001, nIter=nIter,
                                method="minmax", step=stepGrad, power=power, monitoring=True)

        proportions = Results_Minmax['proportions']
        monitoringMinmax = Results_Minmax['monitoring']

        pasGrad = 50
        eps = 0.0001

        resTwo = CytOpT(XSource, XTarget, labSource,
                        method="desasc", stepGrad=pasGrad, eps=eps, thetaTrue=thetaTrue)

        return {'hHat': proportions, 'monitoringMinmax': monitoringMinmax, 'resTwo': resTwo}


if __name__ == '__main__':
    data = {
        # Source Data
        'Stanford1A_values': pd.read_csv('data/W2_1_values.csv',
                                         usecols=np.arange(1, 8)),
        'Stanford1A_clust': pd.read_csv('data/W2_1_clust.csv',
                                        usecols=[1]),

        # Target Data
        'Stanford3A_values': pd.read_csv('data/W2_7_values.csv',
                                         usecols=np.arange(1, 8)),
        'Stanford3A_clust': pd.read_csv('data/W2_7_clust.csv',
                                        usecols=[1])}
    test3 = bothMethod(data)
    print(test3.__dict__)
    print(test3.comparisonOptTenClasses())
