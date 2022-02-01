# Copyright (C) 2021, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
import pandas as pd
import time

from CytOpT import CytOpT


class cytOptHIPCData:

    def __init__(self, inputs):
        """
        2D projection using two markers.
        The data are divided into two classes: the CD4 cells where the CD4 marker is present and the CD8 cells where the CD8 marker is present.
        """
        self.inputs = inputs

    def cytOpt_HIPC(self):
        # Source Data

        X_source = np.asarray(self.inputs['Stanford1A_values'])

        Lab_source = np.asarray(self.inputs['Stanford1A_clust']['x'])

        X_target = np.asarray(self.inputs['Stanford3C_values'])

        Lab_target = np.asarray(self.inputs['Stanford3C_clust']['x'])

        theta_true = np.zeros(10)
        for k in range(10):
            theta_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

        eps = 0.0001
        n_out = 1000
        n_stoc = 10
        step_grad = 50

        Results_Desasc = CytOpT(X_source, X_target, Lab_source,
                                method="desasc", nItGrad=n_out, nItSto=n_stoc, stepGrad=step_grad, eps=eps)

        h_hat1 = Results_Desasc
        print(Results_Desasc)

        n_iter = 4000
        step_grad = 5
        power = 0.99

        t0 = time.time()
        Results_Minmax = CytOpT(X_source, X_target, Lab_source, eps=0.0001, lbd=0.0001, nIter=n_iter,
                                method="minmax", thetaTrue=theta_true, step=step_grad, power=power, monitoring=False)
        elapsed_time = time.time() - t0
        print('Elapsed time : ', elapsed_time / 60, 'Mins')
        h_hat2 = Results_Minmax[0]

        Proportion = np.hstack((h_hat1, h_hat2, theta_true))
        Classes = np.tile(np.arange(1, 11), 3)
        Methode = np.repeat(['CytOpt_DesAsc', 'CytOpt_Minmax', 'Manual'], 10)
        df_res1 = pd.DataFrame({'Proportions': Proportion, 'Classes': Classes, 'Methode': Methode})
        return df_res1


if __name__ == '__main__':
    data = {
        # Source Data
        'Stanford1A_values': pd.read_csv('data/W2_1_values.csv',
                                         usecols=np.arange(1, 8)),
        'Stanford1A_clust': pd.read_csv('data/W2_1_clust.csv',
                                        usecols=[1]),
        # Target Data
        'Stanford3C_values': pd.read_csv('data/W2_9_values.csv',
                                         usecols=np.arange(1, 8)),
        'Stanford3C_clust': pd.read_csv('data/W2_9_clust.csv',
                                        usecols=[1])}
    test4 = cytOptHIPCData(inputs=data)
    print(test4)
