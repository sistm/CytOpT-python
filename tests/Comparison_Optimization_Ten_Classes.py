# Copyright (C) 2021, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)
import time

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from CytOpT import CytOpT


class Bland_Altman_Full_HIPC:

    def __init__(self, data):
        """
        2D projection using two markers.
        The data are divided into two classes: the CD4 cells where the CD4 marker is present and the CD8 cells where the CD8 marker is present.
        """
        self.data = data

    def comparison_Opt_Ten_Classes(self):
        X_source = np.asarray(self.data['Stanford1A_values'])
        X_target = np.asarray(self.data['Stanford3A_values'])
        Lab_source = np.asarray(self.data['Stanford1A_clust']['x'])
        Lab_target = np.asarray(self.data['Stanford3A_clust']['x'])

        # Thresholding of the negative values
        X_source = X_source * (X_source > 0)
        X_target = X_target * (X_target > 0)

        scaler = MinMaxScaler()
        X_source = scaler.fit_transform(X_source)
        X_target = scaler.fit_transform(X_target)

        theta_true = np.zeros(10)
        for k in range(10):
            theta_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

        print(theta_true)

        n_iter = 1000
        step_grad = 5
        power = 0.99

        Results_Minmax = CytOpT(X_source, X_target, Lab_source, eps=0.0001, lbd=0.0001, n_iter=n_iter,
                                method="minmax", theta_true=theta_true, step=step_grad, power=power, monitoring=True)

        proportions = Results_Minmax['proportions']
        Minmax_monitoring = Results_Minmax['monitoring']

        pas_grad = 50
        eps = 0.0001

        res_two = CytOpT(X_source, X_target, Lab_source,
                         method="desasc", step_grad=pas_grad, eps=eps, theta_true=theta_true)
        proportions_desasc = res_two['proportions']
        Minmax_monitoring_desasc = res_two['monitoring']
        return {'h_hat': proportions, 'Minmax_monitoring': Minmax_monitoring, 'h_hat_two': proportions_desasc,
                'Desasc_monitoring': Minmax_monitoring_desasc}


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
    test3 = Bland_Altman_Full_HIPC(data)
    print(test3.__dict__)
    print(test3.comparison_Opt_Ten_Classes())
