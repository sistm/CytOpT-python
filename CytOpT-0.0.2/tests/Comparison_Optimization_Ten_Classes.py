# Copyright (C) 2021, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)
import time

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from CytOpT import cytopt_minmax, cytopt_desasc


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

        n_iter = 4000
        step_grad = 5
        power = 0.99

        t0 = time.time()
        Results_Minmax = cytopt_minmax(X_source, X_target, Lab_source, eps=0.0001, lbd=0.0001, n_iter=n_iter,
                                       theta_true=theta_true, step=step_grad, power=power, monitoring=True)
        elapsed_time = time.time() - t0
        print('Elapsed time : ', elapsed_time / 60, 'Mins')

        h_hat = Results_Minmax[0]
        Minmax_monitoring = Results_Minmax[1]

        n_it_grad = 1000
        n_it_sto = 10
        pas_grad = 50
        eps = 0.0001

        t0 = time.time()
        res_two = cytopt_desasc(X_s=X_source, X_t=X_target, Lab_source=Lab_source, eps=eps, n_out=n_it_grad,
                                n_stoc=n_it_sto, step_grad=pas_grad, theta_true=theta_true)

        elapsed_time = time.time() - t0
        print('Elapsed time : ', elapsed_time / 60, 'Mins')
        h_hat_two = res_two[0]
        Desasc_monitoring = res_two[1]
        return {'h_hat': h_hat, 'Minmax_monitoring': Minmax_monitoring, 'h_hat_two': h_hat_two,
                'Desasc_monitoring': Desasc_monitoring}


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
