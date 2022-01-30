# Copyright (C) 2021, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from CytOpT import cytopt_desasc, cytopt_minmax


class cytOpt_HIPC_Data:

    def __init__(self,data):
        """
        2D projection using two markers.
        The data are divided into two classes: the CD4 cells where the CD4 marker is present and the CD8 cells where the CD8 marker is present.
        """
        self.data = data

    def cytOpt_HIPC(self):
        # Source Data

        X_source = np.asarray(self.data['Stanford1A_values'])
        X_source = X_source * (X_source > 0)
        scaler = MinMaxScaler()
        X_source = scaler.fit_transform(X_source)
        Lab_source = np.asarray(self.data['Stanford1A_clust']['x'])

        X_target = np.asarray(self.data['Stanford3C_values'])
        X_target = X_target * (X_target > 0)
        X_target = scaler.fit_transform(X_target)
        Lab_target = np.asarray(self.data['Stanford3C_clust']['x'])

        theta_true = np.zeros(10)
        for k in range(10):
            theta_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

        eps = 0.0001
        n_out = 4000
        n_stoc = 10
        step_grad = 50

        t0 = time.time()
        Results_Desasc = cytopt_desasc(X_source, X_target, Lab_source, eps=eps, n_out=n_out, n_stoc=n_stoc,
                                       step_grad=step_grad, theta_true=theta_true)
        elapsed_time = time.time() - t0

        h_hat1 = Results_Desasc[0]

        eps = 0.0001
        n_iter = 4000
        step_grad = 5
        power = 0.99

        t0 = time.time()
        Results_Minmax = cytopt_minmax(X_source, X_target, Lab_source, eps=0.0001, lbd=0.0001, n_iter=n_iter,
                                       theta_true=theta_true, step=step_grad, power=power, monitoring=False)
        elapsed_time = time.time() - t0
        print('Elapsed time : ', elapsed_time / 60, 'Mins')
        h_hat2 = Results_Minmax[0]

        Proportion = np.hstack((h_hat1, h_hat2, theta_true))
        Classes = np.tile(np.arange(1, 11), 3)
        Methode = np.repeat(['CytOpt_DesAsc', 'CytOpt_Minmax', 'Manual'], 10)
        df_res1 = pd.DataFrame({'Proportions': Proportion, 'Classes': Classes, 'Methode': Methode})
        plot_py_prop2(df_res1)

        X_target = np.asarray(self.data['Miami3A_values'])
        X_target = X_target * (X_target > 0)
        X_target = scaler.fit_transform(X_target)
        Lab_target = np.asarray(self.data['Miami3A_clust']['x'])

        h_true = np.zeros(10)
        for k in range(10):
            h_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

        t0 = time.time()
        h_hat1 = cytopt_desasc(X_source, X_target, Lab_source, eps=eps, n_out=n_out, n_stoc=n_stoc,
                               step_grad=step_grad, theta_true=theta_true)[0]
        elapsed_time = time.time() - t0
        print('Elapsed_time :', elapsed_time / 60, 'mins')

        t0 = time.time()
        Results_Minmax = cytopt_minmax(X_source, X_target, Lab_source, eps=0.0001, lbd=0.0001, n_iter=n_iter,
                                       theta_true=theta_true, step=step_grad, power=power, monitoring=False)
        elapsed_time = time.time() - t0

        print('Elapsed time : ', elapsed_time / 60, 'Mins')

        h_hat2 = Results_Minmax[0]

        Proportion = np.hstack((h_hat1, h_hat2, h_true))
        Classes = np.tile(np.arange(1, 11), 3)
        Methode = np.repeat(['CytOpt_DesAsc', 'CytOpt_Minmax', 'Manual'], 10)
        df_res1 = pd.DataFrame({'Proportions': Proportion, 'Classes': Classes, 'Methode': Methode})
        plot_py_prop2(df_res1)

        X_target = np.asarray(self.data['Ucla2B_values'])
        X_target = X_target * (X_target > 0)
        X_target = scaler.fit_transform(X_target)
        Lab_target = np.asarray(self.data['Ucla2B_clust']['x'])

        h_true = np.zeros(10)
        for k in range(10):
            h_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

        t0 = time.time()
        h_hat1 = cytopt_desasc(X_source, X_target, Lab_source, eps=eps, n_out=n_out, n_stoc=n_stoc,
                               step_grad=step_grad, theta_true=h_true)[0]
        elapsed_time = time.time() - t0
        print('Elapsed_time :', elapsed_time / 60, 'mins')

        t0 = time.time()
        results = cytopt_minmax(X_source, X_target, Lab_source, eps=0.0001, lbd=0.0001, n_iter=n_iter,
                                theta_true=h_true, step=step_grad, power=power, monitoring=False)
        elapsed_time = time.time() - t0
        print('Elapsed time : ', elapsed_time / 60, 'Mins')

        h_hat2 = results[0]

        Proportion = np.hstack((h_hat1, h_hat2, h_true))
        Classes = np.tile(np.arange(1, 11), 3)
        Methode = np.repeat(['CytOpt_DesAsc', 'CytOpt_Minmax', 'Manual'], 10)
        df_res1 = pd.DataFrame({'Proportions': Proportion, 'Classes': Classes, 'Methode': Methode})
        plot_py_prop2(df_res1)


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
                                        usecols=[1]),
        'Miami3A_values': pd.read_csv('data/pM_7_values.csv',
                                      usecols=np.arange(1, 8)),
        'Miami3A_clust': pd.read_csv('data/pM_7_clust.csv',
                                     usecols=[1]),
        'Ucla2B_values': pd.read_csv('data/IU_5_values.csv',
                                     usecols=np.arange(1, 8)),
        'Ucla2B_clust': pd.read_csv('data/IU_5_clust.csv',
                                    usecols=[1])}
    test4 = cytOpt_HIPC_Data(data=data)
    print(test4.__dict__)
    print(test4.cytOpt_HIPC())
