# Copyright (C) 2021, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)
import sys
from os import path
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler

from CytOpT import robbinsWass, CytOpT
from CytOpT.labelPropSto import labelPropSto, cTransform, cost

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


class TwoClassesTwoDimension:

    def __init__(self, inputs):
        """
        2D projection using two markers.
        The data are divided into two classes: the CD4 cells where the CD4 marker is present and the CD8 cells where the CD8 marker is present.
        """
        self.inputs = inputs
        self.I = self.inputs['X_source'].shape[0]
        self.J = self.inputs['X_target'].shape[0]

    def initClassProportions(self):
        # Computation of the benchmark class proportions
        h_source = np.zeros(2)
        for k in range(2):
            h_source[k] = np.sum(self.inputs['Lab_source'] == k) / len(self.inputs['Lab_source'])

        h_true = np.zeros(2)
        for k in range(2):
            h_true[k] = np.sum(self.inputs['Lab_target'] == k) / len(self.inputs['Lab_target'])
        """
        Illustration of the framework
        A segmented data set and an unlabelled target data set.
        """

        return {'h_source': h_source, 'h_true': h_true}

    def optimalTransport(self):
        """'
        Approximation of the optimal dual vector u.
        In order to compute an approximation of the optimal transportation plan, we need to approximate  𝑃𝜀 .
        """

        alpha = 1 / self.I * np.ones(self.I)
        beta = 1 / self.J * np.ones(self.J)

        # Preprocessing of the data

        self.inputs['X_source'] = self.inputs['X_source'] * (self.inputs['X_source'] > 0)
        self.inputs['X_target'] = self.inputs['X_target'] * (self.inputs['X_target'] > 0)

        scaler = MinMaxScaler()
        self.inputs['X_source'] = scaler.fit_transform(self.inputs['X_source'])
        self.inputs['X_target'] = scaler.fit_transform(self.inputs['X_target'])

        eps = 0.0001
        n_iter = 15000

        t0 = time.time()
        u_last = robbinsWass(self.inputs['X_source'], self.inputs['X_target'],
                             alpha, beta, eps=eps, nIter=n_iter)
        elapsed_time = time.time() - t0
        print('Elapsed time :', elapsed_time / 60, 'mins')

        # Label propagation

        L_source = np.zeros((2, self.I))
        for k in range(2):
            L_source[k] = np.asarray(self.inputs['Lab_source'] == k, dtype=int)

        t0 = time.time()
        Result_LP = labelPropSto(L_source, u_last, self.inputs['X_source'], self.inputs['X_target'],
                                 alpha, beta, eps)
        elapsed_time = time.time() - t0
        Lab_target_hat_one = Result_LP[1]
        print('Elapsed_time ', elapsed_time / 60, 'mins')
        return {'Lab_target_hat_one': Lab_target_hat_one, 'u_last': u_last}

    def estimateCytOpT(self):
        """
        Class proportions estimation with  𝙲𝚢𝚝𝙾𝚙𝚝
        Descent-Ascent procedure
        Setting of the parameters
        """
        n_it_grad = 1000
        n_it_sto = 10
        pas_grad = 50
        eps = 0.0001
        h_true = self.initClassProportions()['h_true']

        h_hat = CytOpT(xSource=self.inputs['X_source'], xTarget=self.inputs['X_target'],
                       labSource=self.inputs['Lab_source'],
                       method="both", eps=eps, nIter=n_it_grad, nItSto=n_it_sto,
                       stepGrad=pas_grad,
                       thetaTrue=h_true, monitoring=False)

        return {'h_hat': h_hat['proportions']}

    def estimateMinmax(self):
        """
        Minmax swapping procedure
        Setting of the parameters
        """
        eps = 0.0001
        lbd = 0.0001
        n_iter = 4000
        step_grad = 5
        power = 0.99

        Results_Minmax = CytOpT(self.inputs['X_source'], self.inputs['X_target'], self.inputs['Lab_source'],
                                eps=eps, lbd=lbd, nIter=n_iter,
                                step=step_grad, power=power, monitoring=False)
        return {'h_hat': Results_Minmax['proportions']}

    def optimalReweighted(self):
        """
        Classification using optimal transport with reweighted proportions
        The target measure  𝛽  is reweighted in order to match the weight vector  ℎ̂   estimated with  𝙲𝚢𝚝𝙾𝚙𝚝 .

        """
        value_h = self.estimateCytOpT()
        beta = 1 / self.J * np.ones(self.J)
        optimal_return = self.optimalTransport()
        D = np.zeros((self.I, 2))
        D[:, 0] = 1 / np.sum(self.inputs['Lab_source'] == 0) * np.asarray(self.inputs['Lab_source'] == 0, dtype=float)
        D[:, 1] = 1 / np.sum(self.inputs['Lab_source'] == 1) * np.asarray(self.inputs['Lab_source'] == 1, dtype=float)
        alpha_mod = D.dot(value_h['h_hat'])

        # Approximation of the optimal dual vector u.
        eps = 0.0001
        n_iter = 150000

        u_last_two = robbinsWass(self.inputs['X_source'], self.inputs['X_target'],
                                 alpha_mod, beta, eps=eps, nIter=n_iter)

        Result_LP = labelPropSto(self.inputs['Lab_source'], u_last_two, self.inputs['X_source'],
                                 self.inputs['X_target'],
                                 alpha_mod, beta, eps)

        Lab_target_hat_two = Result_LP[1]

        return {'Lab_target_hat_two': Lab_target_hat_two, 'u_last_two': u_last_two}

    def optimalWithoutReweighted(self):
        """
        Transportation plan with or without reweighting
        Without reweighting
        """
        n_sub = 500
        eps = 0.0001
        opt_transport_return = self.optimalTransport()
        opt_rew = self.optimalReweighted()
        beta = 1 / self.J * np.ones(self.J)

        source_indices = np.random.choice(self.I, size=n_sub, replace=False)
        u_ce_storage = np.zeros(self.J)
        for j in range(self.J):
            u_ce_storage[j] = cTransform(opt_transport_return['u_last'], self.inputs['X_source'],
                                         self.inputs['X_target'],
                                         j, beta)

        indices = np.zeros((n_sub, 2))
        for k, it in enumerate(source_indices):
            indices[k, 0] = it
            cost_x = cost(self.inputs['X_target'], self.inputs['X_source'][it])
            arg = np.exp((opt_transport_return['u_last'][it] + u_ce_storage - cost_x) / eps)
            indices[k, 1] = np.argmax(arg)

        indices = np.asarray(indices, dtype=int)

        # with reweighting
        u_ce_storage_two = np.zeros(self.J)
        for j in range(self.J):
            u_ce_storage_two[j] = cTransform(opt_rew['u_last_two'], self.inputs['X_source'], self.inputs['X_target'],
                                             j, beta)

        indices_two = np.zeros((n_sub, 2))
        for k, it in enumerate(source_indices):
            indices_two[k, 0] = it
            cost_x = cost(self.inputs['X_target'], self.inputs['X_source'][it])
            arg = np.exp((opt_rew['u_last_two'][it] + u_ce_storage_two - cost_x) / eps)
            indices_two[k, 1] = np.argmax(arg)

        indices_two = np.asarray(indices_two, dtype=int)

        X_target_lag = self.inputs['X_tar_display'].copy()
        X_target_lag[:, 0] = X_target_lag[:, 0] + 3500


if __name__ == '__main__':
    data = {"X_source": np.asarray(pd.read_csv('data/W2_1_values.csv',
                                               usecols=np.arange(1, 8))[['CD4', 'CD8']]),

            'X_target': np.asarray(pd.read_csv('data/W2_7_values.csv',
                                               usecols=np.arange(1, 8))[['CD4', 'CD8']]),

            'X_sou_display': np.asarray(pd.read_csv('data/W2_1_values.csv',
                                                    usecols=np.arange(1, 8))[['CD4', 'CD8']]),

            'Lab_source': np.asarray(pd.read_csv('data/W2_1_clust.csv',
                                                 usecols=[1])['x'] >= 6, dtype=int),

            'Lab_target': np.asarray(pd.read_csv('data/W2_7_clust.csv',
                                                 usecols=[1])['x'] >= 6, dtype=int),

            'X_tar_display': np.asarray(pd.read_csv('data/W2_7_values.csv',
                                                    usecols=np.arange(1, 8))[['CD4', 'CD8']]),

            'names_pop': ['CD8', 'CD4']}
    test1 = TwoClassesTwoDimension(data)
    print(test1.estimateCytOpT())
