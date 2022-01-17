# Copyright (C) 2022, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
import pandas as pd
from scipy.stats import entropy
from CytOpT.CytOpt_Descent_Ascent import gammatrix, cost


# __all__ = ['cytopt_minmax']

def grad_f(lbd, eps, X_s, X_t, j, u, D):
    """    Compute the gradient with respect to u of the function f inside the expectation

    :param lbd:
    :param eps:
    :param X_s:
    :param X_t:
    :param j:
    :param u:
    :param D:
    :return:
    """
    arg1 = (u - cost(X_s, X_t[j])) / eps
    cor1 = np.max(arg1)
    vec1 = np.exp(arg1 - cor1)
    t1 = - vec1 / np.sum(vec1)

    arg2 = -D.T.dot(u) / lbd
    cor2 = np.max(arg2)
    vec2 = np.exp(arg2 - cor2)
    t2 = D.dot(vec2) / np.sum(vec2)

    return t1 + t2


# cytopt
def cytopt_minmax(X_s, X_t, Lab_source, eps=0.0001, lbd=0.0001, n_iter=4000,
                  step=5, power=0.99, theta_true=0, monitoring=False):
    """ Robbins-Monro algorithm to compute an approximate of the vector u^* solution of the maximization problem
    At each step, it is possible to evaluate the vector h_hat to study the convergence of this algorithm.

    :param X_s: a cytometry dataframe. The columns correspond to the different biological markers tracked.
                One line corresponds to the cytometry measurements performed on one cell. The classification
                of this Cytometry data set must be provided with the Lab_source parameters.
    :param X_t: a cytometry dataframe. The columns correspond to the different biological markers tracked.
                One line corresponds to the cytometry measurements performed on one cell. The CytOpt algorithm
                targets the cell type proportion in this Cytometry data set.
    :param Lab_source: a vector of length ``n`` Classification of the ``X_s`` cytometry data set
    :param eps: an float value of regularization parameter of the Wasserstein distance. Default is ``1e-04``.
    :param lbd:  an float constant that multiply the step-size policy. Default is ``1e-04``.
    :param n_iter:  an integer Constant that iterate method select. Default is ``4000``.
    :param step: an integer constant that multiply the step-size policy. Default is ``5``.
    :param power: an float constant the step size policy of the gradient ascent method is step/n^power. Default is ``0.99``.
    :param theta_true: If available, the true proportions in the target data set ``X_s``. It allows to assess
                        the gap between the estimate of our method and the estimate of the cell type proportions derived from
                        manual gating.
    :param monitoring: a logical flag indicating to possibly monitor the gap between the estimated proprotions and the manual
                        gold-standard. Default is ``FALSE``.
    :return:
    """

    I = X_s.shape[0]
    J = X_t.shape[0]
    U = np.zeros(I)
    D = gammatrix(X_s, Lab_source)[0]

    # Step size policy
    if step == 0:
        gamma = I * eps / 1.9
    else:
        gamma = step

    if power == 0:
        c = 0.51
    else:
        c = power

    sample = np.random.choice(J, n_iter)

    # Storage of the KL divergence between theta_hat and theta_true
    KL_storage = np.zeros(n_iter)

    for it in range(1, n_iter):
        idx = sample[it]
        grd = grad_f(lbd, eps, X_s, X_t, idx, U, D)
        U = U + gamma / (it + 1) ** c * grd

        if monitoring:
            # Computation of the estimate h_hat
            arg = -D.T.dot(U) / lbd
            M = np.max(arg)

            theta_hat = np.exp(arg - M)
            theta_hat = theta_hat / theta_hat.sum()
            if it % 100 == 0:
                print('Iteration ', it, ' - Current theta_hat: \n', theta_hat)
            KL_storage[it] = entropy(pk=theta_hat, qk=theta_true)

        arg = -D.T.dot(U) / lbd
        M = np.max(arg)

        theta_hat = np.exp(arg - M)
        theta_hat = theta_hat / theta_hat.sum()

    return [theta_hat, KL_storage]


if __name__ == '__main__':
    # Source Data
    Stanford1A_values = pd.read_csv('../tests/data/W2_1_values.csv',
                                    usecols=np.arange(1, 8))
    Stanford1A_clust = pd.read_csv('../tests/data/W2_1_clust.csv',
                                   usecols=[1])

    # Target Data
    Stanford3A_values = pd.read_csv('../tests/data/W2_7_values.csv',
                                    usecols=np.arange(1, 8))
    Stanford3A_clust = pd.read_csv('../tests/data/W2_7_clust.csv',
                                   usecols=[1])

    X_source = np.asarray(Stanford1A_values)
    X_target = np.asarray(Stanford3A_values)
    Lab_source = np.asarray(Stanford1A_clust['x'])
    Lab_target = np.asarray(Stanford3A_clust['x'])

    cytopt_minmax(X_source, X_target, Lab_source, monitoring=True)
