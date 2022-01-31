# Copyright (C) 2022, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)


import numpy as np
import pandas as pd
from CytOpT.Label_Prop_sto import Robbins_Wass


# __all__ = ['cytopt_desasc', 'Label_Prop_sto']


def diff_simplex(h):
    """
    Computation of the Jacobian matrix of the softmax function.

    :param h:
    :return:
    """

    try:
        h = np.array(h, np.float)
    except Exception as e:
        print(e)
    K = len(h)
    Diff = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if i == j:
                Diff[i, j] = (np.exp(h[i]) * np.sum(np.exp(h)) - np.exp(2 * h[i])) / (np.sum(np.exp(h)) ** 2)
            else:
                Diff[i, j] = - np.exp(h[i] + h[j]) / (np.sum(np.exp(h)) ** 2)
    return Diff


def gammatrix(X_s, Lab_source):
    """    Computation of the operator D that maps the class proportions with the weights.

    :param X_s:
    :param Lab_source:
    :return:
    """

    I = X_s.shape[0]
    if min(Lab_source) == 0:
        K = int(max(Lab_source))
        D = np.zeros((I, K + 1))
        for k in range(K + 1):
            D[:, k] = 1 / np.sum(Lab_source == k) * np.asarray(Lab_source == k, dtype=float)

        h = np.ones(K + 1)

    else:
        K = int(max(Lab_source))
        D = np.zeros((I, K))
        for k in range(K):
            D[:, k] = 1 / np.sum(Lab_source == k + 1) * np.asarray(Lab_source == k + 1, dtype=float)

        h = np.ones(K)
    return D, h


# cytopt_desasc
def cytopt_desasc(X_s, X_t, Lab_source,
                  eps=0.0001, n_it_grad=4000, n_it_sto=10,
                  step_grad=50, cont=True, theta_true=None, monitoring=True):
    """ CytOpT algorithm. This methods is designed to estimate the proportions of cells in an unclassified Cytometry
    data set denoted X_t. CytOpT is a supervised method that leverage the classification denoted Lab_source associated
    to the flow cytometry data set X_s. The estimation relies on the resolution of an optimization problem.
    The optimization problem of this function is solved with a descent-ascent optimization procedure.

    :param X_s: np.array of shape (n_samples_source, n_biomarkers). The source cytometry data set.
    :param X_t: np.array of shape (n_samples_target, n_biomarkers). The target cytometry data set.
    :param Lab_source: np.array of shape (n_samples_source,). The classification of the source data set.
    :param eps: float, ``default=0.0001``. Regularization parameter of the Wasserstein distance.
        This parameter must be positive.
    :param n_it_grad: int, ``default=10000``. Number of iterations of the outer loop of the descent-ascent optimization method.
        This loop corresponds to the descent part of descent-ascent strategy.
    :param n_it_sto: int, ``default = 10``. Number of iterations of the inner loop of the descent-ascent optimization method.
        This loop corresponds to the stochastic ascent part of this optimization procedure.
    :param step_grad: float, ``default=10``. Constant step_size policy for the gradient descent of the descent-ascent optimization strategy.
    :param cont: bool, ``default=True``. When set to true, the progress is displayed.
    :param theta_true: np.array of shape (K,), ``default=None``. This array stores the true proportions of the K type of
        cells estimated in the target data set. This parameter is required if the user enables the monitoring option.
    :param monitoring: bool, ``default=False``. When set to true, the evolution of the Kullback-Leibler between the
        estimated proportions and the benchmark proportions is tracked and stored.

    :return:
        - hat_theta - np.array of shape (K,), where K is the number of different type of cell populations in the source data set.
        - KL_storage - np.array of shape (n_out, ). This array stores the evolution of the Kullback-Leibler divergence between the estimate and benchmark proportions, if monitoring==True.
    """

    print('\n Epsilon: ', eps)
    I, J, prop_classes_new = X_s.shape[0], X_t.shape[0], 0

    # Definition of the operator D that maps the class proportions with the weights.
    D, h = gammatrix(X_s, Lab_source)

    # Weights of the target distribution
    beta = 1 / J * np.ones(J)

    # Storage of the KL between theta_hat and theta_true
    KL_Storage = np.zeros(n_it_grad)

    # Descent-Ascent procedure
    if monitoring:
        for i in range(n_it_grad):
            prop_classes = np.exp(h)
            prop_classes = prop_classes / np.sum(prop_classes)
            Dif = diff_simplex(h)
            alpha_mod = D.dot(prop_classes)
            f_star_hat = Robbins_Wass(X_s, X_t, alpha_mod, beta, eps=eps, n_iter=n_it_sto)[0]
            h = h - step_grad * (D.dot(Dif)).T.dot(f_star_hat)
            prop_classes_new = np.exp(h)
            prop_classes_new = prop_classes_new / np.sum(prop_classes_new)
            if i % 100 == 0:
                if cont:
                    print('Iteration ', i)
                    print('Current h_hat')
                    print(prop_classes_new)
            KL_current = np.sum(prop_classes_new * np.log(prop_classes_new / theta_true))
            KL_Storage[i] = KL_current

        return [prop_classes_new, KL_Storage]

    else:
        for i in range(n_it_grad):
            prop_classes = np.exp(h)
            prop_classes = prop_classes / np.sum(prop_classes)
            Dif = diff_simplex(h)
            alpha_mod = D.dot(prop_classes)
            f_star_hat = Robbins_Wass(X_s, X_t, alpha_mod, beta, eps=eps, n_iter=n_it_sto)[0]
            h = h - step_grad * (D.dot(Dif)).T.dot(f_star_hat)
            prop_classes_new = np.exp(h)
            prop_classes_new = prop_classes_new / np.sum(prop_classes_new)
            if i % 100 == 0:
                if cont:
                    print('Iteration ', i)
                    print('Current h_hat')
                    print(prop_classes_new)
        return [prop_classes_new]


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
    h_target = np.zeros(10)
    for k in range(10):
        h_target[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

    res = cytopt_desasc(X_source, X_target, Lab_source,
                        eps=0.0001, n_it_grad=1000, n_it_sto=10,
                        step_grad=50, theta_true=h_target, monitoring=False)
