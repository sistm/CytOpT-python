# Copyright (C) 2022, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from CytOpT.labelPropSto import cost
# __all__ = ['cytopt_minmax']


def func_f(X_s, X_t, lbd, eps, j, u, G):
    """
    Compute the function f inside the expectation at the point (y_j, u).
    """

    arg1 = (u - cost(X_s, X_t[j])) / eps
    t1 = logsumexp(arg1)

    arg2 = -(G.T).dot(u) / lbd
    t2 = logsumexp(arg2)

    result = -eps * t1 - lbd * t2

    return result


def grad_f(X_s, X_t, lbd, eps, j, u, G):
    """
    Compute the gradient with respect to u of the function f inside the expectation
    """

    arg1 = (u - cost(X_s, X_t[j])) / eps
    cor1 = np.max(arg1)
    vec1 = np.exp(arg1 - cor1)
    t1 = - vec1 / np.sum(vec1)

    arg2 = -(G.T).dot(u) / lbd
    cor2 = np.max(arg2)
    vec2 = np.exp(arg2 - cor2)
    t2 = G.dot(vec2) / np.sum(vec2)

    return t1 + t2


def Gam_mat(Lab_source):
    """
    Compute the Gamma matrix that allows to pass from the class proportions to the weight vector
    """
    if Lab_source.min() == 0:
        K = int(Lab_source.max()) + 1
        I = Lab_source.shape[0]
        Gamma = np.zeros((I, K))

        for k in range(K):
            Gamma[:, k] = 1 / np.sum(Lab_source == k) * np.asarray(Lab_source == k, dtype=float)


    else:
        K = int(Lab_source.max())
        I = Lab_source.shape[0]
        Gamma = np.zeros((I, K))

        for k in range(K):
            Gamma[:, k] = 1 / np.sum(Lab_source == k + 1) * np.asarray(Lab_source == k + 1, dtype=float)

    return Gamma


def stomax(X_s, X_t, G, lbd, eps, n_iter):
    """
    Robbins-Monro algorithm to compute an approximate of the vector u^* solution of the maximization problem
    """

    I = X_s.shape[0]
    U = np.zeros(I)

    # Step size policy
    gamma = I * eps / 1.9
    c = 0.51

    sample = np.random.choice(I, n_iter)

    for n in range(n_iter):
        idx = sample[n]
        grd = grad_f(X_s, X_t, lbd, eps, idx, U, G)

        U = U + gamma / (n + 1) ** c * grd

    return U


# cytopt
def cytopt_minmax(X_s, X_t, Lab_source, eps=0.0001, lbd=0.0001, n_iter=4000,
                  step=5, power=0.99, theta_true=None, monitoring=False):
    """ CytOpT algorithm. This methods is designed to estimate the proportions of cells in an unclassified Cytometry
    data set denoted X_t. CytOpT is a supervised method that leverage the classification denoted Lab_source associated
    to the flow cytometry data set X_s. The estimation relies on the resolution of an optimization problem.
    The optimization problem of this function involves an additional regularization term lambda. This regularization
    allows the application of a simple stochastic gradient-ascent to solve the optimization problem. We advocate the
    use of this method as it is faster than 'cytopt_desasc'.

    :param X_s: np.array of shape (n_samples_source, n_biomarkers). The source cytometry data set.
    :param X_t: np.array of shape (n_samples_target, n_biomarkers). The target cytometry data set.
    :param Lab_source: np.array of shape (n_samples_source,). The classification of the source data set.
    :param eps: float, ``default=0.0001``. Regularization parameter of the Wasserstein distance. This parameter must be positive.
    :param lbd: float, ``default=0.0001``. Additionnal regularization parameter of the Minmax swapping optimization method.
        This parameter lbd should be greater or equal to eps.
    :param n_iter: int, ``default=10000``. Number of iterations of the stochastic gradient ascent.
    :param step: float, ``default=5``. Multiplication factor of the stochastic gradient ascent step-size policy for
        the minmax optimization method.
    :param power: float, ``default=0.99``. Decreasing rate for the step-size policy of the stochastic gradient ascent for
        the Minmax swapping optimization method. The step-size decreases at a rate of 1/n^power.
    :param theta_true: np.array of shape (K,), ``default=None``. This array stores the true proportions of the K type of
        cells estimated in the target data set. This parameter is required if the user enables the monitoring option.
    :param monitoring: bool, ``default=False``. When set to true, the evolution of the Kullback-Leibler between the
        estimated proportions and the benchmark proportions is tracked and stored.

    :return:
            - hat_theta - np.array of shape (K,), where K is the number of different type of cell populations in the source data set.
            - KL_storage - np.array of shape (n_iter,) This array stores the evolution of the Kullback-Leibler divergence between the estimate and benchmark proportions, if monitoring==True and the theta_true variable is completed.

    Reference:
     Paul Freulon, Jérémie Bigot,and Boris P. Hejblum CytOpT: Optimal Transport with Domain Adaptation for Interpreting Flow Cytometry data,
     arXiv:2006.09003 [stat.AP].
    """

    I, J = X_s.shape[0], X_t.shape[0]
    U = np.zeros(I)
    D = Gam_mat(Lab_source)
    theta_hat = None

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
    if monitoring:
        for it in range(1, n_iter):
            idx = sample[it]
            grd = grad_f(X_s, X_t, lbd, eps, idx, U, D)
            U = U + gamma / (it + 1) ** c * grd

            # Computation of the estimate h_hat
            arg = -D.T.dot(U) / lbd
            M = np.max(arg)

            theta_hat = np.exp(arg - M)
            theta_hat = theta_hat / theta_hat.sum()
            KL_current = np.sum(theta_hat * np.log(theta_hat / theta_true))
            KL_storage[it] = KL_current

        return [theta_hat, KL_storage]
    else:
        G = Gam_mat(Lab_source)
        u_hat = stomax(X_s, X_t, G, lbd, eps, n_iter)

        # computation of the estimate of the class proportions
        theta_hat = np.exp(-(G.T).dot(u_hat) / lbd)
        theta_hat = theta_hat / theta_hat.sum()
        return [theta_hat]


if __name__ == '__main__':
    # Source Data
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
    h_target = np.zeros(10)
    for k in range(10):
        h_target[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

    res = cytopt_minmax(X_source, X_target, Lab_source, theta_true=h_target, monitoring=True)
