# Copyright (C) 2022, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)
from time import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import MinMaxScaler

from CytOpT.labelPropSto import cost
# __all__ = ['cytopt_minmax']


def funcF(xSource, xTarget, lbd, eps, j, u, G):
    """
    Compute the function f inside the expectation at the point (y_j, u).
    """

    arg1 = (u - cost(xSource, xTarget[j])) / eps
    t1 = logsumexp(arg1)

    arg2 = -(G.T).dot(u) / lbd
    t2 = logsumexp(arg2)

    result = -eps * t1 - lbd * t2

    return result


def gradF(xSource, xTarget, lbd, eps, j, u, G):
    """
    Compute the gradient with respect to u of the function f inside the expectation
    """

    arg1 = (u - cost(xSource, xTarget[j])) / eps
    cor1 = np.max(arg1)
    vec1 = np.exp(arg1 - cor1)
    t1 = - vec1 / np.sum(vec1)

    arg2 = -(G.T).dot(u) / lbd
    cor2 = np.max(arg2)
    vec2 = np.exp(arg2 - cor2)
    t2 = G.dot(vec2) / np.sum(vec2)

    return t1 + t2


def GamMat(labSource):
    """
    Compute the Gamma matrix that allows to pass from the class proportions to the weight vector
    """
    if labSource.min() == 0:
        K = int(labSource.max()) + 1
        I = labSource.shape[0]
        Gamma = np.zeros((I, K))

        for k in range(K):
            Gamma[:, k] = 1 / np.sum(labSource == k) * np.asarray(labSource == k, dtype=float)


    else:
        K = int(labSource.max())
        I = labSource.shape[0]
        Gamma = np.zeros((I, K))

        for k in range(K):
            Gamma[:, k] = 1 / np.sum(labSource == k + 1) * np.asarray(labSource == k + 1, dtype=float)

    return Gamma


def stomax(xSource, xTarget, G, lbd, eps, nIter):
    """
    Robbins-Monro algorithm to compute an approximate of the vector u^* solution of the maximization problem
    """

    I = xSource.shape[0]
    U = np.zeros(I)

    # Step size policy
    gamma = I * eps / 1.9
    c = 0.51

    sample = np.random.choice(I, nIter)

    for n in range(nIter):
        idx = sample[n]
        grd = gradF(xSource, xTarget, lbd, eps, idx, U, G)

        U = U + gamma / (n + 1) ** c * grd

    return U


# cytopt
def cytoptMinmax(xSource, xTarget, labSource, eps=0.0001, lbd=0.0001, nIter=4000, cont=True,
                 step=5, power=0.99, thetaTrue=None,
                 monitoring=False, thresholding=True, minMaxScaler=True):
    """ CytOpT algorithm. This methods is designed to estimate the proportions of cells in an unclassified Cytometry
    data set denoted xTarget. CytOpT is a supervised method that leverage the classification denoted labSource associated
    to the flow cytometry data set xSource. The estimation relies on the resolution of an optimization problem.
    The optimization problem of this function involves an additional regularization term lambda. This regularization
    allows the application of a simple stochastic gradient-ascent to solve the optimization problem. We advocate the
    use of this method as it is faster than 'cytopt_desasc'.

    :param xSource: np.array of shape (n_samples_source, n_biomarkers). The source cytometry data set.
    :param xTarget: np.array of shape (n_samples_target, n_biomarkers). The target cytometry data set.
    :param labSource: np.array of shape (n_samples_source,). The classification of the source data set.
    :param eps: float, ``default=0.0001``. Regularization parameter of the Wasserstein distance. This parameter must be positive.
    :param lbd: float, ``default=0.0001``. Additionnal regularization parameter of the Minmax swapping optimization method.
        This parameter lbd should be greater or equal to eps.
    :param nIter: int, ``default=10000``. Number of iterations of the stochastic gradient ascent.
    :param cont: bool, ``default=True``. When set to true, the progress is displayed.
    :param step: float, ``default=5``. Multiplication factor of the stochastic gradient ascent step-size policy for
        the minmax optimization method.
    :param power: float, ``default=0.99``. Decreasing rate for the step-size policy of the stochastic gradient ascent for
        the Minmax swapping optimization method. The step-size decreases at a rate of 1/n^power.
    :param thetaTrue: np.array of shape (K,), ``default=None``. This array stores the true proportions of the K type of
        cells estimated in the target data set. This parameter is required if the user enables the monitoring option.
    :param monitoring: bool, ``default=False``. When set to true, the evolution of the Kullback-Leibler between the
        estimated proportions and the benchmark proportions is tracked and stored.
    :param minMaxScaler: bool, ``default = True``. When set to True, the source and target data sets are scaled in [0,1]^d,
        where d is the  number of biomarkers monitored.
    :param thresholding: bool, ``default = True``. When set to True, all the coefficients of the source and target data sets
        are replaced by their positive part. This preprocessing is relevant for Cytometry Data as the signal acquisition of
        the cytometer can induce convtrived negative values.

    :return:
            - hat_theta - np.array of shape (K,), where K is the number of different type of cell populations in the source data set.
            - KL_storage - np.array of shape (nIter,) This array stores the evolution of the Kullback-Leibler divergence between the estimate and benchmark proportions, if monitoring==True and the thetaTrue variable is completed.

    Reference:
     Paul Freulon, Jérémie Bigot,and Boris P. Hejblum CytOpT: Optimal Transport with Domain Adaptation for Interpreting Flow Cytometry data,
     arXiv:2006.09003 [stat.AP].
    """
    if thresholding:
        xSource = xSource * (xSource > 0)
        xTarget = xTarget * (xTarget > 0)

    if minMaxScaler:
        Scaler = MinMaxScaler()
        xSource = Scaler.fit_transform(xSource)
        xTarget = Scaler.fit_transform(xTarget)

    I, J = xSource.shape[0], xTarget.shape[0]
    U = np.zeros(I)
    D = GamMat(labSource)
    thetaHat = None

    # Step size policy
    if step == 0:
        gamma = I * eps / 1.9
    else:
        gamma = step

    if power == 0:
        c = 0.51
    else:
        c = power

    sample = np.random.choice(J, nIter)

    # Storage of the KL divergence between thetaHat and thetaTrue
    KL_storage = np.zeros(nIter)
    print("Running MinMax optimization...")
    t0 = time()
    if monitoring:
        for it in range(1, nIter):
            idx = sample[it]
            grd = gradF(xSource, xTarget, lbd, eps, idx, U, D)
            U = U + gamma / (it + 1) ** c * grd

            # Computation of the estimate h_hat
            arg = -D.T.dot(U) / lbd
            M = np.max(arg)

            thetaHat = np.exp(arg - M)
            thetaHat = thetaHat / thetaHat.sum()
            if it % 1000 == 0:
                if cont:
                    print('Iteration ', it)
                    print('Current h_hat')
                    print(thetaHat)
            KL_current = np.sum(thetaHat * np.log(thetaHat / thetaTrue))
            KL_storage[it] = KL_current
        elapsed_time_minmax = time() - t0
        print("Done (", round(elapsed_time_minmax, 3), "s)")
        return [thetaHat, KL_storage]
    else:
        G = GamMat(labSource)
        uHat = stomax(xSource, xTarget, G, lbd, eps, nIter)

        # computation of the estimate of the class proportions
        thetaHat = np.exp(-(G.T).dot(uHat) / lbd)
        thetaHat = thetaHat / thetaHat.sum()
        elapsed_time_minmax = time() - t0
        print("Done (", round(elapsed_time_minmax, 3), "s)")
        return [thetaHat]


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

    xSource = np.asarray(Stanford1A_values)
    xTarget = np.asarray(Stanford3A_values)
    labSource = np.asarray(Stanford1A_clust['x'])
    LabTarget = np.asarray(Stanford3A_clust['x'])
    hTarget = np.zeros(10)
    for k in range(10):
        hTarget[k] = np.sum(LabTarget == k + 1) / len(LabTarget)

    res = cytoptMinmax(xSource, xTarget, labSource, thetaTrue=hTarget, monitoring=True)
