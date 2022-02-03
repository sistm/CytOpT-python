# Copyright (C) 2022, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from CytOpT.labelPropSto import robbinsWass


# __all__ = ['cytoptDesasc']


def diffSimplex(h):
    """
    Computation of the Jacobian matrix of the softmax function.

    :param h:
    :return:
    """

    try:
        h = np.array(h)
    except Exception as e:
        print(e)
    K = len(h)
    Diff = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                Diff[i, j] = (np.exp(h[i]) * np.sum(np.exp(h)) - np.exp(2 * h[i])) / (np.sum(np.exp(h)) ** 2)
            else:
                Diff[i, j] = - np.exp(h[i] + h[j]) / (np.sum(np.exp(h)) ** 2)
    return Diff


def gammatrix(xSource, labSource):
    """    Computation of the operator D that maps the class proportions with the weights.

    :param xSource:
    :param labSource:
    :return:
    """

    I = xSource.shape[0]
    if min(labSource) == 0:
        K = int(max(labSource))
        D = np.zeros((I, K + 1))
        for it in range(K + 1):
            D[:, it] = 1 / np.sum(labSource == it) * np.asarray(labSource == it, dtype=float)

        h = np.ones(K + 1)

    else:
        K = int(max(labSource))
        D = np.zeros((I, K))
        for it in range(K):
            D[:, it] = 1 / np.sum(labSource == it + 1) * np.asarray(labSource == it + 1, dtype=float)

        h = np.ones(K)
    return D, h


# cytopt_desasc
def cytoptDesasc(xSource, xTarget, labSource,
                 eps=1, nItGrad=4000, nItSto=10,
                 stepGrad=1 / 1000, cont=True, thetaTrue=None, monitoring=True):
    """ CytOpT algorithm. This methods is designed to estimate the proportions of cells in an unclassified Cytometry
    data set denoted xTarget. CytOpT is a supervised method that leverage the classification denoted labSource associated
    to the flow cytometry data set xSource. The estimation relies on the resolution of an optimization problem.
    The optimization problem of this function is solved with a descent-ascent optimization procedure.

    :param xSource: np.array of shape (n_samples_source, n_biomarkers). The source cytometry data set.
    :param xTarget: np.array of shape (n_samples_target, n_biomarkers). The target cytometry data set.
    :param labSource: np.array of shape (n_samples_source,). The classification of the source data set.
    :param eps: float, ``default=0.0001``. Regularization parameter of the Wasserstein distance.
        This parameter must be positive.
    :param nItGrad: int, ``default=10000``. Number of iterations of the outer loop of the descent-ascent optimization method.
        This loop corresponds to the descent part of descent-ascent strategy.
    :param nItSto: int, ``default = 10``. Number of iterations of the inner loop of the descent-ascent optimization method.
        This loop corresponds to the stochastic ascent part of this optimization procedure.
    :param stepGrad: float, ``default=10``. Constant step_size policy for the gradient descent of the descent-ascent optimization strategy.
    :param cont: bool, ``default=True``. When set to true, the progress is displayed.
    :param thetaTrue: np.array of shape (K,), ``default=None``. This array stores the true proportions of the K type of
        cells estimated in the target data set. This parameter is required if the user enables the monitoring option.
    :param monitoring: bool, ``default=False``. When set to true, the evolution of the Kullback-Leibler between the
        estimated proportions and the benchmark proportions is tracked and stored.

    :return:
        - hat_theta - np.array of shape (K,), where K is the number of different type of cell populations in the source data set.
        - KLStorage - np.array of shape (n_out, ). This array stores the evolution of the Kullback-Leibler divergence between the estimate and benchmark proportions, if monitoring==True.
    """
    I, J, propClassesNew = xSource.shape[0], xTarget.shape[0], 0

    # Definition of the operator D that maps the class proportions with the weights.
    D, h = gammatrix(xSource, labSource)

    # Weights of the target distribution
    beta = 1 / J * np.ones(J)

    # Storage of the KL between theta_hat and thetaTrue
    KLStorage = np.zeros(nItGrad)

    # Descent-Ascent procedure
    print("Running Desent-ascent optimization...")
    t0 = time()
    if monitoring:
        for i in range(nItGrad):
            propClasses = np.exp(h)
            propClasses = propClasses / np.sum(propClasses)
            Dif = diffSimplex(h)
            alphaMod = D.dot(propClasses)
            fStarHat = robbinsWass(xSource, xTarget, alphaMod, beta, eps=eps, nIter=nItSto)[0]
            h = h - stepGrad * ((D.dot(Dif)).T).dot(fStarHat)
            propClassesNew = np.exp(h)
            propClassesNew = propClassesNew / np.sum(propClassesNew)
            if i % 1000 == 0:
                if cont:
                    print('Iteration ', i)
                    print('Current h_hat')
                    print(propClassesNew)
            KLCurrent = np.sum(propClassesNew * np.log(propClassesNew / thetaTrue))
            KLStorage[i] = KLCurrent
        elapsed_time_desac = time() - t0
        print("Done (", round(elapsed_time_desac, 3), "s)")
        return [propClassesNew, KLStorage]

    else:
        for i in range(nItGrad):
            propClasses = np.exp(h)
            propClasses = propClasses / np.sum(propClasses)
            Dif = diffSimplex(h)
            alphaMod = D.dot(propClasses)
            fStarHat = robbinsWass(xSource, xTarget, alphaMod, beta, eps=eps, nIter=nItSto)[0]
            h = h - stepGrad * ((D.dot(Dif)).T).dot(fStarHat)
            propClassesNew = np.exp(h)
            propClassesNew = propClassesNew / np.sum(propClassesNew)
            if i % 1000 == 0:
                if cont:
                    print('Iteration ', i)
                    print('Current h_hat')
                    print(propClassesNew)
        elapsed_time_desac = time() - t0
        print("Done (", round(elapsed_time_desac, 3), "s)")
        return [propClassesNew]


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

    xSource = np.asarray(Stanford1A_values[['CD4', 'CD8']])
    xTarget = np.asarray(Stanford3A_values[['CD4', 'CD8']])
    labSource = np.asarray(Stanford1A_clust['x'] >= 6, dtype=int)
    labTarget = np.asarray(Stanford3A_clust['x'] >= 6, dtype=int)

    xSource = xSource * (xSource > 0)
    xTarget = xTarget * (xTarget > 0)

    scaler = MinMaxScaler()
    xSource = scaler.fit_transform(xSource)
    xTarget = scaler.fit_transform(xTarget)

    res = cytoptDesasc(xSource, xTarget, labSource,
                       eps=0.0005, nItGrad=1000, nItSto=10,
                       stepGrad=1, monitoring=False)
    print(res)
