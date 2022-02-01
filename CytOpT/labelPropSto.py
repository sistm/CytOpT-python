# Copyright (C) 2021, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
from scipy.special import logsumexp


def cost(xSource, y):
    """ Squared euclidean distance between y and the I points of xSource.

    :param xSource: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param y:
    :return:
    """

    diff = xSource - y
    return np.linalg.norm(diff, axis=1) ** 2


def cTransform(f, xSource, xTarget, j, beta, eps=0.1):
    """     Calculate the c_transform of f in the non regularized case if eps=0.
    Otherwise, it computes the smooth c_transform with respect to the usual entropy.

    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param xSource: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param xTarget: np.array of shape (n_obs_target, dimension). Support of the target distribution
    :param j:
    :param beta: np.array of shape (n_obs_target,). Weights of the target  distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.

    :return:
    """
    if eps == 0:
        costY = cost(xSource, xTarget[j])
        return np.min(costY - f)

    else:
        arg = (f - cost(xSource, xTarget[j])) / eps
        return eps * (np.log(beta[j]) - logsumexp(arg))


def gradH(f, xSource, y, alpha, eps=0.1):
    """ This function calculates the gradient of the function that we aim to maximize.
    The expectation of this function computed at a maximizer equals the wasserstein disctance,
    or its regularized counterpart.

    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param xSource: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param y:
    :param alpha: np.array of shape (n_obs_source,). Weights of the source distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.

    :return:
    """

    if eps == 0:
        costY = cost(xSource, y)
        i_star = np.argmin(costY - f)
        to_return = alpha.copy()
        to_return[i_star] = alpha[i_star] - 1
        return to_return

    else:
        arg = (f - cost(xSource, y)) / eps
        Mx = np.max(arg)
        pi = np.exp(arg - Mx)
        pi = pi / pi.sum()
        return alpha - pi


def hFunction(f, xSource, xTarget, j, alpha, beta, eps=0.1):
    """ Calculate the function h whose expectation equals the semi-dual loss.
    Maximizing the semi-dual loss allows us to compute the wasserstein distance.

    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param xSource: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param xTarget: np.array of shape (n_obs_target, dimension). Support of the target distribution
    :param j:
    :param beta: np.array of shape (n_obs_target,). Weights of the target  distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.
    :param alpha: np.array of shape (n_obs_source,). Weights of the source distribution.

    :return:
    """
    if eps == 0:
        return np.sum(f * alpha) + cTransform(f, xSource, xTarget[j])

    else:
        return np.sum(f * alpha) + cTransform(f, xSource, xTarget, j, beta, eps) - eps


def robbinsWass(xSource, xTarget, alpha, beta, eps=0.1, nIter=10000):
    """ Function that calculates the approximation of the optimal dual vector associated
    to the source distribution. The regularized optimal-transport problem is computed between a distribution with
    support xSource and weights alpha, and a distribution with support xTarget and weights beta. This function solves the
    semi-dual formulation of the regularized OT problem with the stochastic algorithm of Robbins-Monro.

    :param xSource: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param xTarget: np.array of shape (n_obs_target, dimension). Support of the target distribution
    :param alpha: np.array of shape (n_obs_source,). Weights of the source distribution.
    :param beta: np.array of shape (n_obs_target,). Weights of the target  distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.
    :param nIter: int, ``default=10000``. Number of iterations of the Robbins-Monro algorithm.

    :return:
        - f - np.array of shape (n_obs_source,). Optimal kantorovich potential associated to the source distribution.
    """
    nIter = int(nIter)
    I, J = xSource.shape[0], xTarget.shape[0]

    # Definition of the step size policy
    gamma = eps / (1.9 * min(beta))
    c = 0.51

    # Storage of the estimates
    wHatStorage = np.zeros(nIter)
    sigmaHatStorage = np.zeros(nIter)
    hEpsStorage = np.zeros(nIter)
    hEpsSquare = np.zeros(nIter)

    # Sampling according to the target distribution.
    sample = np.random.choice(a=np.arange(J), size=nIter, p=beta)

    # Initialisation of the dual vector f.
    f = np.random.random(I)
    f = f - np.mean(f)

    # First iteration to start the loop.
    wHatStorage[0] = hFunction(f, xSource, xTarget, sample[0], alpha, beta, eps)
    hEpsStorage[0] = hFunction(f, xSource, xTarget, sample[0], alpha, beta, eps)
    hEpsSquare[0] = hFunction(f, xSource, xTarget, sample[0], alpha, beta, eps) ** 2

    # Robbins-Monro Algorithm.

    for k in range(1, nIter):
        # Sample from the target measure.
        j = sample[k]

        # Update of f.
        f = f + gamma / ((k + 1) ** c) * gradH(f, xSource, xTarget[j, :], alpha, eps)
        hEpsStorage[k] = hFunction(f, xSource, xTarget, j, alpha, beta, eps)

        # Update of the estimator of the regularized Wasserstein distance.
        wHatStorage[k] = k / (k + 1) * wHatStorage[k - 1] + 1 / (k + 1) * hEpsStorage[k]

        # Update of the estimator of the asymptotic variance
        hEpsSquare[k] = k / (k + 1) * hEpsSquare[k - 1] + 1 / (k + 1) * hEpsStorage[k] ** 2
        sigmaHatStorage[k] = hEpsSquare[k] - wHatStorage[k] ** 2

    return [f, wHatStorage, sigmaHatStorage]


def labelPropSto(labSource, f, X, Y, alpha, beta, eps=0.0001):
    """     Function that calculates a classification of the target data with an optimal-transport based soft assignment.
    For optimal result, the source distribution must be re-weighted thanks to the estimation of the class proportions
    in the target data set.  This estimation can be produced with the Cytopt function. To compute an optimal dual
    vector f associated to the source distribution, we advocate the use of the robbinsWass function with a CytOpT
    re-weighting of the source distribution.

    :param labSource: np.array of shape (X.shape[0],). The labels associated to the source data set X_s
    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param X: np.array of shape (n_obs_source, dimension). The support of the source distribution.
    :param Y: np.array of shape (n_obs_target, dimension). The support of the target distribution.
    :param alpha: np.array of shape (n_obs_source,). The weights of the source distribution.
    :param beta: np.array of shape (n_obs_target,). The weights of the target distribution.
    :param eps: float, ``default=0.0001``. The regularization parameter of the Wasserstein distance.

    :return:
        - labTarget - np.array of shape (K,n_obs_target), where K is the number of different type of cell populations in the source data set. The coefficient labTarget[k,j] corresponds to the probability that the observation xTarget[j] belongs to the class k.
        - clustarget - np.array of shape (n_obs_target,). This array stores the optimal transport based classification of the target data set.
    """
    J = Y.shape[0]
    N_cl = labSource.shape[0]

    # Computation of the c-transform on the target distribution support.
    fCeY = np.zeros(J)
    for j in range(J):
        fCeY[j] = cTransform(f, X, Y, j, beta, eps)

    print('Computation of ctransform done.')

    labTarget = np.zeros((N_cl, J))

    for j in range(J):
        costY = cost(X, Y[j])
        arg = (f + fCeY[j] - costY) / eps
        pCol = np.exp(arg)
        labTarget[:, j] = labSource.dot(pCol)

    clustarget = np.argmax(labTarget, axis=0) + 1
    return [labTarget, clustarget]
