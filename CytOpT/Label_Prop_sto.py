# Copyright (C) 2021, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
from scipy.special import logsumexp


def cost(X_s, y):
    """ Squared euclidean distance between y and the I points of X_s.

    :param X_s: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param y:
    :return:
    """

    diff = X_s - y
    return np.linalg.norm(diff, axis=1) ** 2


def c_transform(f, X_s, X_t, j, beta, eps=0.1):
    """     Calculate the c_transform of f in the non regularized case if eps=0.
    Otherwise, it computes the smooth c_transform with respect to the usual entropy.

    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param X_s: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param X_t: np.array of shape (n_obs_target, dimension). Support of the target distribution
    :param j:
    :param beta: np.array of shape (n_obs_target,). Weights of the target  distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.

    :return:
    """
    if eps == 0:
        cost_y = cost(X_s, X_t[j])
        return np.min(cost_y - f)

    else:
        arg = (f - cost(X_s, X_t[j])) / eps
        return eps * (np.log(beta[j]) - logsumexp(arg))


def grad_h(f, X_s, y, alpha, eps=0.1):
    """ This function calculates the gradient of the function that we aim to maximize.
    The expectation of this function computed at a maximizer equals the wasserstein disctance,
    or its regularized counterpart.

    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param X_s: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param y:
    :param alpha: np.array of shape (n_obs_source,). Weights of the source distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.

    :return:
    """

    if eps == 0:
        cost_y = cost(X_s, y)
        i_star = np.argmin(cost_y - f)
        to_return = alpha.copy()
        to_return[i_star] = alpha[i_star] - 1
        return to_return

    else:
        arg = (f - cost(X_s, y)) / eps
        Mx = np.max(arg)
        pi = np.exp(arg - Mx)
        pi = pi / pi.sum()
        return alpha - pi


def h_function(f, X_s, X_t, j, alpha, beta, eps=0.1):
    """ Calculate the function h whose expectation equals the semi-dual loss.
    Maximizing the semi-dual loss allows us to compute the wasserstein distance.

    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param X_s: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param X_t: np.array of shape (n_obs_target, dimension). Support of the target distribution
    :param j:
    :param beta: np.array of shape (n_obs_target,). Weights of the target  distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.
    :param alpha: np.array of shape (n_obs_source,). Weights of the source distribution.

    :return:
    """
    if eps == 0:
        return np.sum(f * alpha) + c_transform(f, X_s, X_t[j])

    else:
        return np.sum(f * alpha) + c_transform(f, X_s, X_t, j, beta, eps) - eps


def Robbins_Wass(X_s, X_t, alpha, beta, eps=0.1, n_iter=10000):
    """ Function that calculates the approximation of the optimal dual vector associated
    to the source distribution. The regularized optimal-transport problem is computed between a distribution with
    support X_s and weights alpha, and a distribution with support X_t and weights beta. This function solves the
    semi-dual formulation of the regularized OT problem with the stochastic algorithm of Robbins-Monro.

    :param X_s: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param X_t: np.array of shape (n_obs_target, dimension). Support of the target distribution
    :param alpha: np.array of shape (n_obs_source,). Weights of the source distribution.
    :param beta: np.array of shape (n_obs_target,). Weights of the target  distribution.
    :param eps: float, ``default=0.1``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.
    :param n_iter: int, ``default=10000``. Number of iterations of the Robbins-Monro algorithm.

    :return:
        - f - np.array of shape (n_obs_source,). Optimal kantorovich potential associated to the source distribution.
    """
    n_iter = int(n_iter)
    I, J = X_s.shape[0], X_t.shape[0]

    # Definition of the step size policy
    gamma = eps / (1.9 * min(beta))
    c = 0.51

    # Storage of the estimates
    W_hat_storage = np.zeros(n_iter)
    Sigma_hat_storage = np.zeros(n_iter)
    h_eps_storage = np.zeros(n_iter)
    h_eps_square = np.zeros(n_iter)

    # Sampling according to the target distribution.
    sample = np.random.choice(a=np.arange(J), size=n_iter, p=beta)

    # Initialisation of the dual vector f.
    f = np.random.random(I)
    f = f - np.mean(f)

    # First iteration to start the loop.
    W_hat_storage[0] = h_function(f, X_s, X_t, sample[0], alpha, beta, eps)
    h_eps_storage[0] = h_function(f, X_s, X_t, sample[0], alpha, beta, eps)
    h_eps_square[0] = h_function(f, X_s, X_t, sample[0], alpha, beta, eps) ** 2

    # Robbins-Monro Algorithm.

    for k in range(1, n_iter):
        # Sample from the target measure.
        j = sample[k]

        # Update of f.
        f = f + gamma / ((k + 1) ** c) * grad_h(f, X_s, X_t[j, :], alpha, eps)
        h_eps_storage[k] = h_function(f, X_s, X_t, j, alpha, beta, eps)

        # Update of the estimator of the regularized Wasserstein distance.
        W_hat_storage[k] = k / (k + 1) * W_hat_storage[k - 1] + 1 / (k + 1) * h_eps_storage[k]

        # Update of the estimator of the asymptotic variance
        h_eps_square[k] = k / (k + 1) * h_eps_square[k - 1] + 1 / (k + 1) * h_eps_storage[k] ** 2
        Sigma_hat_storage[k] = h_eps_square[k] - W_hat_storage[k] ** 2

    return [f, W_hat_storage, Sigma_hat_storage]


def Label_Prop_sto(L_source, f, X, Y, alpha, beta, eps=0.0001):
    """     Function that calculates a classification of the target data with an optimal-transport based soft assignment.
    For optimal result, the source distribution must be re-weighted thanks to the estimation of the class proportions
    in the target data set.  This estimation can be produced with the Cytopt function. To compute an optimal dual
    vector f associated to the source distribution, we advocate the use of the Robbins_Wass function with a CytOpT
    re-weighting of the source distribution.

    :param L_source: np.array of shape (X.shape[0],). The labels associated to the source data set X_s
    :param f: np.array of shape (X.shape[0],). The optimal dual vector associated to the source distribution.
        Here, the Wasserstein distance is computed between the distribution with weights alpha and support X and the
        distribution with weights beta and support Y.
    :param X: np.array of shape (n_obs_source, dimension). The support of the source distribution.
    :param Y: np.array of shape (n_obs_target, dimension). The support of the target distribution.
    :param alpha: np.array of shape (n_obs_source,). The weights of the source distribution.
    :param beta: np.array of shape (n_obs_target,). The weights of the target distribution.
    :param eps: float, ``default=0.0001``. The regularization parameter of the Wasserstein distance.

    :return:
        - L_target - np.array of shape (K,n_obs_target), where K is the number of different type of cell populations in the source data set. The coefficient L_target[k,j] corresponds to the probability that the observation X_t[j] belongs to the class k.
        - clustarget - np.array of shape (n_obs_target,). This array stores the optimal transport based classification of the target data set.
    """
    print(alpha)
    J = Y.shape[0]
    N_cl = L_source.shape[0]

    # Computation of the c-transform on the target distribution support.
    f_ce_Y = np.zeros(J)
    for j in range(J):
        f_ce_Y[j] = c_transform(f, X, Y, j, beta, eps)

    print('Computation of ctransform done.')

    L_target = np.zeros((N_cl, J))

    for j in range(J):
        cost_y = cost(X, Y[j])
        arg = (f + f_ce_Y[j] - cost_y) / eps
        P_col = np.exp(arg)
        L_target[:, j] = L_source.dot(P_col)

    clustarget = np.argmax(L_target, axis=0) + 1
    return [L_target, clustarget]
