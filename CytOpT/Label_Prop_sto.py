# Copyright (C) 2021, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy
from CytOpT.CytOpt_Descent_Ascent import cost, grad_h


def c_transform(f, X_s, X_t=None, j=None, beta=None, eps=0):
    """     Calculate the c_transform of f in the non regularized case if eps=0.
    Otherwise, it computes the smooth c_transform with respect to the usual entropy.

    :param f:
    :param X_s:
    :param X_t:
    :param j:
    :param beta:
    :param eps:
    :return:
    """
    if eps == 0:
        cost_y = cost(X_s, X_t[j])
        return np.min(cost_y - f)

    else:
        arg = (f - cost(X_s, X_t[j])) / eps
        return eps * (np.log(beta[j]) - logsumexp(arg))


def h_function(f, X_s, X_t=None, j=None, beta=None, eps=0, alpha=0):
    """     Calculate the function h whose expectation equals the semi-dual loss.
    Maximizing the semi-dual loss allows us to compute the wasserstein distance.

    :param f:
    :param X_s:
    :param X_t:
    :param j:
    :param beta:
    :param eps:
    :param alpha:
    :return:
    """
    if eps == 0:
        return np.sum(f * alpha) + c_transform(f, X_s, X_t=X_t[j])

    elif entropy == 'standard':
        return np.sum(f * alpha) + c_transform(f, X_s, X_t=X_t, j=j, beta=beta, eps=eps) - eps


def Robbins_Wass(X_s, X_t, alpha, beta, eps=0, const=0.1, n_iter=10000):
    """ Function that calculates the approximation of the optimal dual vector associated
    to the source distribution. The regularized optimal-transport problem is computed between a distribution with
    support X_s and weights alpha, and a distribution with support X_t and weights beta. This function solves the
    semi-dual formulation of the regularized OT problem with the stochastic algorithm of Robbins-Monro.

    :param X_s: np.array of shape (n_obs_source, dimension). Support of the source distribution.
    :param X_t: np.array of shape (n_obs_target, dimension). Support of the target distribution
    :param alpha: np.array of shape (n_obs_source,). Weights of the source distribution.
    :param beta: np.array of shape (n_obs_target,). Weights of the target  distribution.
    :param eps: float, ``default=0.0001``. Regularization parameter of the Wasserstein distance. This parameter
        should be greater than 0.
    :param const: float, ``default=0.1``. Constant involved in the Robbins-Monro algorithm when the regularization parameter
        eps=0.
    :param n_iter: int, ``default=10000``. Number of iterations of the Robbins-Monro algorithm.

    :return:
        - f - np.array of shape (n_obs_source,). Optimal kantorovich potential associated to the source distribution.
    """
    n_iter = int(n_iter)
    I = X_s.shape[0]
    J = X_t.shape[0]

    alpha = alpha.ravel()

    # Step size policy.
    if eps == 0:
        gamma = 0.01 / (4 * np.min(alpha))
    else:
        gamma = eps / (1.9 * min(alpha))
    c = 0.51
    # Sampling with respect to the distribution beta.
    sample = np.random.choice(a=np.arange(J), size=n_iter, p=beta.ravel())

    # Initialization of the dual vector f.
    f = np.zeros(I)

    # f_I vector:useful for the unregularized case.
    F_I = 1 / np.sqrt(I) * np.ones(I)

    for k in range(n_iter):

        # One sample of the beta distributions.
        y = X_t[sample[k], :]

        # Computation of the gradient if eps>0 or subgradient if eps=0.
        if eps == 0:
            grad_temp = grad_h(f, X_s, y, alpha)
            grad = (grad_temp - const * np.sum(f * F_I) * F_I)
        else:
            grad = grad_h(f, X_s, y, alpha, eps=eps)

        # Update of the dual variable
        f = f + gamma / ((k + 1) ** c) * grad

    return f


def Label_Prop_sto(L_source, f, X, Y, alpha, beta, eps):
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
