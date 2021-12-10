# Copyright (C) 2021, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)

import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy


def cost(X_s, y):
    """
    Squared euclidean distance between y and the I points of X_s.
    """
    diff = X_s - y
    return (np.linalg.norm(diff, axis=1) ** 2)


def c_transform(f, X_s, X_t=None, j=None, beta=None, eps=0):
    """
    Calculate the c_transform of f in the non regularized case if eps=0.
    Otherwise, it computes the smooth c_transform with respect to the usual entropy.
    """
    if eps == 0:
        cost_y = cost(X_s, X_t[j])
        return (np.min(cost_y - f))

    else:
        arg = (f - cost(X_s, X_t[j])) / eps
        return eps * (np.log(beta[j]) - logsumexp(arg))


def grad_h(f, X_s, y, alpha, eps=0):
    """
    This function calculates the gradient of the function that we aim to maximize.
    The expectation of this function computed at a maximizer equals the wasserstein disctance,
    or its regularized counterpart.
    """
    if eps == 0:
        cost_y = cost(X_s, y)
        i_star = np.argmin(cost_y - f)
        to_return = alpha.copy()
        to_return[i_star] = alpha[i_star] - 1
        return (to_return)

    else:
        arg = (f - cost(X_s, y)) / eps
        Mx = np.max(arg)
        pi = np.exp(arg - Mx)
        pi = pi / pi.sum()
        return alpha - pi


def h_function(f, X_s, X_t=None, j=None, beta=None, eps=0, alpha=0):
    """
    Calculate the function h whose expectation equals the semi-dual loss.
    Maximizing the semi-dual loss allows us to compute the wasserstein distance.
    """
    if eps == 0:
        return np.sum(f * alpha) + c_transform(f, X_s, X_t=X_t[j])

    elif entropy == 'standard':
        return np.sum(f * alpha) + c_transform(f, X_s, X_t=X_t, j=j, beta=beta, eps=eps) - eps


def Robbins_Wass(X_s, X_t, alpha, beta, eps=0, const=0.1, n_iter=10000):
    """
    Function that calculates the approximation of the Wasserstein distance between mu and nu
    thanks to the Robbins-Monro Algorithm. X and Y are the supports of the source and target
    distribution. Alpha and beta are the weights of the distributions.
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
    """
    Function that calculates a classification on the target data
    thanks to the approximation of the transport plan and the classification of the source data.
    We got the approximation of the transport plan with the stochastic algorithm.
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
