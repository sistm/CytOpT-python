# Copyright (C) 2021, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>
#
# License: MIT (see COPYING file)

import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# __all__ = ['CytOpT']
from CytOpT.CytOpt_plot import result_plot
from CytOpT.CytOpt_Descent_Ascent import cytopt_desasc
from CytOpT.CytOpt_MinMax_Swapping import cytopt_minmax


def stop_running():
    warnings.warn("deprecated", DeprecationWarning)


def get_length_unique_numbers(values):
    list_of_unique_value = []
    unique_values = set(values)
    for number in unique_values:
        list_of_unique_value.append(number)
    return {'list_of_unique_value': list_of_unique_value, 'length': len(list_of_unique_value)}


# CytOpT
def CytOpT(X_s, X_t, Lab_source, Lab_target=None, theta_true=None,
           method=None, eps=1e-04, n_iter=4000, power=0.99,
           step_grad=10, step=5, lbd=1e-04, n_it_grad=10000, n_it_sto=10,
           cont=True, monitoring=False, minMaxScaler=True, thresholding=True):
    
    """ CytOpT algorithm. This methods is designed to estimate the proportions of cells in an unclassified Cytometry
    data set denoted X_t. CytOpT is a supervised method that levarge the classification denoted Lab_source associated
    to the flow cytometry data set X_s. The estimation relies on the resolution of an optimization problem.
    two procedures are provided "minmax" and "desasc". We recommend to use the default method that is
    ``minmax``.

    :param X_s: np.array of shape (n_samples_source, n_biomarkers). The source cytometry data set.
        A cytometry dataframe. The columns correspond to the different biological markers tracked.
        One line corresponds to the cytometry measurements performed on one cell. The classification
        of this Cytometry data set must be provided with the Lab_source parameters.
    :param X_t: np.array of shape (n_samples_target, n_biomarkers). The target cytometry data set.
        A cytometry dataframe. The columns correspond to the different biological markers tracked.
        One line corresponds to the cytometry measurements performed on one cell. The CytOpt algorithm
        targets the cell type proportion in this Cytometry data set
    :param Lab_source: np.array of shape (n_samples_source,). The classification of the source data set.
    :param Lab_target: np.array of shape (n_samples_target,), ``default=None``. The classification of the target data set.
    :param theta_true: np.array of shape (K,), ``default=None``. This array stores the true proportions of the K type of
        cells estimated in the target data set. This parameter is required if the user enables the monitoring option.
    :param method: {"minmax", "desasc", "both"}, ``default="minmax"``. Method chosen to
        to solve the optimization problem involved in CytOpT. It is advised to rely on the default choice that is
        "minmax".
    :param eps: float, ``default=0.0001``. Regularization parameter of the Wasserstein distance. This parameter must be
        positive.
    :param n_iter: int, ``default=10000``. Number of iterations of the stochastic gradient ascent for the Minmax swapping
        optimization method.
    :param power: float, ``default=0.99``. Decreasing rate for the step-size policy of the stochastic gradient ascent
        for the Minmax swapping optimization method. The step-size decreases at a rate of 1/n^power.
    :param step_grad: float, ``default=10``. Constant step_size policy for the gradient descent of the descent-ascent
        optimization strategy.
    :param step: float, ``default=5``. Multiplication factor of the stochastic gradient ascent step-size policy for
        the minmax optimization method.
    :param lbd: float, ``default=0.0001``. Additionnal regularization parameter of the Minmax swapping optimization method.
        This parameter lbd should be greater or equal to eps.
    :param n_it_grad: int, ``default=10000``. Number of iterations of the outer loop of the descent-ascent optimization method.
        This loop corresponds to the descent part of descent-ascent strategy.
    :param n_it_sto: int, ``default = 10``. Number of iterations of the inner loop of the descent-ascent optimization method.
        This loop corresponds to the stochastic ascent part of this optimization procedure.
    :param const: float, ``default=0.1``. Constant involved in the stochastic algorithm when the regularization parameter
        is set to eps=0.
    :param cont: bool, ``default=True``. When set to true, the progress is displayed.
    :param monitoring: bool, ``default=False``. When set to true, the evolution of the Kullback-Leibler between the
        estimated proportions and the benchmark proportions is tracked and stored.
    :param minMaxScaler: bool, ``default = True``. When set to True, the source and target data sets are scaled in [0,1]^d,
        where d is the  number of biomarkers monitored.
    :param thresholding: bool, ``default = True``. When set to True, all the coefficients of the source and target data sets
        are replaced by their positive part. This preprocessing is relevant for Cytometry Data as the signal acquisition of
        the cytometer can induce convtrived negative values.

    :return:
        - hat_theta : np.array of shape (K,), where K is the number of different type of cell populations in the source data set.
        
        - KL_monitoring: np.array of shape (n_out, ) or (n_iter,) depending on the choice of the optimization method. This array stores the evolution of the Kullback-Leibler divergence between the estimate and benchmark proportions, if monitoring==True.
        
    Reference:
     Paul Freulon, Jérémie Bigot,and Boris P. Hejblum CytOpT: Optimal Transport with Domain Adaptation for Interpreting Flow Cytometry data,
     arXiv:2006.09003 [stat.AP].
    """

    if method is None:
        method = ["minmax", "desasc", "both"]

    if isinstance(method, list):
        method = method[0]

    if method not in ["minmax", "desasc", "both"]:
        warnings.warn('"choose method in list : \"minmax or","desasc or", "both\""')
        method = "minmax"

    if theta_true is None:
        if Lab_target is None and Lab_source is None:
            with warnings.catch_warnings():
                warnings.simplefilter("Lab_target and theta can not be null at the same time\n"
                                      "Initialize at least one of the two parameters")
                stop_running()
        elif Lab_target is not None:
            labTargetInfo = get_length_unique_numbers(Lab_target)
            theta_true = np.zeros(labTargetInfo['length'])
            for index in range(labTargetInfo['length']):
                theta_true[index] = sum(Lab_target == index + 1) / len(Lab_target)
        else:
            labSourceInfo = get_length_unique_numbers(Lab_source)
            theta_true = np.zeros(labSourceInfo['length'])
            for index in range(labSourceInfo['length']):
                theta_true[index] = sum(Lab_source == index + 1) / len(Lab_source)

    if X_s is None or X_t is None:
        with warnings.catch_warnings():
            warnings.simplefilter("X_s and X_t can not be null\n"
                                  "Initialize at two parameters")
            stop_running()
    else:
        X_s = np.asarray(X_s)
        X_t = np.asarray(X_t)

    if thresholding:
        X_s = X_s * (X_s > 0)
        X_t = X_t * (X_t > 0)

    if minMaxScaler:
        Scaler = MinMaxScaler()
        X_s = Scaler.fit_transform(X_s)
        X_t = Scaler.fit_transform(X_t)

    h_res = {}
    monitoring_res = {}

    h_res["Gold_standard"] = theta_true
    if method in ["minmax", "both"]:
        t0 = time.time()
        results = cytopt_minmax(X_s, X_t, Lab_source,
                                eps=eps, lbd=lbd, n_iter=n_iter,
                                step=step, power=power, theta_true=theta_true,
                                monitoring=monitoring)
        elapsed_time = time.time() - t0
        print("Done (", elapsed_time, 's)\n')
        h_res['minmax'] = results[0]
        if monitoring:
            monitoring_res["minmax"] = results[1]

    if method in ["desasc", "both"]:
        t0 = time.time()
        results = cytopt_desasc(X_s, X_t, Lab_source,
                                eps=eps, n_it_grad=n_it_grad, n_it_sto=n_it_sto,
                                step_grad=step_grad, cont=cont, theta_true=theta_true,
                                monitoring=monitoring)
        elapsed_time = time.time() - t0
        print("Done (", elapsed_time, 's)\n')
        h_res['desasc'] = results[0]
        if monitoring:
            monitoring_res["desasc"] = results[1]

    if monitoring:
        return {"proportions": pd.DataFrame(h_res),
                "monitoring": pd.DataFrame(monitoring_res)}
    else:
        return {"proportions": pd.DataFrame(h_res)}


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
    theta_true = np.zeros(10)
    for k in range(10):
        theta_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

    n_it_grad = 1000
    n_it_sto = 10
    pas_grad = 10
    eps = 0.0005
    monitoring = False
    h_hat1 = CytOpT(X_source, X_target, Lab_source,
                    method="desasc", n_it_grad=n_it_grad, n_it_sto=n_it_sto, step_grad=pas_grad, eps=eps,
                    monitoring=monitoring)
    result_plot(h_hat1, n_0=10, n_stop=1000)
