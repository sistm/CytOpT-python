# Copyright (C) 2021, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>
#
# License: MIT (see COPYING file)

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# __all__ = ['CytOpT']

from CytOpT.descentAscent import cytoptDesasc
from CytOpT.minmaxSwapping import cytoptMinmax
from CytOpT.plots import resultPlot, BlandAltman


def stopRunning():
    warnings.warn("deprecated", DeprecationWarning)


def getLengthUniqueNumbers(values):
    list_of_unique_value = []
    unique_values = set(values)
    for number in unique_values:
        list_of_unique_value.append(number)
    return {'list_of_unique_value': list_of_unique_value, 'length': len(list_of_unique_value)}


# CytOpT
def CytOpT(xSource, xTarget, labSource, labTarget=None, thetaTrue=None,
           method=None, eps=1e-04, nIter=4000, power=0.99,
           stepGrad=10, step=5, lbd=1e-04, nItGrad=10000, nItSto=10,
           cont=True, monitoring=False, minMaxScaler=True, thresholding=True):
    """ CytOpT algorithm. This methods is designed to estimate the proportions of cells in an unclassified Cytometry
    data set denoted xTarget. CytOpT is a supervised method that levarge the classification denoted labSource associated
    to the flow cytometry data set xSource. The estimation relies on the resolution of an optimization problem.
    two procedures are provided "minmax" and "desasc". We recommend to use the default method that is
    ``minmax``.

    :param xSource: np.array of shape (n_samples_source, n_biomarkers). The source cytometry data set.
        A cytometry dataframe. The columns correspond to the different biological markers tracked.
        One line corresponds to the cytometry measurements performed on one cell. The classification
        of this Cytometry data set must be provided with the labSource parameters.
    :param xTarget: np.array of shape (n_samples_target, n_biomarkers). The target cytometry data set.
        A cytometry dataframe. The columns correspond to the different biological markers tracked.
        One line corresponds to the cytometry measurements performed on one cell. The CytOpT algorithm
        targets the cell type proportion in this Cytometry data set
    :param labSource: np.array of shape (n_samples_source,). The classification of the source data set.
    :param labTarget: np.array of shape (n_samples_target,), ``default=None``. The classification of the target data set.
    :param thetaTrue: np.array of shape (K,), ``default=None``. This array stores the true proportions of the K type of
        cells estimated in the target data set. This parameter is required if the user enables the monitoring option.
    :param method: {"minmax", "desasc", "both"}, ``default="minmax"``. Method chosen to
        to solve the optimization problem involved in CytOpT. It is advised to rely on the default choice that is
        "minmax".
    :param eps: float, ``default=0.0001``. Regularization parameter of the Wasserstein distance. This parameter must be
        positive.
    :param nIter: int, ``default=10000``. Number of iterations of the stochastic gradient ascent for the Minmax swapping
        optimization method.
    :param power: float, ``default=0.99``. Decreasing rate for the step-size policy of the stochastic gradient ascent
        for the Minmax swapping optimization method. The step-size decreases at a rate of 1/n^power.
    :param stepGrad: float, ``default=10``. Constant step_size policy for the gradient descent of the descent-ascent
        optimization strategy.
    :param step: float, ``default=5``. Multiplication factor of the stochastic gradient ascent step-size policy for
        the minmax optimization method.
    :param lbd: float, ``default=0.0001``. Additionnal regularization parameter of the Minmax swapping optimization method.
        This parameter lbd should be greater or equal to eps.
    :param nItGrad: int, ``default=10000``. Number of iterations of the outer loop of the descent-ascent optimization method.
        This loop corresponds to the descent part of descent-ascent strategy.
    :param nItSto: int, ``default = 10``. Number of iterations of the inner loop of the descent-ascent optimization method.
        This loop corresponds to the stochastic ascent part of this optimization procedure.
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

        - KL_monitoring: np.array of shape (n_out, ) or (nIter,) depending on the choice of the optimization method. This array stores the evolution of the Kullback-Leibler divergence between the estimate and benchmark proportions, if monitoring==True.

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

    if thetaTrue is None:
        if labTarget is None and labSource is None:
            with warnings.catch_warnings():
                warnings.simplefilter("labTarget and theta can not be null at the same time\n"
                                      "Initialize at least one of the two parameters")
                stopRunning()
        elif labTarget is not None:
            labTargetInfo = getLengthUniqueNumbers(labTarget)
            thetaTrue = np.zeros(labTargetInfo['length'])
            for index in range(labTargetInfo['length']):
                thetaTrue[index] = sum(labTarget == index + 1) / len(labTarget)
        else:
            labSourceInfo = getLengthUniqueNumbers(labSource)
            thetaTrue = np.zeros(labSourceInfo['length'])
            for index in range(labSourceInfo['length']):
                thetaTrue[index] = sum(labSource == index + 1) / len(labSource)

    if xSource is None or xTarget is None:
        with warnings.catch_warnings():
            warnings.simplefilter("xSource and xTarget can not be null\n"
                                  "Initialize at two parameters")
            stopRunning()
    else:
        xSource = np.asarray(xSource)
        xTarget = np.asarray(xTarget)

    if thresholding:
        xSource = xSource * (xSource > 0)
        xTarget = xTarget * (xTarget > 0)

    if minMaxScaler:
        Scaler = MinMaxScaler()
        xSource = Scaler.fit_transform(xSource)
        xTarget = Scaler.fit_transform(xTarget)

    h_res = {}
    monitoring_res = {}

    h_res["GoldStandard"] = thetaTrue
    if method in ["minmax", "both"]:
        results = cytoptMinmax(xSource, xTarget, labSource,
                               eps=eps, lbd=lbd, nIter=nIter,
                               step=step, cont=cont, power=power, thetaTrue=thetaTrue,
                               monitoring=monitoring)
        h_res['minmax'] = results[0]
        if monitoring:
            monitoring_res["minmax"] = results[1][:min(nIter, nItGrad)]

    if method in ["desasc", "both"]:
        results = cytoptDesasc(xSource, xTarget, labSource,
                               eps=eps, nItGrad=nItGrad, nItSto=nItSto,
                               stepGrad=stepGrad, cont=cont, thetaTrue=thetaTrue,
                               monitoring=monitoring)
        h_res['desasc'] = results[0]
        if monitoring:
            monitoring_res["desasc"] = results[1][:min(nIter, nItGrad)]

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

    xSource = np.asarray(Stanford1A_values)
    xTarget = np.asarray(Stanford3A_values)
    labSource = np.asarray(Stanford1A_clust['x'])
    labTarget = np.asarray(Stanford3A_clust['x'])
    thetaTrue = np.zeros(10)
    for k in range(10):
        thetaTrue[k] = np.sum(labTarget == k + 1) / len(labTarget)

    nItGrad = 10000
    nIter = 10000
    nItSto = 10
    pas_grad = 10
    eps = 0.0005
    monitoring = True
    results = CytOpT(xSource, xTarget, labSource, thetaTrue=thetaTrue,
                     method="both", nItGrad=nItGrad, nItSto=nItSto, stepGrad=pas_grad, eps=eps, nIter=nIter,
                     monitoring=monitoring)

    resultPlot(results, n0=10, nStop=8000)
    BlandAltman(results['proportions'])
