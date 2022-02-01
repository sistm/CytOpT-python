===========================================
 CyOpt
===========================================

Overview
========
`CytOpT` is a `python` package that provides a new algorithm relying regularized
optimal transport to directly estimate the different cell population proportions
from a biological sample characterized with flow cytometry measurements. Algorithm
is based on the regularized Wasserstein metric to compare cytometry measurements
from different samples, thus accounting for possible mis-alignment of a given cell
population across sample (due to technical variability from the technology of measurements).

The main function of the package is `CytOpT()`.

The methods implemented in this package are detailed in the following
article:
<https://arxiv.org/abs/2006.09003>`_. The ``source code`` of the package is available on  `github
<https://github.com/sistm/CytOpt-python>`_.

Getting started
===============
Install CytOpT
______________

Install the CytOpT package from pypi as follows::

    pip install -r requirements.txt

    pip install CytOpT # pip3 install CytOpT

Example
_______
Packages::

    import numpy as np
    import pandas as pd
    from CytOpT import CytOpt as cytopt

Preparing data::

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






Run CytOpT:
-----------
::

    # Initialization of parameters
    nItGrad = 5000
    nIter = 5000
    nItSto = 10
    pas_grad = 10
    eps = 0.0005
    monitoring = True

    # Run Minmax and Desasc
    res = cytopt.CytOpT(xSource, xTarget, labSource,thetaTrue=thetaTrue,
                        method="both", nItGrad=nItGrad, nIter=nIter, nItSto=nItSto,
                        stepGrad=pas_grad, eps=eps, monitoring=monitoring)

    # CytOpT Minmax with default params
    cytopt.CytOpT(xSource, xTarget, labSource, thetaTrue=thetaTrue, method='desasc')

    # CytOpT Desasc with default params
    cytopt.CytOpT(xSource, xTarget, labSource, thetaTrue=thetaTrue, method = 'minmax')

`KLPlot`:
    - Display a bland plot in order to visually assess the agreement between CytOpt estimation of the class proportions and the estimate of the class proportions provided through manual gating.

`barPlot`:
    - Display a bland plot in order to visually assess the agreement between CytOpt estimation of the class proportions and the estimate of the class proportions provided through manual gating.

::

    resultPlot(res, n0=10, nStop=4000)

