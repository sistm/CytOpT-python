# Copyright (C) 2021, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>
#
# License: MIT (see COPYING file)

import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# __all__ = ['CytOpt']

from CytOpT import cytopt_minmax, cytopt_desasc
from CytOpT.cytopt_plot import plot_py_1, plot_py_prop1, Bland_Altman, plot_py_Comp, plot_py_prop2, Bland_Altman_Comp


def stop_running():
    warnings.warn("deprecated", DeprecationWarning)


def get_length_unique_numbers(values):
    list_of_unique_value = []
    unique_values = set(values)
    for number in unique_values:
        list_of_unique_value.append(number)
    return {'list_of_unique_value': list_of_unique_value, 'length': len(list_of_unique_value)}


# CytOpt
def CytOpt(X_s, X_t, Lab_source, Lab_target=None, cell_type=None, names_pop=None,
           method=None, theta_true=None, eps=1e-04, n_iter=10000, power=0.99,
           step_grad=10, step=5, lbd=1e-04, n_out=10000, n_stoc=10, n_0=10,
           n_stop=10000, monitoring=False, minMaxScaler=True, thresholding=True):
    """
    CytOpT algorithm. This methods is designed to estimate the proportions of cells in an unclassified Cytometry
    data set denoted X_t. CytOpT is a supervised method that levarge the classification denoted Lab_source associated
    to the flow cytometry data set X_s. The estimation relies on the resolution of an optimization problem.
    two procedures are provided "cytopt_minmax" and "cytopt_desasc". We recommend to use the default method that is
    "cytopt_minmax".

    :param X_s: np.array of shape (n_samples_source, n_biomarkers). The source cytometry data set.

    :param X_t: np.array of shape (n_samples_target, n_biomarkers). The target cytometry data set.

    :param Lab_source: np.array of shape (n_samples_source,). The classification of the source data set.

    :param Lab_target: np.array of shape (n_samples_target,), default=None. The classification of the target data set.

    :param cell_type: list, default=None. Contains the names of the different cell types estimated.

    :param names_pop:

    :param method: {"cytopt_minmax", "cytopt_desasc", "comparison_opt"}, default="cytopt_minmax". Method chosen to
    to solve the optimization problem involved in CytOpT. It is advised to rely on the default choice that is
    "cytopt_minmax".

    :param theta_true: np.array of shape (K,), default=None. This array stores the true proportions of the K type of
     cells estimated in the target data set. This parameter is required if the user enables the monitoring option.

    :param eps: float, default=0.0001. Regularization parameter of the Wasserstein distance. This parameter must be
    positive.

    :param n_iter: int, default=10000. Number of iterations of the stochastic gradient ascent for the Minmax swapping
    optimization method.

    :param power: float, default=0.99. Decreasing rate for the step-size policy of the stochastic gradient ascent
    for the Minmax swapping optimization method. The step-size decreases at a rate of 1/n^power.

    :param step_grad: float, default=10. Constant step_size policy for the gradient descent of the descent-ascent
    optimization strategy.

    :param step: float, default=5. Multiplication factor of the stochastic gradient ascent step-size policy for
    the minmax optimization method.

    :param lbd: float, default=0.0001. Additionnal regularization parameter of the Minmax swapping optimization method.
    This parameter lbd should be greater or equal to eps.

    :param n_out: int, default=10000. Number of iterations of the outer loop of the descent-ascent optimization method.
    This loop corresponds to the descent part of descent-ascent strategy.

    :param n_stoc: int, default = 10. Number of iterations of the inner loop of the descent-ascent optimization method.
    This loop corresponds to the stochastic ascent part of this optimization procedure.

    :param n_0: int, default = 1, First iteration to display the evolution of the Kullback-Leibler divergence along
    the steps of the optimization strategy chosen. The monitoring variable should be equal to monitoring=True.

    :param n_stop: int, default=1000. Last iteration to display the evolution of the Kullback-Leibler divergence along
    the steps of the optimization strategy chosen. The monitoring variable should be equal to monitoring=True.

    :param monitoring: bool, default=False. When set to true, the evolution of the Kullback-Leibler between the
    estimated proportions and the benchmark proportions is tracked and stored.

    :param minMaxScaler: bool, default = True. When set to True, the source and target data sets are scaled in [0,1]^d,
    where d is the  number of biomarkers monitored.

    :param thresholding: bool, default = True. When set to True, all the coefficients of the source and target data sets
    are replaced by their positive part. This preprocessing is relevant for Cytometry Data as the signal acquisition of
    the cytometer can induce convtrived negative values.

    :return:
    hat_theta : np.array of shape (K,), where K is the number of different type of cell populations in the source
    data set.
    KL_monitoring: np.array of shape (n_out, ) or (n_iter,) depending on the choice of the optimization method. This
    array stores the evolution of the Kullback-Leibler divergence between the estimate and benchmark proportions, if
    monitoring==True.


    Reference:
     Paul Freulon, Jérémie Bigot,and Boris P. Hejblum CytOpT: Optimal Transport with Domain Adaptation for Interpreting Flow Cytometry data,
     arXiv:2006.09003 [stat.AP].
    """

    if theta_true is None:
        theta_true = []
    if method is None:
        method = ["cytopt_minmax", "cytopt_desasc", "comparison_opt"]
    if isinstance(method, list):
        method = method[0]

    if method not in ["cytopt_minmax", "cytopt_desasc", "comparison_opt"]:
        warnings.warn('"choose method in list : \"cytopt_minmax or","cytopt_desasc or", "comparison_opt\""')
        method = "cytopt_minmax"
    labSourceInfo = get_length_unique_numbers(Lab_source)

    if names_pop is not None and len(names_pop) >= 2:
        h_source = np.zeros(2)
        for k in range(labSourceInfo['length']):
            h_source[k] = np.sum(Lab_source == k) / len(Lab_source)
        names_pop = names_pop[0:2]
        plot_py_1(X_s, X_t, Lab_source, 100 * h_source, names_pop)

    if theta_true is None:
        if Lab_target is None:
            with warnings.catch_warnings():
                warnings.simplefilter("Lab_target and theta can not be null at the same time\n"
                                      "Initialize at least one of the two parameters")
                stop_running()
        else:
            h_source = np.zeros(labSourceInfo['length'])
            theta_true = np.zeros(labSourceInfo['length'])
            for index in range(labSourceInfo['length']):
                h_source[index] = sum(Lab_source == index + 1) / len(Lab_source)
                theta_true[index] = sum(Lab_target == index + 1) / len(Lab_target)

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

    if cell_type is None:
        warnings.warn("WARNING: cell_type is null, We create cell_type value")
        cell_type = np.tile(labSourceInfo['list_of_unique_value'], 2)

    if method == "cytopt_minmax":
        res = cytopt_minmax(X_s, X_t, Lab_source, eps=eps, lbd=lbd, n_iter=n_iter,
                            step=step, power=power, theta_true=theta_true, monitoring=monitoring)
        h_hat = res[0]

        # Display of the estimation Results
        percentage = np.concatenate((theta_true, h_hat), axis=None)
        method = np.repeat(['Manual Benchmark', 'CytOpt Minmax'], labSourceInfo['length'])
        Res_df = pd.DataFrame({'Percentage': percentage, 'Cell_Type': cell_type, 'Method': method})
        print(Res_df)
        plot_py_prop1(Res_df)

        Diff_prop = theta_true.ravel() - h_hat.ravel()
        sd_diff = np.std(Diff_prop)
        print('Standard deviation:', sd_diff)

        Mean_prop = (theta_true.ravel() + h_hat.ravel()) / 2

        sd_diff = np.std(Diff_prop)
        print('Standard deviation:', sd_diff)

        print('Percentage of classes where the estimation error is below 10% with CytOpT Minmax')
        print(np.sum(abs(Diff_prop) < 0.1) / len(Diff_prop) * 100)
        print('Percentage of classes where the estimation error is below 5% with CytOpT Minmax')
        print(np.sum(abs(Diff_prop) < 0.05) / len(Diff_prop) * 100)
        Dico_res = pd.DataFrame({'h_hat': h_hat.ravel(),
                                 'True_Prop': theta_true.ravel(),
                                 'Diff': Diff_prop,
                                 'Mean': Mean_prop,
                                 'Classe': np.tile(labSourceInfo['list_of_unique_value'],
                                                   int(Mean_prop.shape[0] / labSourceInfo['length']))})
        Dico_res['Classe'] = Dico_res['Classe'].astype('object')

        Bland_Altman(Dico_res, sd_diff,
                     labSourceInfo['length'], title='CytOpT Minmax')

        res = {'Minmax_hat': h_hat, 'Res_df': Res_df, 'Dico_res': Dico_res}

    elif method == "cytopt_desasc":
        res = cytopt_desasc(X_s, X_t, Lab_source, theta_true=theta_true, eps=eps, n_out=n_out,
                            n_stoc=n_stoc, step_grad=step_grad)
        h_hat = res[0]

        # Display of the estimation Results
        percentage = np.concatenate((theta_true, h_hat), axis=None)
        method = np.repeat(['Manual Benchmark', 'CytOpT Desac'], labSourceInfo['length'])

        Res_df = pd.DataFrame({'Percentage': percentage, 'Cell_Type': cell_type, 'Method': method})
        plot_py_prop1(Res_df)

        Diff_prop = theta_true.ravel() - h_hat.ravel()
        sd_diff = np.std(Diff_prop)
        print('Standard deviation:', sd_diff)

        Mean_prop = (theta_true.ravel() + h_hat.ravel()) / 2

        sd_diff = np.std(Diff_prop)
        print('Standard deviation:', sd_diff)

        print('Percentage of classes where the estimation error is below 10% with CytOpT Desasc')
        print(np.sum(abs(Diff_prop) < 0.1) / len(Diff_prop) * 100)
        print('Percentage of classes where the estimation error is below 5% with CytOpT Desasc')
        print(np.sum(abs(Diff_prop) < 0.05) / len(Diff_prop) * 100)

        Dico_res = pd.DataFrame({'h_hat': h_hat.ravel(),
                                 'True_Prop': theta_true.ravel(),
                                 'Diff': Diff_prop,
                                 'Mean': Mean_prop,
                                 'Classe': np.tile(labSourceInfo['list_of_unique_value'],
                                                   int(Mean_prop.shape[0] / labSourceInfo['length']))})
        Dico_res['Classe'] = Dico_res['Classe'].astype('object')

        Bland_Altman(Dico_res, sd_diff,
                     labSourceInfo['length'], title='CytOpT Desasc')

        res = {'Desac_hat': h_hat, 'Res_df': Res_df, 'Dico_res': Dico_res}

    else:
        t0 = time.time()
        res_desasc = cytopt_desasc(X_s, X_t, Lab_source=Lab_source, eps=eps, n_out=n_out,
                                   n_stoc=n_stoc, step_grad=step_grad, theta_true=theta_true)

        elapsed_time = time.time() - t0
        print("Time running execution Desac ->", elapsed_time, 's\n')
        Desac_hat = res_desasc[0]
        Desasc_monitoring = res_desasc[1]

        # Display of the estimation Results
        percentage = np.concatenate((theta_true, Desac_hat), axis=None)
        method = np.repeat(['Manual Benchmark', 'CytOpT Desasc'], labSourceInfo['length'])

        Res_df = pd.DataFrame({'Percentage': percentage, 'Cell_Type': cell_type, 'Method': method})
        plot_py_prop1(Res_df)

        # Minmax method
        t0 = time.time()
        Results_Minmax = cytopt_minmax(X_s, X_t, Lab_source, eps=eps, lbd=lbd, n_iter=n_iter,
                                       theta_true=theta_true, step=step_grad, power=power, monitoring=True)
        elapsed_time = time.time() - t0
        print('Elapsed time : ', elapsed_time / 60, 'Mins')

        Minmax_hat = Results_Minmax[0]
        Minmax_monitoring = Results_Minmax[1]

        # Display of the estimation Results
        percentage = np.concatenate((theta_true, Minmax_hat), axis=None)
        method = np.repeat(['Manual Benchmark', 'CytOpT Minmax'], labSourceInfo['length'])
        Res_df = pd.DataFrame({'Percentage': percentage, 'Cell_Type': cell_type, 'Method': method})
        plot_py_prop1(Res_df)

        plot_py_Comp(n_0, n_stop, Minmax_monitoring, Desasc_monitoring)
        Proportion = np.concatenate((Desac_hat, Minmax_hat, theta_true), axis=None)
        Classes = np.tile(labSourceInfo['list_of_unique_value'], 3)
        Methode = np.repeat(['CytOpt_DesAsc', 'CytOpt_Minmax', 'Manual'], labSourceInfo['length'])
        df_res1 = pd.DataFrame({'Proportions': Proportion, 'Classes': Classes, 'Methode': Methode})
        plot_py_prop2(df_res1)

        Diff_prop_Desasc = theta_true.ravel() - Desac_hat.ravel()
        Mean_prop_Desasc = (theta_true.ravel() + Desac_hat.ravel()) / 2

        print('Percentage of classes where the estimation error is below 10% with CytOpT desac')
        print(np.sum(abs(Diff_prop_Desasc) < 0.1) / len(Diff_prop_Desasc) * 100)
        print('Percentage of classes where the estimation error is below 5% with CytOpT desac')
        print(np.sum(abs(Diff_prop_Desasc) < 0.05) / len(Diff_prop_Desasc) * 100)

        Diff_prop_Minmax = theta_true.ravel() - Minmax_hat.ravel()
        Mean_prop_Minmax = (theta_true.ravel() + Minmax_hat.ravel()) / 2

        print('Percentage of classes where the estimation error is below 10% with CytOpT Minmax')
        print(np.sum(abs(Diff_prop_Minmax) < 0.1) / len(Diff_prop_Minmax) * 100)
        print('Percentage of classes where the estimation error is below 5% with CytOpT Minmax')
        print(np.sum(abs(Diff_prop_Minmax) < 0.05) / len(Diff_prop_Minmax) * 100)
        sd_diff = [np.std(Diff_prop_Desasc), np.std(Diff_prop_Minmax)]
        print('Standard deviation:', sd_diff)
        ClassesDesac = np.repeat(labSourceInfo['list_of_unique_value'], Desac_hat.shape[0])
        n_pal = get_length_unique_numbers(ClassesDesac)['length']

        # Concat two graphs
        Dico_resDesac = pd.DataFrame({'Desac_hat': Desac_hat.ravel(),
                                      'True_Prop': theta_true,
                                      'Diff': Diff_prop_Desasc,
                                      'Mean': Mean_prop_Desasc,
                                      'Class': np.tile(labSourceInfo['list_of_unique_value'],
                                                       int(Mean_prop_Desasc.shape[0] / labSourceInfo['length']))})

        Dico_resMinmax = pd.DataFrame({'Desac_hat': Minmax_hat.ravel(),
                                       'True_Prop': theta_true,
                                       'Diff': Diff_prop_Minmax,
                                       'Mean': Mean_prop_Minmax,
                                       'Class': np.tile(labSourceInfo['list_of_unique_value'],
                                                        int(Mean_prop_Minmax.shape[0] / labSourceInfo['length']))})
        Bland_Altman_Comp(Dico_resDesac, Dico_resMinmax, sd_diff, n_pal)

        res = {"Desac_h_hat": Desac_hat, "Desasc_monitoring": Desasc_monitoring,
               "Minmax_h_hat": Minmax_hat, "Minmax_monitoring": Minmax_monitoring}
    return res


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

    X_source = np.asarray(Stanford1A_values)
    X_target = np.asarray(Stanford3A_values)
    Lab_source = np.asarray(Stanford1A_clust['x'])
    Lab_target = np.asarray(Stanford3A_clust['x'])
    theta_true = np.zeros(10)
    for k in range(10):
        theta_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)

    CytOpt(X_source, X_target, Lab_source, Lab_target=None, cell_type=None,
           method="comparison_opt", theta_true=theta_true, eps=1e-04, n_iter=4000, power=0.99,
           step_grad=50, step=5, lbd=1e-04, n_out=1000, n_stoc=10, n_0=10,
           n_stop=1000, monitoring=False, minMaxScaler=True)
