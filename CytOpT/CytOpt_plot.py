# Copyright (C) 2022, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Bland_Altman
def Bland_Altman(proportions, Class=None):
    """ Function to display a bland plot in order to visually assess the agreement between CytOpt estimation
    of the class proportions and the estimate of the class proportions provided through manual gating.

    :param proportions: proportions ``data.frame`` of true and proportion estimates from ``CytOpt()``
    :param Class: Population classes
    """
    if Class is None:
        proportions.insert(0, "Populations", proportions.index.values, True)
    else:
        proportions.insert(0, "Populations", Class, True)

    plotData = pd.melt(proportions, id_vars=["Gold_standard", "Populations"],
                       var_name='Method', value_name='Estimate')

    plotData['Diff'] = plotData['Gold_standard'].ravel() - plotData['Estimate'].ravel()
    plotData['Mean'] = (plotData['Gold_standard'].ravel() + plotData['Estimate'].ravel()) / 2
    plotData['Method'] = plotData['Method'].replace(["minmax", "desasc"],
                                                    ["MinMax swapping", "Descent-Ascent"])

    sd_diff = np.std(plotData['Diff'])
    uniqueValues = set(plotData['Method'])

    BA = sns.relplot(
        data=plotData, x="Mean", y="Diff",
        col="Method", hue="Populations",
        kind="scatter", palette="deep"
    )

    fig = BA.fig
    fig.subplots_adjust(top=.85)
    fig.suptitle("Bland-Altman concordance plot", size=16)
    fig.supylabel(r'$(p_i - \hat{p}_i)$', size=16)
    fig.supxlabel(r'$(p_i + \hat{p}_i)/2$', size=16)

    for idx, item in enumerate(uniqueValues):
        ptlData = plotData[plotData['Method'] == item]
        fig.axes[idx].set_title(item, fontweight="bold")
        fig.axes[idx].axhline(sd_diff + (1.96 * np.std(ptlData['Diff'])), xmin=0,
                              linestyle='dashed', label='+1.96 SD')
        fig.axes[idx].text(max(plotData['Mean']), sd_diff + (1.96 * np.std(ptlData['Diff'])),
                           '+1.96 SD', fontsize=10)

        fig.axes[idx].axhline(np.mean(np.std(ptlData['Diff'])) - (1.96 * np.std(ptlData['Diff'])),
                              xmin=0, linestyle='dashed', label='-1.96 SD')
        fig.axes[idx].text(max(plotData['Mean']), np.mean(np.std(ptlData['Diff'])) - (1.96 * np.std(ptlData['Diff'])),
                           '+1.96 SD', fontsize=10)

        fig.axes[idx].axhline(np.mean(np.std(ptlData['Diff'])), xmin=0, label='Mean')
        fig.axes[idx].get_xaxis().set_visible(False)
        fig.axes[idx].get_yaxis().set_visible(False)

    BA.legend.remove()
    fig.legend(fontsize=10, markerscale=2, loc=7)
    plt.show()


def bar_plot(proportions, Class=None, title='CytOpt estimation and Manual estimation'):
    """ Function to display a bland plot in order to visually assess the agreement between CytOpt estimation
    of the class proportions and the estimate of the class proportions provided through manual gating.

    :param proportions: proportions ``data.frame`` of true and proportion estimates from ``CytOpt()`` and
    :param Class: Population classes
    :param title: plot title. Default is ``CytOpt estimation and Manual estimation``, i.e. no title.
    """
    if Class is None:
        proportions["Populations"] = proportions.index.values
    else:
        proportions["Populations"] = Class

    plotData = pd.melt(proportions, id_vars="Populations",
                       var_name='Method', value_name='Proportion')

    plotData['Method'] = plotData['Method'].replace(["minmax", "desasc"],
                                                    ["MinMax swapping", "Descent-Ascent"])
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Populations', y='Proportion', hue='Method', data=plotData,
                palette=['darkgreen', 'lime', 'lightcoral'])
    plt.legend(loc='upper left', fontsize=14)
    plt.xlabel('Population', size=14)
    plt.ylabel('Proportion', size=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontweight="bold", loc='left', size=16)
    plt.show()


def KL_plot(monitoring, n_0=10, n_stop=10000, title="Kullback-Liebler divergence trace"):
    """ Function to display a bland plot in order to visually assess the agreement between CytOpt estimation
    of the class proportions and the estimate of the class proportions provided through manual gating.

    :param monitoring: list of monitoring estimates from ``CytOpt()`` output.
    :param n_0: first iteration to plot. Default is ``10``.
    :param n_stop: last iteration to plot. Default is ``1000``.
    :param title: plot title. Default is ``Kullback-Liebler divergence trace``.
    :return:
    """
    n_0 = int(n_0)
    n_stop = int(n_stop)

    index = np.arange(n_0, n_stop)
    Monitoring = monitoring.loc[index, :]

    plotData = {'index': np.tile(index, len(Monitoring.columns)),
                'values': list(pd.concat([Monitoring[col] for col in Monitoring])),
                'Method': np.repeat(list(Monitoring.columns), len(index), axis=0)}

    plotData = pd.DataFrame(data=plotData)

    plotData['Method'] = plotData['Method'].replace(["minmax", "desasc"],
                                                    ["MinMax swapping", "Descent-Ascent"])
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=plotData, x="index", y="values", hue="Method")
    plt.legend(loc='best', fontsize=16)
    plt.xlabel('Iteration', size=20)
    plt.ylabel(r'KL$(\hat{p}|p)$', size=20)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.title(title, fontweight="bold", loc='left', size=16)
    plt.show()


def result_plot(results, Class=None, n_0=10, n_stop=1000):
    """ Fonction permettant d'afficher un graphique afin d'évaluer visuellement la concordance entre
    l'estimation par CytOpt des proportions de classe et l'estimation des proportions de classe fournie
    par le biais de la sélection manuelle et d'évaluer visuellement la concordance entre l'estimation par
    CytOpt du suivi et l'estimation du suivi fournie par le biais de la sélection manuelle.

    :param results: a list of ``data.frame`` of true and proportion estimates from ``CytOpt()`` and ``dataframe ``of monitoring estimates from ``CytOpt()`` output.
    :param Class: Population classes
    :param n_0: first iteration to plot. Default is ``10``.
    :param n_stop: last iteration to plot. Default is ``1000``.
    """
    for item, value in results.items():
        if item == 'proportions':
            Proportion = results['proportions']
            bar_plot(Proportion, Class=Class, title="")
        elif item == 'monitoring':
            monitoring = results['monitoring']
            KL_plot(monitoring, n_0=n_0, n_stop=n_stop)
        else:
            warnings.warn("WARNING: Not items in [proportions,monitoring]")
