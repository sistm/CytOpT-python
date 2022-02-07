# Copyright (C) 2022, Kalidou BA, Paul Freulon <paul.freulon@math.u-bordeaux.fr>=
#
# License: MIT (see COPYING file)
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Bland_Altman
def BlandAltman(proportions, Class=None, Center=None):
    """ Function to display a bland plot in order to visually assess the agreement between CytOpt estimation
    of the class proportions and the estimate of the class proportions provided through manual gating.

    :param proportions: proportions ``data.frame`` of true and proportion estimates from ``CytOpt()``
    :param Class: Population classes
    :param Center: Center of class population
    """
    if Class is None:
        proportions.insert(0, "Populations", proportions.index.values, True)
    else:
        proportions.insert(0, "Populations", Class, True)

    plotData = pd.melt(proportions, id_vars=["GoldStandard", "Populations"],
                       var_name='Method', value_name='Estimate')

    plotData['Diff'] = plotData['GoldStandard'].ravel() - plotData['Estimate'].ravel()
    plotData['Mean'] = (plotData['GoldStandard'].ravel() + plotData['Estimate'].ravel()) / 2
    plotData['Method'] = plotData['Method'].replace(["minmax", "desasc"],
                                                    ["MinMax swapping", "Descent-Ascent"])
    if Center is not None:
        if len(Center) == plotData.shape[0]:
            plotData['Center'] = Center

    sd_diff = np.std(plotData['Diff'])
    uniqueValues = set(plotData['Method'])
    BA = sns.relplot(
        data=plotData, x="Mean", y="Diff",
        col="Method", hue="Populations",
        palette="dark", style=Center)
    fig = BA.fig
    fig.suptitle("Bland-Altman concordance plot", size=16)
    fig.supylabel(r'$(p_i - \hat{p}_i)$', size=16)
    fig.supxlabel(r'$(p_i + \hat{p}_i)/2$', size=16)
    labelLines = ['+1.96 SD', '-1.96 SD', 'Mean']
    noLabelLines = np.repeat('_legend_', 3)
    labels = noLabelLines
    for idx, item in enumerate(uniqueValues):
        if idx == len(uniqueValues) - 1:
            labels = labelLines
        pltData = plotData[plotData['Method'] == item]
        fig.axes[idx].set_title(item, fontweight="bold")
        fig.axes[idx].axhline(np.mean(pltData['Diff']) + (1.96 * np.std(pltData['Diff'])), xmin=0,
                              linestyle='dashed', label=labels[0])
        fig.axes[idx].text(max(plotData['Mean']), sd_diff + (1.96 * np.std(pltData['Diff'])),
                           '+1.96 SD', fontsize=10)

        fig.axes[idx].axhline(np.mean(pltData['Diff']) - (1.96 * np.std(pltData['Diff'])),
                              xmin=0, linestyle='dashed', label=labels[1])
        fig.axes[idx].text(max(plotData['Mean']), np.mean(np.std(pltData['Diff'])) - (1.96 * np.std(pltData['Diff'])),
                           '-1.96 SD', fontsize=10)

        fig.axes[idx].axhline(np.mean(pltData['Diff']), xmin=0, label=labels[2])
        fig.axes[idx].set_xlabel('')
        fig.axes[idx].set_ylabel('')

    BA.legend.remove()
    plt.tight_layout()
    fig.legend(markerscale=1, loc=7)
    plt.show()
    proportions.drop("Populations", axis=1, inplace=True)


def barPlot(proportions, Class=None, title='CytOpt estimation and Manual estimation'):
    """ Function to display a bland plot in order to visually assess the agreement between CytOpt estimation
    of the class proportions and the estimate of the class proportions provided through manual gating.

    :param proportions: proportions ``data.frame`` of true and proportion estimates from ``CytOpt()`` and
    :param Class: Population classes
    :param title: plot title. Default is ``CytOpt estimation and Manual estimation``, i.e. no title.
    """
    if Class is None:
        proportions.insert(0, "Populations", proportions.index.values, True)
    else:
        proportions.insert(0, "Populations", Class, True)

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
    proportions.drop("Populations", axis=1, inplace=True)


def KLPlot(monitoring, n0=10, nStop=10000, title="Kullback-Liebler divergence trace"):
    """ Function to display a bland plot in order to visually assess the agreement between CytOpt estimation
    of the class proportions and the estimate of the class proportions provided through manual gating.

    :param monitoring: list of monitoring estimates from ``CytOpt()`` output.
    :param n0: first iteration to plot. Default is ``10``.
    :param nStop: last iteration to plot. Default is ``1000``.
    :param title: plot title. Default is ``Kullback-Liebler divergence trace``.
    :return:
    """
    n0 = int(n0)
    nStop = int(nStop)

    index = np.arange(n0, nStop)
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


def resultPlot(results, Class=None, n0=10, nStop=1000):
    """ Function to display a graph to visually assess the agreement between the CytOpt estimate of class proportions;
    the estimate of class proportions provided by manual selection and to visually assess the agreement between the
    CytOpt estimate of follow-up and the estimate of follow-up provided by manual selection.

    :param results: a list of ``data.frame`` of true and proportion estimates from ``CytOpt()`` and ``dataframe ``of monitoring estimates from ``CytOpt()`` output.
    :param Class: Population classes
    :param n0: first iteration to plot. Default is ``10``.
    :param nStop: last iteration to plot. Default is ``1000``.
    """
    resultsPlot = results
    for item, value in resultsPlot.items():
        if item == 'proportions':
            Proportion = resultsPlot[item]
            barPlot(Proportion, Class=Class)
        elif item == 'monitoring':
            monitoring = resultsPlot[item]
            KLPlot(monitoring, n0=n0, nStop=nStop)
        else:
            warnings.warn("WARNING: Not items in [proportions,monitoring]")


if __name__ == '__main__':
    # CytOpt estimation
    Estimate_Prop = pd.read_csv('../tests/data/Res_Estimation_Stan1A.txt',
                                index_col=0)
    # Benchmark estimation
    True_Prop = pd.read_csv('../tests/data/True_proportion_Stan1A.txt',
                            index_col=0)
    True_Prop = True_Prop.drop(['Baylor1A'])
    Estimate_Prop = Estimate_Prop.drop(['Baylor1A'])
    Estimate_Prop = np.asarray(Estimate_Prop)
    True_Prop = np.asarray(True_Prop)
    Classes = np.tile(np.arange(1, 11), 61)
    Centre_1 = np.repeat(['Yale', 'UCLA', 'NHLBI', 'CIMR', 'Miami'], 10)
    Centre_2 = np.repeat(['Standford', 'Yale', 'UCLA', 'NHLBI', 'CIMR', 'Baylor', 'Miami'], 10)
    Centre = np.hstack((Centre_1, Centre_2, Centre_2, Centre_2,
                        Centre_2, Centre_2, Centre_2, Centre_2, Centre_2))

    props = pd.DataFrame({'GoldStandard': True_Prop.ravel(), 'minmax': Estimate_Prop.ravel()})

    BlandAltman(props, Class=Classes, Center=Centre)
