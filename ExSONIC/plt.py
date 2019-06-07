# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-26 17:11:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-07 13:58:45


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.plt import SchemePlot


def plotSignals(t, signals, states=None, ax=None, onset=None, lbls=None, fs=10, cmode='qual',
                linestyle='-', cmap='winter'):
    ''' Plot several signals on one graph.

        :param t: time vector
        :param signals: list of signal vectors
        :param states (optional): stimulation state vector
        :param ax (optional): handle to figure axis
        :param onset (optional): onset to add to signals on graph
        :param lbls (optional): list of legend labels
        :param fs (optional): font size to use on graph
        :param cmode: color mode ('seq' for sequentiual or 'qual' for qualitative)
        :param linestyle: linestyle string ('-', '--', 'o--' or '.--')
        :return: handles to created lines
    '''

    # Set axis aspect
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)

    # Compute number of signals
    nsignals = len(signals)

    # Adapt labels for sequential color mode
    if cmode == 'seq' and lbls is not None:
        lbls[1:-1] = ['.'] * (nsignals - 2)

    # Add stimulation patches if states provided
    if states is not None:
        tpatch_on, tpatch_off = SchemePlot.getStimPulses(_, t, states)
        for i in range(tpatch_on.size):
            ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                       facecolor='#8A8A8A', alpha=0.2)

    # Add onset of provided
    if onset is not None:
        t0, y0 = onset
        t = np.hstack((np.array([t0, 0.]), t))
        signals = np.hstack((np.ones((nsignals, 2)) * y0, signals))

    # Determine colorset
    nlevels = nsignals
    if cmode == 'seq':
        norm = matplotlib.colors.Normalize(0, nlevels - 1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm._A = []
        colors = [sm.to_rgba(i) for i in range(nlevels)]
    elif cmode == 'qual':
        nlevels_max = 10
        if nlevels > nlevels_max:
            raise Warning('Number of signals higher than number of color levels')
        colors = ['C{}'.format(i) for i in range(nlevels)]
    else:
        raise ValueError('Unknown color mode')

    # Plot signals
    handles = []
    for i, var in enumerate(signals):
        lh = ax.plot(t, var, linestyle, label=lbls[i] if lbls is not None else None, c=colors[i])[0]
        handles.append(lh)

    # Add legend
    if lbls is not None:
        ax.legend(fontsize=fs, frameon=False)

    return handles
