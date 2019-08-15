# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 17:11:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-15 21:43:16


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.plt import GroupedTimeSeries


def loadData(fpath):
    ''' Load dataframe and metadata dictionary from pickle file. '''
    logger.info('Loading data from "%s"', os.path.basename(fpath))
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        data = frame['data']
        meta = frame['meta']
        return data, meta


def getData(entry, trange=None):
    if entry is None:
        raise ValueError('non-existing data')
    if isinstance(entry, str):
        data, meta = loadData(entry)
    else:
        data, meta = entry
    if trange is not None:
        tmin, tmax = trange
        data = {k: df.loc[(df['t'] >= tmin) & (df['t'] <= tmax)] for k, df in data.items()}
    return data, meta


def comparativePlot(signals, labels, pltvars):

    naxes = len(pltvars)

    # Create figure
    fig = plt.figure(figsize=(12, 2 * naxes))
    wratios = [4, 4, 0.2, 1] if cmode == 'seq' else [4, 4, 1]
    gs = gridspec.GridSpec(1, len(wratios), width_ratios=wratios, wspace=0.3)
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} - {}'.format(
        model.pprint(),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
    ), fontsize=13)

    # for ax in axes[:2]:
    #     ax.set_ylim(-150, 50)
    tonset = -0.05 * (t[-1] - t[0])
    Vm0 = model.neuron.Vm0

    # Plot charge density profiles
    ax = axes[0]
    ax.set_title('membrane charge density', fontsize=fs)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm Qm\ (nC/cm^2)$', fontsize=fs)
    plotSignals(t, Qprobes, states=stimon, ax=ax, onset=(tonset, Vm0 * neuron.Cm0 * 1e2),
                fs=fs, cmode=cmode, cmap=cmap)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)

    # Plot effective potential profiles
    if cmode == 'seq':
        lbls = None
    ax = axes[1]
    ax.set_title('effective membrane potential', fontsize=fs)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm V_{m,eff}\ (mV)$', fontsize=fs)
    plotSignals(t, Vmeffprobes, states=stimon, ax=ax, onset=(tonset, Vm0),
                fs=fs, cmode=cmode, cmap=cmap, lbls=lbls)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)

    # Plot node index reference
    if cmode == 'seq':
        cbar_ax = axes[2]
        bounds = np.arange(nnodes + 1) + 1
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        mpl.colorbar.ColorbarBase(
            cbar_ax,
            cmap=cmap,
            norm=norm,
            spacing='proportional',
            ticks=bounds[:-1] + 0.5,
            ticklocation='left',
            boundaries=bounds,
            format='%1i'
        )
        cbar_ax.tick_params(axis='both', which='both', length=0)
        cbar_ax.set_title('node index', size=fs)
        iax = 3
    else:
        iax = 2

    # Plot computation time
    ax = axes[iax]
    ax.set_title('comp. time (s)', fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_xticks([])
    ax.bar(1, tcomp, align='center', color='dimgrey')
    ax.text(1, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
            horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    return fig

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
        tpatch_on, tpatch_off = GroupedTimeSeries.getStimPulses(_, t, states)
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
