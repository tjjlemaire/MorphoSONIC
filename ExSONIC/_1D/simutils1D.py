# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 11:29:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-17 17:26:24

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from PySONIC.utils import si_format, pow10_format
from ..plt import plotSignals
from ..utils import getSpikesTimings, getConductionSpeeds
from .sonic1D import Sonic1D, computeVext


def compareEStim(neuron, rs, connector, nodeD, nodeL, interD, interL, Iinj, tstim, toffset,
                 PRF, DC, nnodes=None, dt=None, atol=None, cmode='qual', verbose=False, z0=None):
    ''' Compare results of E-STIM simulations of the 1D extended SONIC model with different sections
        connection schemes.
    '''

    vconds_classic, vconds_custom = np.nan, np.nan

    # Run model simulation with classic connect scheme
    model = Sonic1D(neuron, rs, nodeD, nodeL, interD=interD, interL=interL, nnodes=nnodes,
                    connector=None, verbose=verbose)

    # If electrode z-coordinate is provided, compute extracellular potentials (mV) from Iinj (uA).
    # Otherwise, Iinj is interpreted as intracellular injections (mA/m2)
    if z0 is None:
        if Iinj is None:
            Iinj = model.titrateIinjIntra(tstim, toffset, PRF, DC, dt, atol)
        lbls = model.setIinj(Iinj)
    else:
        if Iinj is None:
            Iinj = model.titrateIinjExtra(z0, tstim, toffset, PRF, DC, dt, atol)
        Vexts = computeVext(model, Iinj, z0)
        lbls = model.setVext(Vexts)

    tstart = time.time()
    t_classic, stimon_classic, _, Vmeffprobes_classic, *_ = model.simulate(
        tstim, toffset, PRF, DC, dt, atol)
    tcomp_classic = time.time() - tstart
    stimon_classic[0] = 1

    xcoords = model.getNodeCoordinates()  # um

    spiketimings_classic = getSpikesTimings(t_classic, Vmeffprobes_classic)  # ms
    if spiketimings_classic is not None:
        vconds_classic = getConductionSpeeds(xcoords, spiketimings_classic)  # m/s
        print('conduction speed range: {:.2f} - {:.2f} m/s'.format(
            vconds_classic.min(), vconds_classic.max()))

    # Run model simulation with custom connect scheme
    model = Sonic1D(neuron, rs, nodeD, nodeL, interD=interD, interL=interL, nnodes=nnodes,
                    connector=connector, verbose=verbose)
    if z0 is None:
        lbls = model.setIinj(Iinj)
    else:
        lbls = model.setVext(Vexts)
    tstart = time.time()
    t_custom, stimon_custom, _, Vmeffprobes_custom, *_ = model.simulate(
        tstim, toffset, PRF, DC, dt, atol)
    tcomp_custom = time.time() - tstart
    stimon_custom[0] = 1

    spiketimings_custom = getSpikesTimings(t_custom, Vmeffprobes_custom)  # ms
    if spiketimings_custom is not None:
        vconds_custom = getConductionSpeeds(xcoords, spiketimings_custom)  # m/s
        print('conduction speed range: {:.2f} - {:.2f} m/s'.format(
            vconds_custom.min(), vconds_custom.max()))

    # Rescale vectors to appropriate units
    t_classic, t_custom = [t * 1e3 for t in [t_classic, t_custom]]  # ms

    # Create comparative figure
    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[4, 0.2, 1, 1], wspace=0.3)
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} - {}'.format(
        model.pprint(),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
    ), fontsize=13)
    tonset = -0.05 * (t_classic[-1] - t_classic[0])
    fs = 10
    cmap = plt.cm.jet_r

    # Plot Vmeff-traces for classic and custom schemes
    ax = axes[0]
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_m$ (mV)', fontsize=fs)
    handles_classic = plotSignals(
        t_classic, Vmeffprobes_classic, states=stimon_classic, ax=ax, onset=(tonset, neuron.Vm0),
        fs=fs, linestyle='-', cmode=cmode, cmap=cmap)
    handles_custom = plotSignals(
        t_custom, Vmeffprobes_custom, states=stimon_custom, ax=ax, onset=(tonset, neuron.Vm0),
        fs=fs, linestyle='--', cmode=cmode, cmap=cmap)
    ax.legend([handles_classic[-1], handles_custom[-1]], ['classic', 'custom'],
              fontsize=fs, frameon=False)

    cbar_ax = axes[1]
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

    colors = ['dimgrey', 'silver']

    # Histogram comparing conduction speeds
    ax = axes[2]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_title('conduction speed (m/s)', fontsize=fs)
    indices = [1, 2]
    vconds = [np.squeeze(np.reshape(x, (x.size, 1))) for x in [vconds_classic, vconds_custom]]
    ax.set_xticks(indices)
    ax.set_xticklabels(['classic', 'custom'])
    violin_parts = ax.violinplot(vconds, indices, points=60, widths=0.5, showextrema=False)
    for c, pc in zip(colors, violin_parts['bodies']):
        pc.set_facecolor(c)
        pc.set_edgecolor('k')
    for idx, data in zip(indices, vconds):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        wmin = np.clip(q1 - (q3 - q1) * 1.5, min(data), q1)
        wmax = np.clip(q3 + (q3 - q1) * 1.5, q3, max(data))
        ax.scatter(idx, med, marker='o', color='k', s=30, zorder=3)
        ax.vlines(idx, q1, q3, color='grey', linestyle='-', lw=5)
        ax.vlines(idx, wmin, wmax, color='grey', linestyle='-', lw=1)
        ax.text(idx, 0.5 * data.min(), '{:.1f} m/s'.format(med), horizontalalignment='center')
    # ax.plot(indices, vconds, '.k')
    ax.set_yscale('log')
    ax.set_ylim(1e0, 1e3)

    # Histogram comparing computation times
    ax = axes[3]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_title('comp. time (s)', fontsize=fs)
    indices = [1, 2]
    tcomps = [tcomp_classic, tcomp_custom]
    ax.set_xticks(indices)
    ax.set_xticklabels(['classic', 'custom'])
    for i, (idx, tcomp) in enumerate(zip(indices, tcomps)):
        ax.bar(idx, tcomp, align='center', color=colors[i])
        ax.text(idx, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
                horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    # fig.tight_layout()

    return fig


def runPlotAStim(neuron, a, Fdrive, rs, connector, nodeD, nodeL, interD, interL, amps, tstim, toffset,
                 PRF, DC, nnodes=None, dt=None, atol=None, cmode='qual', verbose=False):
    ''' Run A-STIM simulation of 1D extended SONIC model and plot results. '''

    tstart = time.time()

    # Create extended SONIC model with specific US frequency and connection scheme
    model = Sonic1D(neuron, rs, nodeD, nodeL, interD=interD, interL=interL, nnodes=nnodes,
                    a=a, Fdrive=Fdrive, connector=connector, verbose=verbose)

    # Set node-specifc acoustic amplitudes
    if isinstance(amps, float):
        amps = np.insert(np.zeros(nnodes - 1), 0, amps)
    lbls = model.setUSdrive(amps)  # kPa

    # Run model simulation
    t, stimon, Qprobes, Vmeffprobes, _ = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp = time.time() - tstart

    # Rescale vectors to appropriate units
    t *= 1e3  # ms
    Qprobes *= 1e5  # nC/cm2

    # Plot membrane potential and charge profiles
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 4, 1])
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} - {}'.format(
        model.pprint(),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
    ), fontsize=18)
    for ax in axes:
        ax.set_ylim(-150, 50)
    tonset = -10.0
    Vm0 = model.neuron.Vm0
    fs = 10

    # Plot charge density profiles
    ax = axes[0]
    plotSignals(t, Qprobes, states=stimon, ax=ax, onset=(tonset, Vm0),
                lbls=lbls, fs=fs, cmode=cmode)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm Qm\ (nC/cm^2)$', fontsize=fs)
    ax.set_title('membrane charge density', fontsize=fs + 2)

    # Plot effective potential profiles
    ax = axes[1]
    plotSignals(t, Vmeffprobes, states=stimon, ax=ax, onset=(tonset, Vm0),
                lbls=lbls, fs=fs, cmode=cmode)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm V_{m,eff}\ (mV)$', fontsize=fs)
    ax.set_title('effective membrane potential', fontsize=fs + 2)

    # Plot comparative time histogram
    ax = axes[2]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_ylabel('comp. time (s)', fontsize=fs)
    ax.set_xticks([])
    ax.bar(1, tcomp, align='center', color='dimgrey')
    ax.text(1, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
            horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    return fig
