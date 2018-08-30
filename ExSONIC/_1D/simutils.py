# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 11:29:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 20:42:48

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.utils import si_format, pow10_format
from PySONIC.plt import plotSignals

from .sonic1D import Sonic1D


def compareEStim(neuron, nnodes, diam, L, Ra, connector, Astim, tstim, toffset, PRF, DC,
                 dt=None, atol=None, cmode='qual'):
    ''' Compare results of E-STIM simulations of the 1D extended SONIC model with different sections
        connection schemes.
    '''

    # Set amplitude distribution vector
    amps = np.insert(np.zeros(nnodes - 1), 0, Astim)

    # Run model simulation with classic connect scheme
    tstart = time.time()
    model = Sonic1D(neuron, nsec=nnodes, diam=diam, L=L, Ra=Ra)
    model.setElecAmps(amps)
    t_classic, stimon_classic, _, Vmeffprobes_classic, *_ = model.simulate(
        tstim, toffset, PRF, DC, dt, atol)
    tcomp_classic = time.time() - tstart
    stimon_classic[0] = 1

    # Run model simulation with custom connect scheme
    tstart = time.time()
    model = Sonic1D(neuron, nsec=nnodes, diam=diam, L=L, Ra=Ra, connector=connector)
    model.setElecAmps(amps)
    t_custom, stimon_custom, _, Vmeffprobes_custom, *_ = model.simulate(
        tstim, toffset, PRF, DC, dt, atol)
    tcomp_custom = time.time() - tstart
    stimon_custom[0] = 1

    # Create comparative figure
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} neuron, {} node{}, d = {}m, L = {}m, Ra = ${}$ ohm.cm - {}'
                 .format(neuron.name, nnodes, 's' if nnodes > 1 else '',
                         *si_format([diam, L], space=' '), pow10_format(Ra),
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)
    tonset = -10.0
    fs = 10
    lbls = ['node {} ({:.0f} $mA/m^2$)'.format(i + 1, amps[i]) for i in range(nnodes)]

    # Plot Vmeff-traces for classic scheme
    ax = axes[0]
    plotSignals(t_classic, Vmeffprobes_classic, states=stimon_classic, ax=ax,
                onset=(tonset, neuron.Vm0), lbls=lbls, fs=fs, cmode=cmode)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('classic v-based connect scheme', fontsize=12)

    # Plot V-traces for custom scheme
    ax = axes[1]
    plotSignals(t_custom, Vmeffprobes_custom, states=stimon_custom, ax=ax,
                onset=(tonset, neuron.Vm0), lbls=lbls, fs=fs, cmode=cmode)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('custom {}-based connect scheme'.format(connector.vref, fontsize=12))

    # Plot comparative time histogram
    ax = axes[2]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    ax.set_ylabel('comp. time (s)', fontsize=fs)
    indices = [1, 2]
    tcomps = [tcomp_classic, tcomp_custom]
    ax.set_xticks(indices)
    ax.set_xticklabels(['classic', 'custom'])
    for idx, tcomp in zip(indices, tcomps):
        ax.bar(idx, tcomp, align='center')
        ax.text(idx, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
                horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    return fig


def runPlotAStim(neuron, nnodes, diam, L, Ra, connector, a, Fdrive, Adrive, tstim, toffset, PRF, DC,
                 dt=None, atol=None, cmode='qual'):
    ''' Run A-STIM simulation of 1D extended SONIC model and plot results. '''

    # Create extended SONIC model with specific US frequency and connection scheme
    tstart = time.time()
    model = Sonic1D(neuron, nnodes, diam, L, Ra, connector, a, Fdrive)

    # Set node-specifc acoustic amplitudes
    amps = np.insert(np.zeros(nnodes - 1), 0, Adrive)
    model.setUSAmps(amps * 1e-3)

    # Run model simulation
    t, stimon, Qprobes, Vmeffprobes, _ = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp = time.time() - tstart

    # Plot membrane potential and charge profiles
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 4, 1])
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} neuron, {} node{}, d = {}m, L = {}m, Ra = ${}$ ohm.cm - {}'
                 .format(neuron.name, nnodes, 's' if nnodes > 1 else '',
                         *si_format([diam, L], space=' '), pow10_format(Ra),
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)
    for ax in axes:
        ax.set_ylim(-150, 50)
    tonset = -10.0
    Vm0 = model.neuron.Vm0
    fs = 10
    lbls = ['node {} ({:.0f} kPa)'.format(i + 1, amps[i] * 1e-3) for i in range(nnodes)]

    # Plot charge density profiles
    ax = axes[0]
    plotSignals(t, Qprobes, states=stimon, ax=ax, onset=(tonset, Vm0),
                lbls=lbls, fs=fs, cmode=cmode)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Qm (nC/cm2)', fontsize=fs)
    ax.set_title('membrane charge density', fontsize=fs + 2)

    # Plot effective potential profiles
    ax = axes[1]
    plotSignals(t, Vmeffprobes, states=stimon, ax=ax, onset=(tonset, Vm0),
                lbls=lbls, fs=fs, cmode=cmode)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
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
