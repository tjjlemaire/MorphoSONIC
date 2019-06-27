# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 16:41:08
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-27 15:10:48

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.utils import si_format, pow10_format
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.plt import GroupedTimeSeries

from .node import Node, SonicNode


def compare(neuron, A, tstim, toffset, PRF=100., DC=1., a=None, Fdrive=None, dt=None, atol=None,
            verbose=False):
    ''' Compare results of NEURON and Python based A-STIM or E-STIM simulations of the point-neuron
        SONIC model.
    '''
    comp_keys = ['Python', 'NEURON']
    modality = {True: 'Iinj', False: 'US'}[a is None]

    # Create comparative figure
    fs = 10
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
    axes = list(map(plt.subplot, gs))
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for item in ax.get_xticklabels() + ax.get_xticklabels():
            item.set_fontsize(fs)
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    Aunit = 'A/m2' if modality == 'Iinj' else 'Pa'
    Afactor = 1e-3 if modality == 'Iinj' else 1e3
    for ax in axes[:2]:
        ax.set_xlabel('time (ms)', fontsize=fs)
    ax = axes[0]
    ax.set_ylim(neuron.Qbounds() * 1e5)
    ax.set_ylabel('Qm (nC/cm2)', fontsize=fs)
    ax.set_title('membrane charge density', fontsize=fs + 2)
    ax = axes[1]
    ax.set_ylim(-150, 70)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('effective membrane potential', fontsize=fs + 2)
    ax = axes[2]
    ax.set_ylabel('comp. time (s)', fontsize=fs)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(comp_keys)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e2)

    data, tcomp = {}, {}

    # Run NEURON simulation
    key = 'NEURON'
    if modality == 'US':
        nrn_model = SonicNode(neuron, a=a, Fdrive=Fdrive, verbose=verbose)
        nrn_model.setUSdrive(A)
    elif modality == 'Iinj':
        nrn_model = Node(neuron)
        nrn_model.setIinj(A)
    data[key], tcomp[key] = nrn_model.simulate(tstim, toffset, PRF, DC, dt, atol)

    # Run Python stimulation
    key = 'Python'
    args = [tstim, toffset, PRF, DC]
    if modality == 'US':
        py_model = NeuronalBilayerSonophore(a * 1e-9, neuron)
        args = [Fdrive * 1e3, A * 1e3] + args
    elif modality == 'Iinj':
        py_model = neuron
        args = [A] + args
    data[key], tcomp[key] = py_model.simulate(*args)

    # Get pulses timing
    tpatch_on, tpatch_off = GroupedTimeSeries.getStimPulses(
        data['Python']['t'].values, data['Python']['stimstate'].values)

    # Plot charge density and membrane potential profiles
    tonset = -0.05 * (np.ptp(data['Python']['t']))
    for k in comp_keys:
        tplt = np.hstack((np.array([tonset, 0.]), data[k]['t'].values)) * 1e3
        axes[0].plot(tplt, np.hstack((np.ones(2) * neuron.Qm0, data[k]['Qm'])) * 1e5, label=k)
        axes[1].plot(tplt, np.hstack((np.ones(2) * neuron.Vm0, data[k]['Vm'])), label=k)

    # Plot stim patches on both graphs
    for ax in axes[:2]:
        ax.legend(fontsize=fs, frameon=False)
        ax.set_xlim(tonset * 1e3, (tstim + toffset) * 1e3)
        for ton, toff in zip(tpatch_on, tpatch_off):
            ax.axvspan(ton * 1e3, toff * 1e3, edgecolor='none', facecolor='#8A8A8A', alpha=0.2)

    # Plot comparative time histogram
    ax = axes[2]
    for i, k in enumerate(comp_keys):
        tc = tcomp[k]
        idx = i + 1
        ax.bar(idx, tc, align='center')
        ax.text(idx, 1.5 * tc, '{}s'.format(si_format(tc, 2, space=' ')),
                horizontalalignment='center')

    # Add figure title
    fig.suptitle('{}, A = {}{}, {}s'.format(
        nrn_model.strBiophysics(), si_format(A * Afactor, space=' '), Aunit, si_format(tstim, space=' '),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))),
        fontsize=18)

    return fig
