# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-27 16:41:08
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-15 03:08:07

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.utils import si_format, pow10_format
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.plt import getStimPulses

from .sonic0D import Sonic0D


def compare(neuron, A, tstim, toffset, PRF=100., DC=1., a=None, Fdrive=None, dt=None, atol=None,
            verbose=False):
    ''' Compare results of NEURON and Python based A-STIM or E-STIM simulations of the point-neuron
        SONIC model.
    '''

    # Create NEURON point-neuron SONIC model and run simulation
    tstart = time.time()
    model = Sonic0D(neuron, a=a, Fdrive=Fdrive, verbose=verbose)
    if model.modality == 'US':
        model.setUSdrive(A)
    elif model.modality == 'Iinj':
        model.setIinj(A)
    t_NEURON, y_NEURON, stimon_NEURON = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    Qm_NEURON, Vmeff_NEURON = y_NEURON[0:2, :]
    tcomp_NEURON = time.time() - tstart

    # Run Python stimulation
    tstart = time.time()
    if model.modality == 'US':
        nbls = NeuronalBilayerSonophore(a * 1e-9, neuron)
        t_Python, y_Python, stimon_Python = nbls.simulate(Fdrive * 1e3, A * 1e3,
                                                          tstim, toffset, PRF, DC)
        Qm_Python, Vmeff_Python = y_Python[2:4, :]
    elif model.modality == 'Iinj':
        t_Python, y_Python, stimon_Python = neuron.simulate(A, tstim, toffset, PRF, DC)
        Vmeff_Python = y_Python[0, :]  # mV
        Qm_Python = Vmeff_Python * neuron.Cm0 * 1e-3  # C/m2
    tcomp_Python = time.time() - tstart


    # Rescale vectors to appropriate units
    t_Python, t_NEURON = [t * 1e3 for t in [t_Python, t_NEURON]]
    Qm_Python, Qm_NEURON = [Qm * 1e5 for Qm in [Qm_Python, Qm_NEURON]]

    # Get pulses timing
    npatches, tpatch_on, tpatch_off = getStimPulses(t_Python, stimon_Python)

    # Add onset to signals
    tonset = -0.05 * (t_Python[-1] - t_NEURON[0])
    t_Python, t_NEURON = [np.hstack((np.array([tonset, 0.]), t)) for t in [t_Python, t_NEURON]]
    Qm_Python, Qm_NEURON = [np.hstack((np.ones(2) * neuron.Vm0 * neuron.Cm0 * 1e2, Qm))
                            for Qm in [Qm_Python, Qm_NEURON]]
    Vmeff_Python, Vmeff_NEURON = [np.hstack((np.ones(2) * neuron.Vm0, Vm))
                                  for Vm in [Vmeff_Python, Vmeff_NEURON]]

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
    Aunit = 'A/m2' if model.modality == 'Iinj' else 'Pa'
    Afactor = 1e-3 if model.modality == 'Iinj' else 1e3
    fig.suptitle('{}, A = {}{}, {}s'.format(
        model.strBiophysics(), si_format(A * Afactor, space=' '), Aunit, si_format(tstim, space=' '),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))),
        fontsize=18)

    # Plot charge density profiles
    ax = axes[0]
    ax.plot(t_Python, Qm_Python, label='Python')
    ax.plot(t_NEURON, Qm_NEURON, label='NEURON')
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(neuron.Qbounds() * 1e5)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Qm (nC/cm2)', fontsize=fs)
    ax.set_title('membrane charge density', fontsize=fs + 2)

    # Plot effective membrane potential profiles
    ax = axes[1]
    ax.plot(t_Python, Vmeff_Python, label='Python')
    ax.plot(t_NEURON, Vmeff_NEURON, label='NEURON')
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-150, 70)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('effective membrane potential', fontsize=fs + 2)

    # Plot comparative time histogram
    ax = axes[2]
    ax.set_ylabel('comp. time (s)', fontsize=fs)
    indices = [1, 2]
    tcomps = [tcomp_Python, tcomp_NEURON]
    ax.set_xticks(indices)
    ax.set_xticklabels(['Python', 'NEURON'])
    for idx, tcomp in zip(indices, tcomps):
        ax.bar(idx, tcomp, align='center')
        ax.text(idx, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
                horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    return fig
