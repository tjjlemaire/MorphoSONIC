# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-27 16:41:08
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-15 21:49:34

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.utils import si_format, pow10_format, getStimPulses
from PySONIC.core import NeuronalBilayerSonophore

from .sonic0D import Sonic0D


def runPlotEStim(neuron, Astim, tstim, toffset, PRF, DC, dt=None, atol=None, verbose=False):
    ''' Run E-STIM simulation of point-neuron SONIC model and plot results. '''

    # Create NEURON point-neuron SONIC model and run simulation
    model = Sonic0D(neuron, fs=0., verbose=verbose)
    model.setAstim(Astim)
    t, y, stimon = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    Vmeff = y[1, :]

    # Rescale vectors to appropriate units
    t *= 1e3

    # Get pulses timing
    npatches, tpatch_on, tpatch_off = getStimPulses(t, stimon)

    # Add onset to signals
    t0 = -10.0
    t = np.hstack((np.array([t0, 0.]), t))
    Vmeff = np.hstack((np.ones(2) * neuron.Vm0, Vmeff))

    # Create figure and plot effective membrane potential profile
    fs = 10
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)
    fig.subplots_adjust(top=0.8)
    fig.suptitle('{}, A = {}A/m2, {}s'.format(
        model.pprint(), *si_format([Astim * 1e-3, tstim], space=' '),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))),
        fontsize=18)
    ax.plot(t, Vmeff)
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-150, 70)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('effective membrane potential', fontsize=fs + 2)

    return fig


def compareAStim(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC, dt=None, atol=None,
                 verbose=False):
    ''' Compare results of NEURON and Python based A-STIM simulations of the point-neuron
        SONIC model.
    '''

    # Create NEURON point-neuron SONIC model and run simulation
    tstart = time.time()
    model = Sonic0D(neuron, a=a, Fdrive=Fdrive, verbose=verbose)
    model.setAdrive(Adrive)
    t_NEURON, y_NEURON, stimon_NEURON = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp_NEURON = time.time() - tstart
    Qm_NEURON, Vmeff_NEURON = y_NEURON[0:2, :]

    # Run Python stimulation
    tstart = time.time()
    nbls = NeuronalBilayerSonophore(a, neuron)
    t_Python, y_Python, stimon_Python = nbls.simulate(Fdrive, Adrive, tstim, toffset, PRF, DC)
    tcomp_Python = time.time() - tstart
    Qm_Python, Vmeff_Python = y_Python[2:4, :]

    # Rescale vectors to appropriate units
    t_Python, t_NEURON = [t * 1e3 for t in [t_Python, t_NEURON]]
    Qm_Python, Qm_NEURON = [Qm * 1e5 for Qm in [Qm_Python, Qm_NEURON]]

    # Get pulses timing
    npatches, tpatch_on, tpatch_off = getStimPulses(t_Python, stimon_Python)

    # Add onset to signals
    t0 = -10.0
    y0 = neuron.Vm0
    t_Python, t_NEURON = [np.hstack((np.array([t0, 0.]), t)) for t in [t_Python, t_NEURON]]
    Qm_Python, Qm_NEURON = [np.hstack((np.ones(2) * y0, Qm)) for Qm in [Qm_Python, Qm_NEURON]]
    Vmeff_Python, Vmeff_NEURON = [np.hstack((np.ones(2) * y0, Vm))
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
    fig.suptitle('{}, A = {}Pa, {}s'.format(
        model.pprint(), *si_format([Adrive, tstim], space=' '),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
    ), fontsize=18)

    # Plot charge density profiles
    ax = axes[0]
    ax.plot(t_Python, Qm_Python, label='Python')
    ax.plot(t_NEURON, Qm_NEURON, label='NEURON')
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
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
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
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



def runPlotAStim(neuron, a, fs, Fdrive, Adrive, tstim, toffset, PRF, DC, dt=None, atol=None,
                 verbose=None):
    ''' Run and plot results of NEURON A-STIM simulations of the point-neuron SONIC model. '''

    # Create NEURON point-neuron SONIC model and run simulation
    tstart = time.time()
    model = Sonic0D(neuron, a=a, fs=fs, Fdrive=Fdrive, verbose=verbose)
    model.setAdrive(Adrive)
    t, y, stimon = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    tcomp = time.time() - tstart
    Qm, Vmeff = y[0:2, :]

    # Rescale vectors to appropriate units
    t *= 1e3  # ms
    Qm *= 1e5  # nC/cm2

    # Get pulses timing
    npatches, tpatch_on, tpatch_off = getStimPulses(t, stimon)

    # Add onset to signals
    t0 = -10.0
    y0 = neuron.Vm0
    t = np.hstack((np.array([t0, 0.]), t))
    Qm = np.hstack((np.ones(2) * y0, Qm))
    Vmeff = np.hstack((np.ones(2) * y0, Vmeff))

    # Create figure
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
    fig.suptitle('{}, A = {}Pa, {}s'.format(
        model.pprint(), *si_format([Adrive, tstim], space=' '),
        'adaptive time step' if dt is None else 'dt = ${}$ ms'.format(pow10_format(dt * 1e3))
    ), fontsize=18)

    # Plot charge density profiles
    ax = axes[0]
    ax.plot(t, Qm)
    # ax.plot(t, Vmeff, label='$V_{meff}\ (mV)$')
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    # ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm Q_m\ (nC/cm^2)$', fontsize=fs)

    # Plot effective membrane potential profiles
    ax = axes[1]
    ax.plot(t, Vmeff)
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    # ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$\\rm V_{m,eff}\ (mV)$', fontsize=fs)

    # Plot comparative time histogram
    ax = axes[2]
    ax.set_ylabel('comp. time (s)', fontsize=fs)
    indices = [1]
    tcomps = [tcomp]
    ax.set_xticks(indices)
    ax.set_xticklabels(['Python', 'NEURON'])
    for idx, tcomp in zip(indices, tcomps):
        ax.bar(idx, tcomp, align='center')
        ax.text(idx, 1.5 * tcomp, '{}s'.format(si_format(tcomp, 2, space=' ')),
                horizontalalignment='center')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e2)

    return fig
