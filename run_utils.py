# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-21 14:01:30
# @Last Modified by:   Theo
# @Last Modified time: 2018-08-22 02:40:16

''' Utilities to run NEURON simulations and plot results. '''


import time
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, si_format, pow10_format
from PySONIC.plt import plotSignals, getPatchesLoc
from PySONIC.solvers import SolverUS
from ext_sonic import ExtendedSONIC


def compare0D(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC, dt=None):


    # Run Python stimulation
    tstart = time.time()
    t_Python, y_Python, stimstates_Python = SolverUS(a, neuron, Fdrive).run(neuron, Fdrive, Adrive,
                                                                            tstim, toffset, PRF, DC)
    tcomp_Python = time.time() - tstart
    t_Python *= 1e3
    Qm_Python = y_Python[2] * 1e5
    Vm_Python = y_Python[3]
    logger.debug('Python simulation completed in %ss', si_format(tcomp_Python, 2, space=' '))

    npatches, tpatch_on, tpatch_off = getPatchesLoc(t_Python, stimstates_Python)

    # Create NEURON single-node model and run simulation
    tstart = time.time()
    model = ExtendedSONIC(neuron, a=a, Fdrive=Fdrive)
    logger.debug('Creating model: %s', model)
    model.setUSAmps([Adrive * 1e-3])
    t_NEURON, stimstates_NEURON, Qprobes_NEURON, Vprobes_NEURON, _ = model.simulate(tstim, toffset,
                                                                                    PRF, DC, dt)
    tcomp_NEURON = time.time() - tstart
    Qm_NEURON = Qprobes_NEURON[0]
    Vm_NEURON = Vprobes_NEURON[0]
    logger.debug('NEURON Simulation completed in %ss', si_format(tcomp_NEURON, 2, space=' '))

    # Create comparative figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))
    t0 = -10.0
    y0 = neuron.Vm0
    fs = 10
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for item in ax.get_xticklabels() + ax.get_xticklabels():
            item.set_fontsize(fs)

    # Add onset to signals
    t_Python = np.hstack((np.array([t0, 0.]), t_Python))
    t_NEURON = np.hstack((np.array([t0, 0.]), t_NEURON))
    Qm_Python = np.hstack((np.ones(2) * y0, Qm_Python))
    Vm_Python = np.hstack((np.ones(2) * y0, Vm_Python))
    Qm_NEURON = np.hstack((np.ones(2) * y0, Qm_NEURON))
    Vm_NEURON = np.hstack((np.ones(2) * y0, Vm_NEURON))

    # Plot charge profiles
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

    # Plot effective potential profiles
    ax = axes[1]
    ax.plot(t_Python, Vm_Python, label='Python')
    ax.plot(t_NEURON, Vm_NEURON, label='NEURON')
    for i in range(npatches):
        ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                   facecolor='#8A8A8A', alpha=0.2)
    ax.legend(fontsize=fs, frameon=False)
    ax.set_xlim(t0, (tstim + toffset) * 1e3)
    ax.set_ylim(-150, 70)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('$V_{m, eff}$ (mV)', fontsize=fs)
    ax.set_title('membrane potential', fontsize=fs + 2)

    fig.subplots_adjust(top=0.8)
    fig.suptitle('{} point-neuron, a = {}m, f = {}Hz, A = {}Pa, {}s'
                 .format(neuron.name, *si_format([a, Fdrive, Adrive, tstim], space=' '),
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)

    return fig


def runAStim(neuron, nnodes, diam, L, Ra, connector, a, Fdrive, Adrive, tstim, toffset, PRF, DC,
             dt=None):

    # Create extended SONIC model with specific US frequency and connection scheme
    model = ExtendedSONIC(neuron, nnodes, diam, L, Ra, connector, a, Fdrive)
    logger.debug('Creating model: %s (%s) - %s connect scheme', model, model.details(),
                 'classic' if connector is None else 'custom')

    # Set node-specifc acoustic amplitudes
    amps = np.insert(np.zeros(nnodes - 1), 0, Adrive)
    model.setUSAmps(amps * 1e-3)

    # Run model simulation
    tstart = time.time()
    t, stimstates, Qprobes, Vprobes, _ = model.simulate(tstim, toffset, PRF, DC, dt)
    logger.debug('Simulation completed in %.2f ms', (time.time() - tstart) * 1e3)

    # Plot membrane potential and charge profiles
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))
    tonset = -10.0
    Vm0 = model.neuron.Vm0
    fs = 10
    lbls = ['node {} ({:.0f} kPa)'.format(i + 1, amps[i] * 1e-3) for i in range(nnodes)]

    ax = axes[0]
    plotSignals(t, Qprobes, states=stimstates, ax=ax, onset=(tonset, Vm0), lbls=lbls, fs=fs)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Qm (nC/cm2)', fontsize=fs)
    ax.set_title('membrane charge density', fontsize=fs + 2)

    ax = axes[1]
    plotSignals(t, Vprobes, states=stimstates, ax=ax, onset=(tonset, Vm0), lbls=lbls, fs=fs)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Vm (mV)', fontsize=fs)
    ax.set_title('membrane potential', fontsize=fs + 2)

    for ax in axes:
        ax.set_ylim(-150, 50)
    fig.subplots_adjust(top=0.8)
    fig.suptitle('{} neuron, {} node{}, diam = {}m, L = {}m, Ra = ${}$ ohm.cm - {}'
                 .format(neuron.name, nnodes, 's' if nnodes > 1 else '',
                         *si_format([model.diam, model.L], space=' '),
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)
    return fig


def runEStim(ax, neuron, nnodes, diam, L, Ra, connector, Astim, tstim, toffset, PRF, DC, dt):

    # Create extended SONIC model with adapted connection scheme
    model = ExtendedSONIC(neuron, nnodes, diam, L, Ra, connector)
    logger.debug('Creating model: %s (%s) - %s connect scheme', model, model.details(),
                 'classic' if connector is None else 'custom')

    # Attach electrical stimuli to appropriate nodes
    amps = np.insert(np.zeros(nnodes - 1), 0, Astim)
    model.attachEStims(amps, tstim, PRF, DC)

    # Run model simulation
    tstart = time.time()
    t, stimstates, Vprobes, *_ = model.simulate(tstim, toffset, PRF, DC, dt)
    logger.debug('Simulation completed in %.2f ms', (time.time() - tstart) * 1e3)

    # Plot membrane potential profiles
    tonset = -10.0
    fs = 10
    lbls = ['node {} ({:.0f} $mA/m^2$)'.format(i + 1, amps[i]) for i in range(nnodes)]

    plotSignals(t, Vprobes, states=stimstates, ax=ax, onset=(tonset, neuron.Vm0), lbls=lbls, fs=fs)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Vm (mV)', fontsize=fs)

    return ax


def compareEStim(neuron, nnodes, diam, L, Ra, connector, Astim, tstim, toffset, PRF, DC, dt=None):

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))

    # run with classic connect scheme
    runEStim(axes[0], neuron, nnodes, diam, L, Ra, None, Astim, tstim, toffset, PRF, DC, dt)
    axes[0].set_title('classic connect scheme', fontsize=12)

    # run with custom connect scheme
    runEStim(axes[1], neuron, nnodes, diam, L, Ra, connector, Astim, tstim, toffset, PRF, DC, dt)
    axes[1].set_title('custom connect scheme', fontsize=12)

    fig.subplots_adjust(top=0.8)
    fig.suptitle('{} neuron, {} node{}, diam = {}m, L = {}m, Ra = ${}$ ohm.cm - {}'
                 .format(neuron.name, nnodes, 's' if nnodes > 1 else '',
                         *si_format([diam, L], space=' '), pow10_format(Ra),
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)
    return fig
