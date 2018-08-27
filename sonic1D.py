# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-15 15:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-27 22:14:59

import time
import logging
from itertools import repeat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PySONIC.neurons import *
from PySONIC.utils import InputError, si_format, logger, pow10_format
from PySONIC.plt import plotSignals

from pyhoc import *
from sonic0D import Sonic0D
from connectors import SeriesConnector


logger.setLevel(logging.DEBUG)


class Sonic1D(Sonic0D):
    ''' Simple 1D extension of the SONIC model. '''

    def __init__(self, neuron, nsec=1, diam=1e-6, L=1e-6, Ra=1e2, connector=None,
                 a=32e-9, Fdrive=500e3, verbose=False):
        ''' Initialization.

            :param neuron: neuron object
            :param nsec: number of sections
            :param diam: section diameter (m)
            :param L: section length (m)
            :param Ra: cytoplasmic resistivity (Ohm.cm)
            :param connector: object used to connect sections together through a custom
                axial current density mechanism
            :param a: sonophore diameter (nm)
            :param Fdrive: ultrasound frequency (Hz)
            :param verbose: boolean stating whether to print out details
        '''
        self.nsec = nsec
        self.diam = diam * 1e6  # um
        self.L = L * 1e6  # m
        self.Ra = Ra  # Ohm.cm
        self.connector = connector

        # Initialize point-neuron model and delete its single section
        super().__init__(neuron, a, Fdrive, verbose)
        del self.section

        # Create sections and set their geometry
        self.sections = self.createSections(['node{}'.format(i) for i in range(self.nsec)])
        self.defineGeometry()  # must be called PRIOR to build_custom_topology()

        # Set sections membrane mechanism
        self.defineBiophysics(self.sections)
        for sec in self.sections:
            sec.Ra = Ra

        # Connect section together
        if self.nsec > 1:
            if self.connector is None:
                self.buildTopology()
            else:
                self.buildCustomTopology()

        # Set zero acoustic amplitudes
        self.setUSAmps(np.zeros(self.nsec))

    def __str__(self):
        ''' Explicit naming of the model instance. '''
        return super(Sonic1D, self).__str__() + '_{}node{}_{}_connect'.format(
            self.nsec, 's' if self.nsec > 1 else '',
            'classic' if self.connector is None else 'custom')

    def details(self):
        ''' Details about model instance. '''
        return 'diam = {}m, L = {}m, Ra = {:.0e} ohm.cm'\
            .format(*si_format([self.diam * 1e-6, self.L * 1e-6], space=' '), self.Ra)

    def createSections(self, ids):
        ''' Create morphological sections.

            :param id: names of the sections.
        '''
        return list(map(super(Sonic1D, self).createSection, ids))

    def defineGeometry(self):
        ''' Set the 3D geometry of the model. '''
        for sec in self.sections:
            sec.diam = self.diam  # um
            sec.L = self.L  # um
            sec.nseg = 1

    def buildTopology(self):
        ''' Connect the sections in series through classic NEURON implementation. '''
        # for i in range(self.nsec - 1):
        #     self.sections[i + 1].connect(self.sections[i], 1, 0)
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            sec2.connect(sec1, 1, 0)

    def buildCustomTopology(self):
        self.sections = list(map(self.connector.attach, self.sections))
        # for i in range(self.nsec - 1):
        #     self.connector.connect(self.sections[i], self.sections[i + 1])
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            self.connector.connect(sec1, sec2)

    def setUSAmps(self, amps):
        ''' Set section specific US stimulation amplitudes '''
        if len(self.sections) != len(amps):
            raise InputError('Amplitude distribution vector does not match number of sections')
        if self.verbose:
            print('Setting acoustic stimulus amplitudes: Adrive = [{}] kPa'.format(
                ' - '.join('{:.0f}'.format(Adrive) for Adrive in amps)))
        for sec, Adrive in zip(self.sections, amps):
            setattr(sec, 'Adrive_{}'.format(self.mechname), Adrive)
        self.modality = 'US'

    def setElecAmps(self, amps):
        if len(self.sections) != len(amps):
            raise InputError('Amplitude distribution vector does not match number of sections')
        if self.verbose:
            print('Setting electrical stimulus amplitudes: Astim = [{}] mA/m2'.format(
                ' - '.join('{:.0f}'.format(Astim) for Astim in amps)))
        self.Iinjs = [Astim * sec(0.5).area() * 1e-6
                      for Astim, sec in zip(amps, self.sections)]  # nA
        self.iclamps = []
        for sec in self.sections:
            pulse = h.IClamp(sec(0.5))
            pulse.delay = 0  # we want to exert control over amp starting at 0 ms
            pulse.dur = 1e9  # dur must be long enough to span all our changes
            self.iclamps.append(pulse)
        self.modality = 'elec'

    def setStimON(self, value):
        ''' Set US or electrical stimulation ON or OFF by updating the appropriate
            mechanism/object parameter.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        for sec in self.sections:
            setattr(sec, 'stimon_{}'.format(self.mechname), value)
        if self.modality == 'elec':
            for iclamp, Iinj in zip(self.iclamps, self.Iinjs):
                iclamp.amp = value * Iinj
        return value

    def simulate(self, tstim, toffset, PRF, DC, dt, atol):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step (s)
        '''

        # Set recording vectors
        tprobe = setTimeProbe()
        stimprobe = setStimProbe(self.sections[0], self.mechname)
        vprobes = list(map(setRangeProbe, self.sections, repeat('v')))
        Vmeffprobes = list(map(setRangeProbe, self.sections,
                               repeat('Vmeff_{}'.format(self.mechname))))
        statesprobes = [list(map(setRangeProbe, self.sections,
                                 repeat('{}_{}'.format(alias(state), self.mechname))))
                        for state in self.neuron.states_names]

        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Retrieve output variables
        t = Vec2array(tprobe)  # ms
        stimon = Vec2array(stimprobe)
        vprobes = np.array(list(map(Vec2array, vprobes)))  # mV or nC/cm2
        Vmeffprobes = np.array(list(map(Vec2array, Vmeffprobes)))  # mV
        statesprobes = {state: np.array(list(map(Vec2array, probes)))
                        for state, probes in zip(self.neuron.states_names, statesprobes)}

        return t, stimon, vprobes, Vmeffprobes, statesprobes



def runAStim(neuron, nnodes, diam, L, Ra, connector, a, Fdrive, Adrive, tstim, toffset, PRF, DC,
             dt=None, atol=None):

    # Create extended SONIC model with specific US frequency and connection scheme
    model = Sonic1D(neuron, nnodes, diam, L, Ra, connector, a, Fdrive)
    logger.debug('Creating model: %s (%s) - %s connect scheme', model, model.details(),
                 'classic' if connector is None else 'custom')

    # Set node-specifc acoustic amplitudes
    amps = np.insert(np.zeros(nnodes - 1), 0, Adrive)
    model.setUSAmps(amps * 1e-3)

    # Run model simulation
    tstart = time.time()
    t, stimon, Qprobes, Vprobes, _ = model.simulate(tstim, toffset, PRF, DC, dt, atol)
    logger.debug('Simulation completed in %.2f ms', (time.time() - tstart) * 1e3)

    # Plot membrane potential and charge profiles
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))
    tonset = -10.0
    Vm0 = model.neuron.Vm0
    fs = 10
    lbls = ['node {} ({:.0f} kPa)'.format(i + 1, amps[i] * 1e-3) for i in range(nnodes)]

    ax = axes[0]
    plotSignals(t, Qprobes, states=stimon, ax=ax, onset=(tonset, Vm0), lbls=lbls, fs=fs)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Qm (nC/cm2)', fontsize=fs)
    ax.set_title('membrane charge density', fontsize=fs + 2)

    ax = axes[1]
    plotSignals(t, Vprobes, states=stimon, ax=ax, onset=(tonset, Vm0), lbls=lbls, fs=fs)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Vm (mV)', fontsize=fs)
    ax.set_title('membrane potential', fontsize=fs + 2)

    for ax in axes:
        ax.set_ylim(-150, 50)
    fig.subplots_adjust(top=0.8)
    fig.suptitle('{} neuron, {} node{}, diam = {}m, L = {}m, Ra = ${}$ ohm.cm - {}'
                 .format(neuron.name, nnodes, 's' if nnodes > 1 else '',
                         *si_format([model.diam, model.L], space=' '), Ra,
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)
    return fig


def runEStim(neuron, nnodes, diam, L, Ra, connector, amps, tstim, toffset, PRF, DC,
             dt=None, atol=None):
    ''' Create extended SONIC model with adapted connection scheme and run electrical simulation. '''
    model = Sonic1D(neuron, nnodes, diam, L, Ra, connector)
    model.setElecAmps(amps)
    return model.simulate(tstim, toffset, PRF, DC, dt, atol)


def compareEStim(neuron, nnodes, diam, L, Ra, connector, Astim, tstim, toffset, PRF, DC,
                 dt=None, atol=None):

    # Set amplitude distribution vector
    amps = np.insert(np.zeros(nnodes - 1), 0, Astim)

    # Run model simulation with classic connect scheme
    tstart = time.time()
    t_classic, stimon_classic, Vprobes_classic, *_ = runEStim(
        neuron, nnodes, diam, L, Ra, None, amps, tstim, toffset, PRF, DC, dt, atol)
    tcomp_classic = time.time() - tstart
    stimon_classic[0] = 1
    # logger.debug('Simulation completed in %.2f ms', (tcomp_classic * 1e3)

    # Run model simulation with custom connect scheme
    tstart = time.time()
    t_custom, stimon_custom, Vprobes_custom, *_ = runEStim(
        neuron, nnodes, diam, L, Ra, connector, amps, tstim, toffset, PRF, DC, dt, atol)
    tcomp_custom = time.time() - tstart
    stimon_custom[0] = 1
    # logger.debug('Simulation completed in %.2f ms', tcomp_custom * 1e3)

    # Create comparative figure
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
    axes = list(map(plt.subplot, gs))
    fig.subplots_adjust(top=0.8, left=0.05, right=0.95)
    fig.suptitle('{} neuron, {} node{}, diam = {}m, L = {}m, Ra = ${}$ ohm.cm - {}'
                 .format(neuron.name, nnodes, 's' if nnodes > 1 else '',
                         *si_format([diam, L], space=' '), pow10_format(Ra),
                         'adaptive time step' if dt is None else 'dt = ${}$ ms'
                         .format(pow10_format(dt * 1e3))),
                 fontsize=18)
    tonset = -10.0
    fs = 10
    lbls = ['node {} ({:.0f} $mA/m^2$)'.format(i + 1, amps[i]) for i in range(nnodes)]

    # Plot V-traces for classic scheme
    ax = axes[0]
    plotSignals(t_classic, Vprobes_classic, states=stimon_classic, ax=ax, onset=(tonset, neuron.Vm0),
                lbls=lbls, fs=fs)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Vm (mV)', fontsize=fs)
    ax.set_title('classic connect scheme', fontsize=12)

    # Plot V-traces for custom scheme
    ax = axes[1]
    plotSignals(t_custom, Vprobes_custom, states=stimon_custom, ax=ax, onset=(tonset, neuron.Vm0),
                lbls=lbls, fs=fs)
    ax.set_xlim(tonset, (tstim + toffset) * 1e3)
    ax.set_ylim(-100, 50)
    ax.set_xlabel('time (ms)', fontsize=fs)
    ax.set_ylabel('Vm (mV)', fontsize=fs)
    ax.set_title('custom connect scheme', fontsize=12)

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


if __name__ == '__main__':

    # Model parameters
    neuron = CorticalRS()
    a = 32e-9  # sonophore diameter
    nnodes = 2
    Ra = 1e2  # default order of magnitude found in litterature (Ohm.cm)
    diam = 1e-6  # order of magnitude of axon node diameter (m)
    L = 1e-5  # typical in-plane diameter of sonophore structure (m)
    a = 32e-9  # sonophore diameter

    # Stimulation parameters
    Fdrive = 500e3  # Hz
    Adrive = 50e3  # kPa
    Astim = 30.0  # mA/m2
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    PRF = 100.  # Hz
    DC = 1.0

    # SeriesConnector object to connect sections in series through custom implementation
    connector = SeriesConnector(mechname='Iax', vref='Vmeff_{}'.format(neuron.name))
    print(connector)

    # fig1 = compareEStim(neuron, nnodes, diam, L, Ra, connector, Astim, tstim, toffset, PRF, DC, dt=1e-5)

    fig2 = runAStim(neuron, nnodes, diam, L, 1e11, connector, a, Fdrive, Adrive, tstim, toffset,
                    PRF, DC)

    plt.show()
