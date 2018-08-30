# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-15 15:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 11:29:24

from itertools import repeat

from PySONIC.neurons import *
from PySONIC.utils import InputError, si_format
from PySONIC.constants import *

from ..pyhoc import *
from .._0D import Sonic0D


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
            :param a: sonophore diameter (m)
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
        return 'SONIC1D_{}node{}_{}_connect'.format(
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
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            sec2.connect(sec1, 1, 0)

    def buildCustomTopology(self):
        list(map(self.connector.attach, self.sections, [True] + [False] * (self.nsec - 1)))
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            self.connector.connect(sec1, sec2)

    def setUSAmps(self, amps):
        ''' Set section specific acoustic stimulation amplitudes '''
        if len(self.sections) != len(amps):
            raise InputError('Amplitude distribution vector does not match number of sections')
        if self.verbose:
            print('Setting acoustic stimulus amplitudes: Adrive = [{}] kPa'.format(
                ' - '.join('{:.0f}'.format(Adrive) for Adrive in amps)))
        for sec, Adrive in zip(self.sections, amps):
            setattr(sec, 'Adrive_{}'.format(self.mechname), Adrive)
        self.modality = 'US'

    def setElecAmps(self, amps):
        ''' Set section specific electrical stimulation amplitudes '''
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
