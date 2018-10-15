# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-15 15:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-15 23:33:59

import numpy as np
from itertools import repeat

from PySONIC.neurons import *
from PySONIC.utils import si_format, pow10_format
from PySONIC.constants import *

from ..pyhoc import *
from .._0D import Sonic0D


class Sonic1D(Sonic0D):
    ''' Simple 1D extension of the SONIC model. '''

    def __init__(self, neuron, Ra, diams, lengths, connector=None,
                 a=32e-9, covs=1., Fdrive=500e3,
                 verbose=False, nsec=None):
        ''' Initialization.

            :param neuron: neuron object
            :param Ra: cytoplasmic resistivity (Ohm.cm)
            :param diams: list of section diameters (m) or single value (applied to all sections)
            :param lengths: list of section lengths (m) or single value (applied to all sections)
            :param covs: list of membrane sonophore coverage fraction in each section,
              or single value (applied to all sections)
            :param connector: object used to connect sections together through a custom
                axial current density mechanism
            :param a: sonophore diameter (m)
            :param Fdrive: ultrasound frequency (Hz)
            :param verbose: boolean stating whether to print out details
        '''

        # Pre-process inputs
        if nsec is None:
            for item in [diams, lengths, covs]:
                if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, np.ndarray):
                    nsec = len(item)
                    break
            if nsec is None:
                raise ValueError('nsec must be provided for float-typed geometrical parameters')
        if isinstance(diams, float):
            diams = [diams] * nsec
        if isinstance(lengths, float):
            lengths = [lengths] * nsec
        if isinstance(covs, float):
            covs = [covs] * nsec

        # Check inputs validity
        if len(diams) != len(lengths):
            raise ValueError('Inconsistent numbers of section diameters ({}) and lengths ({})'.format(
                len(diams), len(lengths)))
        if len(diams) != len(covs):
            raise ValueError(
                'Inconsistent numbers of section diameters ({}) and coverages ({})'.format(
                    len(diams), len(covs)))
        for i, fs in enumerate(covs):
            if fs > 1. or fs < 0.:
                raise ValueError('covs[{}] ({}) must be within [0-1]'.format(i, fs))

        # Assign inputs as arguments
        self.diams = np.array(diams) * 1e6  # um
        self.lengths = np.array(lengths) * 1e6  # um
        self.covs = np.array(covs)
        self.nsec = self.diams.size
        self.Ra = Ra  # Ohm.cm
        self.connector = connector

        # Initialize point-neuron model and delete its single section
        super().__init__(neuron, a=a, Fdrive=Fdrive, verbose=verbose)
        del self.section

        # Create sections and set their geometry
        self.sections = self.createSections(['node{}'.format(i) for i in range(self.nsec)])
        self.defineGeometry()  # must be called PRIOR to build_custom_topology()

        # Set sections membrane mechanism
        self.defineBiophysics()
        for sec in self.sections:
            sec.Ra = Ra

        # Connect section together
        if self.nsec > 1:
            if self.connector is None:
                self.buildTopology()
            else:
                self.buildCustomTopology()


    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return 'SONIC1D_{}node{}_{}'.format(
            self.nsec, 's' if self.nsec > 1 else '',
            'classic_connect' if self.connector is None else repr(self.connector))

    def pprint(self):
        ''' Pretty-print naming of the model instance. '''
        if np.all(self.diams == self.diams[0]):
            d_str = '{}m'.format(si_format(self.diams[0] * 1e-6, space=' '))
        else:
            d_str = '[{}] um'.format(', '.join(['{:.2f}'.format(x) for x in self.diams]))
        if np.all(self.lengths == self.lengths[0]):
            L_str = '{}m'.format(si_format(self.lengths[0] * 1e-6, space=' '))
        else:
            L_str = '[{}] um'.format(', '.join(['{:.2f}'.format(x) for x in self.lengths]))
        if np.all(self.covs == self.covs[0]):
            cov_str = '{:.0f}%'.format(self.covs[0] * 1e2)
        else:
            cov_str = '[{}] %'.format(', '.join(['{:.0f}'.format(x * 1e2) for x in self.covs]))

        return '{} neuron, {} node{}, Ra = ${}$ ohm.cm, d = {}, L = {}, cov = {}'.format(
            self.neuron.name, self.nsec, 's' if self.nsec > 1 else '',
            pow10_format(self.Ra), d_str, L_str, cov_str)

    def createSections(self, ids):
        ''' Create morphological sections.

            :param id: names of the sections.
        '''
        return list(map(super(Sonic1D, self).createSection, ids))

    def defineGeometry(self):
        ''' Set the 3D geometry of the model. '''
        for i, sec in enumerate(self.sections):
            sec.diam = self.diams[i]  # um
            sec.L = self.lengths[i]  # um
            sec.nseg = 1

    def defineBiophysics(self):
        ''' Set section-specific membrane properties with specific sonophore membrane coverage. '''
        for sec, fs in zip(self.sections, self.covs):
            sec.insert(self.mechname)
            setattr(sec, 'fs_{}'.format(self.mechname), fs)

    def buildTopology(self):
        ''' Connect the sections in series through classic NEURON implementation. '''
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            sec2.connect(sec1, 1, 0)

    def buildCustomTopology(self):
        list(map(self.connector.attach, self.sections, [True] + [False] * (self.nsec - 1)))
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            self.connector.connect(sec1, sec2)

    def setUSAmps(self, amps):
        ''' Set section specific acoustic stimulation amplitudes.

            :param amps: model-sized vector or electrical amplitudes (Pa)
            or single value (assigned to first node)
            :return: section-specific amplitude labels
        '''

        if self.connector is None:
            raise ValueError(
                'attempting to perform A-STIM simulation with standard "v-based" connection scheme')
        if self.nsec != len(amps):
            raise ValueError('Amplitude distribution vector does not match number of sections')

        # Conversion Pa -> kPa
        amps = np.array(amps) * 1e-3

        # Set acoustic amplitudes
        if self.verbose:
            print('Setting acoustic stimulus amplitudes: Adrive = [{}] kPa'.format(
                ' - '.join('{:.0f}'.format(Adrive) for Adrive in amps)))
        for sec, Adrive in zip(self.sections, amps):
            setattr(sec, 'Adrive_{}'.format(self.mechname), Adrive)
        self.modality = 'US'

        # Return section-specific amplitude labels
        return ['node {} ({:.0f} kPa)'.format(i + 1, A) for i, A in enumerate(amps)]

    def setElecAmps(self, amps):
        ''' Set section specific electrical stimulation amplitudes.

            :param amps: model-sized vector or electrical amplitudes (mA/m2)
            or single value (assigned to first node)
            :return: section-specific amplitude labels
        '''

        # Pre-process input
        if isinstance(amps, float):
            amps = np.insert(np.zeros(self.nsec - 1), 0, amps)
        else:
            amps = np.array(amps)
        if len(self.sections) != len(amps):
            raise ValueError('Amplitude distribution vector does not match number of sections')

        # Set IClamp objects
        if self.verbose:
            print('Setting electrical stimulus amplitudes: Astim = [{}] mA/m2'.format(
                ', '.join('{:.0f}'.format(Astim) for Astim in amps)))
        self.Iinjs = [Astim * sec(0.5).area() * 1e-6
                      for Astim, sec in zip(amps, self.sections)]  # nA
        self.iclamps = []
        for sec in self.sections:
            pulse = h.IClamp(sec(0.5))
            pulse.delay = 0  # we want to exert control over amp starting at 0 ms
            pulse.dur = 1e9  # dur must be long enough to span all our changes
            self.iclamps.append(pulse)
        self.modality = 'elec'

        # return node-specific amplitude labels
        return ['node {} ({:.0f} $mA/m^2$)'.format(i + 1, A) for i, A in enumerate(amps)]

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
        stimon = setStimProbe(self.sections[0], self.mechname)
        Qm = list(map(setRangeProbe, self.sections, repeat('v')))
        Vmeff = list(map(setRangeProbe, self.sections,
                         repeat('Vmeff_{}'.format(self.mechname))))
        states = {
            state: [{
                suffix: setRangeProbe(sec, '{}_{}_{}'.format(alias(state), suffix, self.mechname))
                for suffix in ['0', 'US']
            } for sec in self.sections] for state in self.neuron.states_names
        }

        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Retrieve output variables
        t = Vec2array(tprobe) * 1e-3  # s
        stimon = Vec2array(stimon)
        Qm = np.array(list(map(Vec2array, Qm))) * 1e-5  # C/cm2
        Vmeff = np.array(list(map(Vec2array, Vmeff)))  # mV
        states = {
            key: [
                fs * Vec2array(val['US']) + (1 - fs) * Vec2array(val['0'])
                for val, fs in zip(states[key], self.covs)
            ] for key in self.neuron.states_names
        }

        return t, stimon, Qm, Vmeff, states
