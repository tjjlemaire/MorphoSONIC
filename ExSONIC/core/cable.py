# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 18:12:30

import abc
import numpy as np
from itertools import repeat

from PySONIC.neurons import *
from PySONIC.utils import si_format, pow10_format, logger
from PySONIC.constants import *
from PySONIC.postpro import findPeaks

from .pyhoc import *
from .node import IintraNode, SonicNode
from ..utils import VextPointSource
from ..constants import *


class Cable:
    ''' Simple 1D extension of the SONIC model. '''

    def __init__(self, pneuron, nnodes, rs, nodeD, nodeL, interD=0., interL=0., connector=None, verbose=False):
        ''' Class constructor, defining the model's sections geometry, topology and biophysics.
            Note: internodes are not represented in the model but rather used to set appropriate
            axial resistances between node compartments.

            :param nnodes: number of nodes
            :param rs: cytoplasmic resistivity (Ohm.cm)
            :param nodeD: list of node diameters (um) or single value (applied to all nodes)
            :param nodeL: list of node lengths (um) or single value (applied to all nodes)
            :param interD: list of internode diameters (um) or single value (applied to all internodes)
            :param interL: list of internode lengths (um) or single value (applied to all internodes)
            :param connector: object used to connect sections together through a custom
                axial current density mechanism
            :param verbose: boolean stating whether to print out details
        '''

        # Convert vector inputs to arrays and assign class attributes
        self.checkInputs(nnodes, nodeD, nodeL, interD, interL)
        self.pneuron = pneuron
        self.rs = rs  # Ohm.cm
        self.connector = connector
        self.has_vext_mech = False

        # Create node sections and set their geometry
        self.sections = self.createSections(['node{}'.format(i) for i in range(self.nnodes)])
        self.defineGeometry()  # must be called PRIOR to build_custom_topology()

        # Set sections membrane mechanism
        self.defineBiophysics()

        # Set sections resisitvity, corrected with internodal influence
        self.setResistivity()

        # Connect section together
        if self.nnodes > 1:
            if self.connector is None:
                self.buildTopology()
            else:
                self.buildCustomTopology()

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return 'Cable({}, {})'.format(self.strBiophysics(), self.strNodes())
        # 'classic_connect' if self.connector is None else repr(self.connector))

    def checkInputs(self, nnodes, nodeD, nodeL, interD, interL):
        # Pre-process node parameters
        if nnodes is None:
            for item in [nodeD, nodeL]:
                if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, np.ndarray):
                    nnodes = len(item)
                    break
            if nnodes is None:
                raise ValueError('nnodes must be provided for float-typed geometrical parameters')
        if isinstance(nodeD, float):
            nodeD = [nodeD] * nnodes
        if isinstance(nodeL, float):
            nodeL = [nodeL] * nnodes

        # Check consistency of node parameters
        if len(nodeD) != len(nodeL):
            raise ValueError('Inconsistent numbers of node diameters ({}) and lengths ({})'.format(
                len(nodeD), len(nodeL)))

        # Pre-process internode parameters
        if isinstance(interD, float):
            interD = [interD] * (nnodes - 1)
        if isinstance(interL, float):
            interL = [interL] * (nnodes - 1)

        # Check consistency of internode parameters
        if len(interD) != nnodes - 1:
            raise ValueError(
                'Number of internode diameters ({}) does not match nnodes - 1 ({})'.format(
                    len(interD), nnodes - 1))
        if len(interL) != nnodes - 1:
            raise ValueError(
                'Number of internode lengths ({}) does not match nnodes - 1 ({})'.format(
                    len(interD), nnodes - 1))

        self.nnodes = nnodes
        self.nodeD = np.array(nodeD)    # um
        self.nodeL = np.array(nodeL)    # um
        self.interD = np.array(interD)  # um
        self.interL = np.array(interL)  # um

    def getNodeCoordinates(self):
        ''' Return vector of node coordinates along axial dimension, centered at zero (um). '''
        xcoords = np.zeros(self.nnodes)
        for i in range(1, self.nnodes):
            xcoords[i] = xcoords[i - 1] + (self.nodeL[i - 1] + self.nodeL[i]) / 2 + self.interL[i - 1]
        return xcoords - xcoords[-1] / 2.

    def getNode2NodeDistance(self, i, j):
        ''' Return node-to-node distance along axial dimension (um). '''
        xcoords = self.getNodeCoordinates()
        return np.abs(xcoords[j] - xcoords[i])

    def strNodes(self):
        return '{} node{}'.format(self.nnodes, 's' if self.nnodes > 1 else '')

    def strResistivity(self):
        return 'rs = ${}$ ohm.cm'.format(pow10_format(self.rs))

    def strGeom(self):
        ''' Format model geometrical parameters into string. '''
        params = {
            'nodeD': self.nodeD,
            'nodeL': self.nodeL,
            'interD': self.interD,
            'interL': self.interL
        }
        lbls = {}
        for key, val in params.items():
            if np.all(val == val[0]):
                lbls[key] = '{} = {}m'.format(key, si_format(val[0] * 1e-6, 1, space=' '))
            else:
                lbls[key] = '{} = [{}] um'.format(key, ', '.join(['{:.2f}'.format(x) for x in val]))
        return ', '.join(lbls.values())

    def pprint(self):
        ''' Pretty-print naming of the model instance. '''
        return ('{} neuron, {}, {}, {}').format(
            self.pneuron.name, self.strNodes(), self.strResistivity(), self.strGeom())

    @property
    @abc.abstractmethod
    def createSections(self, ids):
        ''' Create morphological sections. '''
        return NotImplementedError

    def defineGeometry(self):
        ''' Set sections geometry. '''
        logger.debug('defining sections geometry: {}'.format(self.strGeom()))
        for i, sec in enumerate(self.sections):
            sec.diam = self.nodeD[i]  # um
            sec.L = self.nodeL[i]     # um
            sec.nseg = 1

    def defineBiophysics(self):
        ''' Set section-specific membrane properties with specific sonophore membrane coverage. '''
        if self.verbose:
            print('defining membrane biophysics: {}'.format(self.strBiophysics()))
        for sec in self.sections:
            sec.insert(self.mechname)

    def relResistance(self, D, L):
        ''' Return relative resistance of cylindrical section based on its diameter and length. '''
        return 4 * L / (np.pi * D**2)

    def setResistivity(self):
        ''' Set sections axial resistivity, corrected to account for internodes and membrane capacitance
            in the Q-based differentiation scheme. '''

        # Set resistivity of nodal sections
        for sec in self.sections:
            sec.Ra = self.rs

        # Adjust resistivity to account for internodal resistances
        if not np.all(self.interL == 0.):
            if self.verbose:
                print('adjusting resistivity to account for internodal sections')
            for i, sec in enumerate(self.sections):
                # compute node relative resistance
                r_node = self.relResistance(self.nodeD[i], self.nodeL[i])

                # compute relative resistances of half of previous and/or next internodal sections, if any
                r_inter = 0.
                if i > 0:
                    r_inter += self.relResistance(self.interD[i - 1], self.interL[i - 1] / 2)
                if i < self.nnodes - 1:
                    r_inter += self.relResistance(self.interD[i], self.interL[i] / 2)

                # correct axial resistivity
                sec.Ra *= (r_node + r_inter) / r_node

        # In case the axial coupling variable is v (an alias for membrane charge density),
        # multiply resistivity by membrane capacitance to ensure consistency of Q-based
        # differential scheme, where Iax = dV / r = dQ / (r * cm)
        if self.connector is None or self.connector.vref == 'v':
            if self.verbose:
                print('adjusting resistivity to account for Q-based differential scheme')
        for sec in self.sections:
            sec.Ra *= self.neuron.Cm0 * 1e2

    def buildTopology(self):
        ''' Connect the sections in series through classic NEURON implementation. '''
        if self.verbose:
            print('building standard topology')
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            sec2.connect(sec1, 1, 0)

    def buildCustomTopology(self):
        if self.verbose:
            print('building custom {}-based topology'.format(self.connector.vref))
        list(map(self.connector.attach, self.sections))
        for sec1, sec2 in zip(self.sections[:-1], self.sections[1:]):
            self.connector.connect(sec1, sec2)

    def processInputs(self, values, config):
        ''' Return section specific inputs for a given configuration.

            :param values: model-sized vector of inputs, or single value
            :param config: spatial configuration ofr input application, used if values is float.
            Can be either 'first', 'central' or 'all'.
            :return: model-sized vector of inputs.
        '''
        # Process inputs
        if isinstance(values, float):
            if config == 'first':
                values = np.insert(np.zeros(self.nnodes - 1), 0, values)
            elif config == 'central':
                if self.nnodes % 2 == 0:
                    raise ValueError('"central" stimulus not applicable for an even number of nodes')
                halfpad = np.zeros(self.nnodes // 2)
                values = np.hstack((halfpad, np.array([values]), halfpad))
            elif config == 'all':
                values = np.ones(self.nodes) * values
            else:
                raise ValueError('Unknown stimulus configuration')
        else:
            if self.nnodes != len(values):
                raise ValueError('Stimulus distribution vector does not match number of nodes')
            values = np.array(values)
        return values

    @property
    @abc.abstractmethod
    def setStimAmp(self, amps, config):
        ''' Set distributed stimulus amplitudes. '''
        return NotImplementedError

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        for sec in self.sections:
            setattr(sec, 'stimon_{}'.format(self.mechname), value)
        return value

    def integrate(self, tstop, tstim, PRF, DC, dt, atol):
        return self.sections[0].integrate(tstop, tstim, PRF, DC, dt, atol)

    def simulate(self, A, tstim, toffset, PRF, DC, dt, atol):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param A: stimulus amplitude (in modality units)
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
        # states = {
        #     state: [{
        #         suffix: setRangeProbe(sec, '{}_{}_{}'.format(alias(state), suffix, self.mechname))
        #         for suffix in ['0', 'US']
        #     } for sec in self.sections] for state in self.neuron.states
        # }

        states = {}
        for key in self.neuron.states:
            # print(key, '{}_{}'.format(alias(key), self.mechname))
            states[key] = [setRangeProbe(sec, '{}_{}'.format(alias(key), self.mechname))
                           for sec in self.sections]

        # states = {
        #     state: [
        #         setRangeProbe(sec, '{}_{}'.format(alias(state), self.mechname))
        #         for sec in self.sections]
        #     for state in self.neuron.states
        # }


        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Retrieve output variables
        t = vec_to_array(tprobe) * 1e-3  # s
        stimon = vec_to_array(stimon)
        Qm = np.array(list(map(vec_to_array, Qm))) * 1e-5  # C/cm2
        Vmeff = np.array(list(map(vec_to_array, Vmeff)))  # mV
        # states = {
        #     key: [
        #         fs * vec_to_array(val['US']) + (1 - fs) * vec_to_array(val['0'])
        #         for val, fs in zip(states[key], self.nodeFs)
        #     ] for key in self.neuron.states
        # }
        states = {key: np.array(list(map(vec_to_array, val))) for key, val in states.items()}

        return t, stimon, Qm, Vmeff, states

    def titrate(self, tstim, toffset, PRF, DC, dt, atol, config, inode=0, Arange=None):
        Iinj = (Irange[0] + Irange[1]) / 2
        self.setIinj(Iinj, config)

        # Run simulation and detect spikes on ith trace
        t, stimon, Qm, Vmeff, states = self.simulate(tstim, toffset, PRF, DC, dt, atol)
        ipeaks, *_ = findPeaks(Vmeff[inode], mph=SPIKE_MIN_VAMP, mpp=SPIKE_MIN_VPROM)
        nspikes = ipeaks.size

        # If accurate threshold is found, return simulation results
        if (Irange[1] - Irange[0]) <= DELTA_IINJ_INTRA_MIN and nspikes == 1:
            print('threshold amplitude: {}A/m2'.format(si_format(Iinj * 1e-3, 2, space=' ')))
            return Iinj

        # Otherwise, refine titration interval and iterate recursively
        else:
            if nspikes == 0:
                Irange = (Iinj, Irange[1])
            else:
                Irange = (Irange[0], Iinj)
            return self.titrateIinjIntra(
                tstim, toffset, PRF, DC, dt, atol, config, inode=inode, Irange=Irange)


class IintraCable(Cable):

    def createSections(self, ids):
        self.sections = [IintraNode(self.pneuron, id) for id in ids]

    def setStimAmp(self, amps, config):
        ''' Set section specific electrical stimulation amplitudes.

            :param amps: model-sized vector of electrical amplitudes (mA/m2) or single value.
            :return: section-specific labels
        '''

        # Set IClamp objects
        amps = self.processInputs(amps, config)  # mA/m2
        if self.verbose:
            print('Injecting intracellular currents: Iinj = [{}] mA/m2'.format(
                ', '.join('{:.0f}'.format(Astim) for Astim in amps)))
        self.Iinjs = [Astim * sec(0.5).area() * 1e-6
                      for Astim, sec in zip(amps, self.sections)]  # nA
        self.iclamps = []
        for sec in self.sections:
            pulse = h.IClamp(sec(0.5))
            pulse.delay = 0  # we want to exert control over amp starting at 0 ms
            pulse.dur = 1e9  # dur must be long enough to span all our changes
            self.iclamps.append(pulse)
        self.modality = 'Iinj'

        # return node-specific labels
        return ['node {} ({:.0f} $mA/m^2$)'.format(i + 1, A) for i, A in enumerate(amps)]

    def setStimON(self, value):
        value = super().setStimON(value)
        for iclamp, Iinj in zip(self.iclamps, self.Iinjs):
            iclamp.amp = value * Iinj
        return value


class VextCable(Cable):

    def createSections(self, ids):
        self.sections = [VextNode(self.pneuron, id) for id in ids]

    def setStimAmp(self, Vexts):
        ''' Insert extracellular mechanism into node sections and set extracellular potential values.

            :param Vexts: model-sized vector of extracellular potentials (mV)
            or single value (assigned to first node)
            :return: section-specific labels
        '''

        # Insert extracellular mechanism in nodes sections
        Vexts = self.processInputs(Vexts, None)  # mV
        if not self.has_vext_mech:
            if self.verbose:
                print('Inserting extracellular mechanism in all nodes')
            for sec in self.sections:
                insertVext(sec)
            self.has_vext_mech = True
            self.modality = 'Vext'

        # Set extracellular potential values
        if self.verbose:
            print('Setting extracellular potentials: Vexts = [{}] mV'.format(
                ', '.join('{:.5f}'.format(Vext) for Vext in Vexts)))
        self.Vexts = Vexts  # mV

        # return node-specific labels
        return ['node {} ({:.0f} $mV$)'.format(i + 1, Vext) for i, Vext in enumerate(Vexts)]



class SonicCable(Cable):

    def createSections(self, ids):
        self.sections = [SonicNode(self.pneuron, id=id, a=self.a, ) for id in ids]

