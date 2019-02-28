# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-15 15:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-23 18:05:54

import numpy as np
from itertools import repeat

from PySONIC.neurons import *
from PySONIC.utils import si_format, pow10_format
from PySONIC.constants import *
from PySONIC.postpro import findPeaks

from ..pyhoc import *
from .._0D import Sonic0D
from ..utils import VextPointSource
from ..constants import *



class Sonic1D(Sonic0D):
    ''' Simple 1D extension of the SONIC model. '''

    def __init__(self, neuron, rs, nodeD, nodeL, interD=0., interL=0., connector=None,
                 nnodes=None, a=None, Fdrive=None, verbose=False):
        ''' Class constructor, defining the model's sections geometry, topology and biophysics.
            Note: internodes are not represented in the model but rather used to set appropriate
            axial resistances between node compartments.

            :param neuron: neuron object
            :param rs: cytoplasmic resistivity (Ohm.cm)
            :param nodeD: list of node diameters (um) or single value (applied to all nodes)
            :param nodeL: list of node lengths (um) or single value (applied to all nodes)
            :param interD: list of internode diameters (um) or single value (applied to all internodes)
            :param interL: list of internode lengths (um) or single value (applied to all internodes)
            :param nnodes: number of nodes (applied only if all parameters are passed as floats)
            :param connector: object used to connect sections together through a custom
                axial current density mechanism
            :param a: sonophore diameter (nm)
            :param Fdrive: ultrasound frequency (kHz)
            :param verbose: boolean stating whether to print out details
        '''

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
        # if isinstance(nodeFs, float):
        #     nodeFs = [nodeFs] * nnodes

        # Check consistency of node parameters
        if len(nodeD) != len(nodeL):
            raise ValueError('Inconsistent numbers of node diameters ({}) and lengths ({})'.format(
                len(nodeD), len(nodeL)))
        # if len(nodeD) != len(nodeFs):
        #     raise ValueError(
        #         'Inconsistent numbers of node diameters ({}) and coverages ({})'.format(
        #             len(nodeD), len(nodeFs)))
        # for i, fs in enumerate(nodeFs):
        #     if fs > 1. or fs < 0.:
        #         raise ValueError('nodeFs[{}] ({}) must be within [0-1]'.format(i, fs))

        # Pre-process internode parameters
        if isinstance(interD, float):
            interD = [interD] * (nnodes - 1)
        if isinstance(interL, float):
            interL = [interL] * (nnodes - 1)
        # if isinstance(nodeFs, float):
        #     nodeFs = [nodeFs] * (nnodes - 1)

        # Check consistency of internode parameters
        if len(interD) != nnodes - 1:
            raise ValueError(
                'Number of internode diameters ({}) does not match nnodes - 1 ({})'.format(
                    len(interD), nnodes - 1))
        if len(interL) != nnodes - 1:
            raise ValueError(
                'Number of internode lengths ({}) does not match nnodes - 1 ({})'.format(
                    len(interD), nnodes - 1))


        # Convert vector inputs to arrays and assign class attributes
        self.nnodes = nnodes
        self.nodeD = np.array(nodeD)  # um
        self.nodeL = np.array(nodeL)  # um
        # self.nodeFs = np.array(nodeFs)
        self.interD = np.array(interD)  # um
        self.interL = np.array(interL)  # um
        self.rs = rs  # Ohm.cm
        self.connector = connector
        self.has_vext_mech = False

        # Initialize point-neuron model and delete its single section
        super().__init__(neuron, a=a, Fdrive=Fdrive, verbose=verbose)
        del self.section

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

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return 'SONIC1D ({}, {})'.format(self.strBiophysics(), self.strNodes())
        # 'classic_connect' if self.connector is None else repr(self.connector))

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
            self.mechname, self.strNodes(), self.strResistivity(), self.strGeom())

    def createSections(self, ids):
        ''' Create morphological sections.

            :param id: names of the sections.
        '''
        if self.verbose:
            print('creating sections')
        return list(map(super(Sonic1D, self).createSection, ids))

    def defineGeometry(self):
        ''' Set the geometry of the nodes sections. '''
        if self.verbose:
            print('defining sections geometry: {}'.format(self.strGeom()))
        for i, sec in enumerate(self.sections):
            sec.diam = self.nodeD[i]  # um
            sec.L = self.nodeL[i]  # um
            sec.nseg = 1

    def defineBiophysics(self):
        ''' Set section-specific membrane properties with specific sonophore membrane coverage. '''
        if self.verbose:
            print('defining membrane biophysics: {}'.format(self.strBiophysics()))
        for sec in self.sections:
            sec.insert(self.mechname)
        # for sec, fs in zip(self.sections, self.nodeFs):
        #     sec.insert(self.mechname)
        #     setattr(sec, 'fs_{}'.format(self.mechname), fs)

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


    def setUSdrive(self, amps, config):
        ''' Set section specific acoustic stimulation amplitudes.

            :param amps: model-sized vector of acoustic amplitudes (kPa) or single value
            :return: section-specific labels
        '''
        # Process inputs
        if self.connector is None:
            raise ValueError(
                'attempting to perform A-STIM simulation with standard "v-based" connection scheme')

        # Set acoustic amplitudes
        amps = self.processInputs(amps, config)  # kPa
        if self.verbose:
            print('Setting acoustic amplitudes: Adrive = [{}] kPa'.format(
                ' - '.join('{:.0f}'.format(Adrive) for Adrive in amps)))
        for sec, Adrive in zip(self.sections, amps):
            setattr(sec, 'Adrive_{}'.format(self.mechname), Adrive)
        self.modality = 'US'

        # Return section-specific labels
        return ['node {} ({:.0f} kPa)'.format(i + 1, A) for i, A in enumerate(amps)]


    def setIinj(self, amps, config):
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


    def setVext(self, Vexts):
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


    def setStimON(self, value):
        ''' Set US or electrical stimulation ON or OFF by updating the appropriate
            mechanism/object parameter.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        for sec in self.sections:
            setattr(sec, 'stimon_{}'.format(self.mechname), value)
        if self.modality == 'Iinj':
            for iclamp, Iinj in zip(self.iclamps, self.Iinjs):
                iclamp.amp = value * Iinj
        elif self.modality == 'Vext':
            for sec, Vext in zip(self.sections, self.Vexts):
                sec.e_extracellular = value * Vext
                # print(sec.e_extracellular, sec.vext[0], sec.vext[1])
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
        # states = {
        #     state: [{
        #         suffix: setRangeProbe(sec, '{}_{}_{}'.format(alias(state), suffix, self.mechname))
        #         for suffix in ['0', 'US']
        #     } for sec in self.sections] for state in self.neuron.states_names
        # }

        states = {}
        for key in self.neuron.states_names:
            # print(key, '{}_{}'.format(alias(key), self.mechname))
            states[key] = [setRangeProbe(sec, '{}_{}'.format(alias(key), self.mechname))
                           for sec in self.sections]

        # states = {
        #     state: [
        #         setRangeProbe(sec, '{}_{}'.format(alias(state), self.mechname))
        #         for sec in self.sections]
        #     for state in self.neuron.states_names
        # }


        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Retrieve output variables
        t = Vec2array(tprobe) * 1e-3  # s
        stimon = Vec2array(stimon)
        Qm = np.array(list(map(Vec2array, Qm))) * 1e-5  # C/cm2
        Vmeff = np.array(list(map(Vec2array, Vmeff)))  # mV
        # states = {
        #     key: [
        #         fs * Vec2array(val['US']) + (1 - fs) * Vec2array(val['0'])
        #         for val, fs in zip(states[key], self.nodeFs)
        #     ] for key in self.neuron.states_names
        # }
        states = {key: np.array(list(map(Vec2array, val))) for key, val in states.items()}

        return t, stimon, Qm, Vmeff, states


    def titrateIinjIntra(self, tstim, toffset, PRF, DC, dt, atol, config,
                         inode=0, Irange=(0., IINJ_INTRA_MAX)):
        ''' Use a dichotomic recursive search to determine the threshold intracellular current
            amplitude needed to obtain neural excitation for a given duration, PRF and duty cycle.

            :param tstim: stimulus duration (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param dt: integration time step
            :param atol: integration error tolerance
            :param inode: node at which to detect for spike occurence
            :param Irange: search interval for Iinj, iteratively refined (mA/m2)
            :return: 5-tuple with the determined threshold, time profile,
                 solution matrix, state vector and response latency
        '''
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


    def titrateUS(self, tstim, toffset, PRF, DC, dt, atol, config, inode=0, Arange=(0., US_AMP_MAX)):
        ''' Use a dichotomic recursive search to determine the threshold acoustic pressure
            amplitude needed to obtain neural excitation for a given duration, PRF and duty cycle.

            :param tstim: stimulus duration (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param dt: integration time step
            :param atol: integration error tolerance
            :param inode: node at which to detect for spike occurence
            :param Arange: search interval for Adrive, iteratively refined (kPa)
            :return: 5-tuple with the determined threshold, time profile,
                 solution matrix, state vector and response latency
        '''
        Adrive = (Arange[0] + Arange[1]) / 2
        self.setUSdrive(Adrive, config)

        # Run simulation and detect spikes on ith trace
        t, stimon, Qm, Vmeff, states = self.simulate(tstim, toffset, PRF, DC, dt, atol)
        ipeaks, *_ = findPeaks(Qm[inode], mph=SPIKE_MIN_QAMP, mpp=SPIKE_MIN_QPROM)
        nspikes = ipeaks.size

        # If accurate threshold is found, return simulation results
        if (Arange[1] - Arange[0]) <= DELTA_US_AMP_MIN and nspikes == 1:
            print('threshold amplitude: {}Pa'.format(si_format(Adrive * 1e3, 2, space=' ')))
            return Adrive

        # Otherwise, refine titration interval and iterate recursively
        else:
            if nspikes == 0:
                Arange = (Adrive, Arange[1])
            else:
                Arange = (Arange[0], Adrive)
            return self.titrateUS(
                tstim, toffset, PRF, DC, dt, atol, config, inode=inode, Arange=Arange)


    def titrateIinjExtra(self, z0, tstim, toffset, PRF, DC, dt, atol, inode=0, Irange=None,
                         Itype='cathodal'):
        ''' Use a dichotomic recursive search to determine the threshold amplitude needed
            to obtain neural excitation for a given duration, PRF and duty cycle.

            :param z0: electrode z-coordinate (um)
            :param tstim: stimulus duration (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param dt: integration time step
            :param atol: integration error tolerance
            :param inode: node at which to detect for spike occurence
            :param Irange: search interval for Iinj, iteratively refined (uA)
            :param Itype: stimulation type ("cathodal" or "anodal")
            :return: 5-tuple with the determined threshold, time profile,
                 solution matrix, state vector and response latency
        '''
        # Determine Irange from stimulation type (cathodic or anodic)
        if Irange is None:
            if Itype == 'cathodal':
                Irange = (-IINJ_EXTRA_CATHODAL_MAX, 0.)
            elif Itype == 'anodal':
                Irange = (0., IINJ_EXTRA_ANODAL_MAX)

        # Update Iinj and resulting Vexts
        Iinj = (Irange[0] + Irange[1]) / 2
        self.setVext(computeVext(self, Iinj, z0))

        # Run simulation and detect spikes on ith trace
        t, stimon, Qm, Vmeff, states = self.simulate(tstim, toffset, PRF, DC, dt, atol)
        ipeaks, *_ = findPeaks(Vmeff[inode], mph=SPIKE_MIN_VAMP, mpp=SPIKE_MIN_VPROM)
        nspikes = ipeaks.size

        # If accurate threshold is found, return simulation results
        if (Irange[1] - Irange[0]) <= DELTA_IINJ_EXTRA_MIN and nspikes == 1:
            print('threshold amplitude: {}A'.format(si_format(Iinj * 1e-6, 2, space=' ')))
            return Iinj

        # Otherwise, refine titration interval and iterate recursively
        else:
            if nspikes == 0:
                Irange = (Iinj, Irange[1])
            else:
                Irange = (Irange[0], Iinj)
            return self.titrateIinjExtra(
                z0, tstim, toffset, PRF, DC, dt, atol, inode=inode, Irange=Irange, Itype=Itype)


def computeVext(fiber, I, z0=None, x0=0):
    ''' Compute the extracellular electric potential at a fiber's nodes of Ranvier generated by
        a point-current source at a given distance from the fiber in a homogenous, isotropic medium.

        :param fiber: Sonic1D fiber model object
        :param I: stimulation current amplitude (uA)
        :param z0: electrode coordinate along z-axis (perpendicular to axon), in um
        :param x0: electrode coordinate along x-axis (parallel to axon), in um
        :return: computed extracellular potential(s) (mV)
    '''

    # if no z-coordinate is provided, place the electrode at one internodal distance from the fiber
    if z0 is None:
        z0 = fiber.interL[0]

    # Get x-coordinates of fiber nodes (centered at x=0), compute node-electrode distances and
    # return induced extracellular potentials
    xnodes = fiber.nodeCoordinates()
    distances = np.sqrt((xnodes - x0)**2 + z0**2)
    return VextPointSource(I, distances)
