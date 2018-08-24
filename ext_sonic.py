# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-15 15:08:23
# @Last Modified by:   Theo
# @Last Modified time: 2018-08-22 01:39:03

''' NEURON utilities '''

import os
import numpy as np
from neuron import h

from PySONIC.utils import getLookups2D, InputError, si_format
from hoc_utils import *


class SeriesConnector:
    ''' The SeriesConnector class allows to connect model sections in series through a
        by inserting axial current as a distributed membrane mechanism in those sections, thereby
        allowing to use any voltage variable (not necessarily 'v') as a reference to compute
        axial currents.

        :param mechname: name of the mechanism that compute axial current contribution
        :param vref: name of the reference voltage varibale to compute axial currents
        :param rmin (optional): lower bound for axial resistance density assigned to sections (Ohm.cm2)
        :param verbose: boolean indicating whether to output details of the connection process
    '''

    def __init__(self, mechname='Iax', vref='v', rmin=1e2, verbose=False):
        self.mechname = mechname
        self.vref = vref
        self.rmin = rmin  # Ohm.cm2
        self.verbose = verbose

    def __str__(self):
        return 'Series connector object: {} density mechanism, reference voltage variable = "{}"{}'\
            .format(self.mechname, self.vref,
                    ', minimal resistance density = {:.2e} Ohm.cm2'.format(self.rmin)
                    if self.rmin is not None else '')

    def membraneArea(self, sec):
        ''' Compute section membrane surface area (in cm2) '''
        return np.pi * (sec.diam * 1e-4) * (sec.L * 1e-4)

    def axialArea(self, sec):
        ''' Compute section axial area (in cm2) '''
        return np.pi * (sec.diam * 1e-4)**2 / 4

    def resistance(self, sec):
        ''' Compute section axial resistance (in Ohm) '''
        return sec.Ra * (sec.L * 1e-4) / self.axialArea(sec)

    def attach(self, sec):
        ''' Insert density mechanism to section and set appropriate axial conduction parameters. '''

        # Insert axial current density mechanism
        sec.insert(self.mechname)

        # Compute section properties
        Am = self.membraneArea(sec)  # membrane surface area (cm2)
        R = self.resistance(sec)  # section resistance (Ohm)

        # Optional: bound resistance to ensure (resistance * membrane area) is always above threshold,
        # in order to limit the mangnitude of axial currents and thus ensure convergence of simulation
        if self.rmin is not None:
            if R * Am < self.rmin:
                if self.verbose:
                    print('R*Am = {:.2e} Ohm.cm2 -> bounded to {:.2e} Ohm.cm2'
                          .format(R * Am, self.rmin))
            R = max(R, self.rmin / Am)

        # Set section propeties to Iax mechanism
        setattr(sec, 'R_{}'.format(self.mechname), R)
        setattr(sec, 'Am_{}'.format(self.mechname), Am)
        h.setpointer(getattr(sec(0.5), '_ref_{}'.format(self.vref)), 'V',
                     getattr(sec(0.5), self.mechname))

        # While section not connected: set neighboring sections' properties (resistance and
        # membrane potential) as those of current section
        for suffix in ['prev', 'next']:
            setattr(sec, 'R{}_{}'.format(suffix, self.mechname), R)  # Ohm
            h.setpointer(getattr(sec(0.5), '_ref_{}'.format(self.vref)),
                         'V{}'.format(suffix), getattr(sec(0.5), self.mechname))

        return sec

    def connect(self, sec1, sec2):
        ''' Connect two adjacent sections in series to enable trans-sectional axial current. '''

        # Inform sections about each other's axial resistance (in Ohm)
        setattr(sec1, 'Rnext_{}'.format(self.mechname), getattr(sec2, 'R_{}'.format(self.mechname)))
        setattr(sec2, 'Rprev_{}'.format(self.mechname), getattr(sec1, 'R_{}'.format(self.mechname)))

        # Set bi-directional pointers to sections about each other's membrane potential
        h.setpointer(getattr(sec1(0.5), '_ref_{}'.format(self.vref)),
                     'Vprev', getattr(sec2(0.5), self.mechname))
        h.setpointer(getattr(sec2(0.5), '_ref_{}'.format(self.vref)),
                     'Vnext', getattr(sec1(0.5), self.mechname))



class ExtendedSONIC:

    default_Fdrive = 500e3  # Hz

    def __init__(self, neuron, nsec=1, diam=1, L=1, Ra=1e2, series_connector=None,
                 a=32e-9, Fdrive=None):
        self.nsec = nsec
        self.neuron = neuron
        self.Ra = Ra
        self.L = L * 1e6  # um
        self.diam = diam * 1e6  # um
        self.a = a  # m
        self.mechname = neuron.name
        self.Fdrive = Fdrive

        # Load mechanisms DLL file
        nmodl_dir = os.path.join(os.getcwd(), 'nmodl')
        mod_file = nmodl_dir + '/{}.mod'.format(self.mechname)
        dll_file = nmodl_dir + '/nrnmech.dll'
        if not os.path.isfile(dll_file) or os.path.getmtime(mod_file) > os.path.getmtime(dll_file):
            raise Warning('"{}.mod" file more recent than compiled dll'.format(self.mechname))
        if not isAlreadyLoaded(dll_file):
            h.nrn_load_dll(dll_file)

        # Create sections and set their geometry
        self.createSections()
        self.defineGeometry()  # needs to be done PRIOR to build_custom_topology()

        # Define sections membrane mechanisms
        if Fdrive is None:
            # if Fdrive not defined -> default lookups used with Adrive = 0
            self.defineBiophysics(self.a, self.default_Fdrive)
            self.setUSAmps(np.zeros(nsec))
        else:
            self.defineBiophysics(self.a, Fdrive)

        # Connect section together
        if self.nsec > 1:
            if series_connector is None:
                self.buildTopology()
            else:
                self.buildCustomTopology(series_connector)

        # Set stimulation boolean value to all model sections
        self.setStimBool(0)


    def __str__(self):
        return 'NBLS_{}_{}node{}'.format(self.neuron.name, self.nsec, 's' if self.nsec > 1 else '')

    def details(self):
        return 'diam = {}m, L = {}m, Ra = {:.0e} ohm.cm'\
            .format(*si_format([self.diam * 1e-6, self.L * 1e-6], space=' '), self.Ra)

    def createSections(self):
        ''' Create morphological sections. '''
        self.sections = [h.Section(name='node{}'.format(i), cell=self) for i in range(self.nsec)]

    def defineGeometry(self):
        ''' Set the 3D geometry of the model. '''
        for sec in self.sections:
            sec.diam = self.diam  # um
            sec.L = self.L  # um
            sec.nseg = 1

    def defineBiophysics(self, a, Fdrive):
        ''' Assign the membrane properties across the model.

            :param a: sonophore diameter (m)
            :param Fdrive: US frequency (Hz)
        '''

        # Set axial resistance of all sections
        for sec in self.sections:
            sec.Ra = self.Ra  # Ohm*cm

        # Get lookups
        Aref, Qref, lookups2D = getLookups2D(self.mechname, a, Fdrive)

        # Rescale rate constants to ms-1
        for k in lookups2D.keys():
            if 'alpha' in k or 'beta' in k:
                lookups2D[k] *= 1e-3

        # Convert lookups independent variables to hoc vectors
        self.Aref = h.Vector(Aref * 1e-3)  # kPa
        self.Qref = h.Vector(Qref * 1e5)  # nC/cm2

        # Convert lookups dependent variables to hoc matrices
        self.lookups2D = {key: array2Matrix(value) for key, value in lookups2D.items()}

        # Assign hoc lookups to as interpolation tables in membrane mechanism
        setFuncTable(self.mechname, 'V', self.lookups2D['V'], self.Aref, self.Qref)
        for gate in self.neuron.getGates():
            gate = gate.lower()
            for rate in ['alpha', 'beta']:
                rname = '{}{}'.format(rate, gate)
                setFuncTable(self.mechname, rname, self.lookups2D[rname], self.Aref, self.Qref)

        # Insert active membrane mechanism in all sections
        for sec in self.sections:
            sec.insert(self.mechname)

    def buildTopology(self):
        ''' Connect the sections in series through classic NEURON implementation. '''
        for i in range(self.nsec - 1):
            self.sections[i + 1].connect(self.sections[i], 1, 0)

    def buildCustomTopology(self, series_connector):
        self.sections = [series_connector.attach(sec) for sec in self.sections]
        for i in range(self.nsec - 1):
            series_connector.connect(self.sections[i], self.sections[i + 1])

    def setUSAmps(self, amps):
        ''' Set section specific US stimulation amplitudes '''
        if len(self.sections) != len(amps):
            raise InputError('Amplitude distribution vector does not match number of sections')
        for sec, Adrive in zip(self.sections, amps):
            setattr(sec, 'Adrive_{}'.format(self.mechname), Adrive)

    def setStimBool(self, stimon):
        ''' Set stimulation boolean value to all model sections. '''
        for sec in self.sections:
            setattr(sec, 'stimon_{}'.format(self.mechname), stimon)
        return stimon

    def attachEStims(self, amps, tstim, PRF, DC, loc=0.5):
        ''' Attach section-sepcific electrical stimuli to a cell.

            :param amps: vector injected current densities (mA/m2).
            :param tstim: duration of the stimulus (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param loc: location on the section where the stimulus is placed
            :return: list of iclamp objects
        '''

        # Convert input parameters to NEURON units
        tstim *= 1e3
        PRF /= 1e3

        ''' Set section specific electrical stimuli '''
        if len(self.sections) != len(amps):
            raise InputError('Amplitude distribution vector does not match number of sections')

        self.iclamps = []
        for sec, Astim in zip(self.sections, amps):
            if Astim > 0:
                self.iclamps += attachEStim(sec, Astim, tstim, PRF, DC, loc)


    def integrate(self, tstop, tstim, PRF, DC, dt):
        ''' Integrate the model differential variables for a given duration, while updating the
            value of the boolean parameter stimon during ON and OFF periods throughout the numerical
            integration, according to stimulus parameters.

            :param tstop: duration of numerical integration (s)
            :param tstim: stimulus duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step (s). If not provided, the adaptive CVODE method is used
        '''

        # Convert input parameters to NEURON units
        tstim *= 1e3
        tstop *= 1e3
        PRF /= 1e3
        if dt is not None:
            dt *= 1e3

        # Update PRF for CW stimuli to optimize integration
        if DC == 1.0:
            PRF = 1 / tstim

        # Set integration parameters
        h.secondorder = 2
        cvode = h.CVode()
        if dt is not None:
            h.dt = dt
            cvode.active(0)
        else:
            cvode.active(1)

        # Initialize
        stimon = self.setStimBool(1)
        h.finitialize(self.neuron.Vm0)

        # Integrate during simulus
        while h.t < tstim:
            # Activate stimon at each pulse onset
            if h.t % (1 / PRF) <= (DC / PRF) and stimon == 0:
                stimon = self.setStimBool(1)
                h.fcurrent()

            # Deactivate stimon at each pulse offset
            elif h.t % (1 / PRF) > (DC / PRF) and stimon == 1:
                stimon = self.setStimBool(0)
                h.fcurrent()

            h.fadvance()

        # Deactivate stimon and integrate stimulus offset
        stimon = self.setStimBool(0)
        h.fcurrent()
        while h.t < tstop:
            h.fadvance()
        h.fadvance()

        return 0

    def simulate(self, tstim, toffset, PRF, DC, dt):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step (s)
        '''

        # Set recording vectors
        tprobe = setTimeProbe()
        stimprobe = setStimProbe(self)
        vprobes = setRangeProbes(self.sections, 'v')
        Vmeffprobes = setRangeProbes(self.sections, 'Vmeff_{}'.format(self.mechname))
        statesprobes = [setRangeProbes(self.sections, '{}_{}'.format(alias(state), self.mechname))
                        for state in self.neuron.states_names]

        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt)

        # Retrieve output variables
        t = Vec2array(tprobe)  # ms
        stimstates = Vec2array(stimprobe)
        vprobes = list(map(Vec2array, vprobes))
        Vmeffprobes = list(map(Vec2array, Vmeffprobes))
        statesprobes = {state: list(map(Vec2array, probes))
                        for state, probes in zip(self.neuron.states_names, statesprobes)}

        return t, stimstates, vprobes, Vmeffprobes, statesprobes
