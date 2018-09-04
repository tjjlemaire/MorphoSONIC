# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-27 09:23:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-03 16:02:31


import os
import numpy as np
from neuron import h

from PySONIC.neurons import *
from PySONIC.utils import getLookups2D, si_format

from ..pyhoc import *
from ..utils import getNmodlDir



class Sonic0D:
    ''' Point-neuron SONIC model in NEURON. '''

    def __init__(self, neuron, a=32e-9, Fdrive=500e3, verbose=False):
        ''' Initialization.

            :param neuron: neuron object
            :param a: sonophore diameter (nm)
            :param Fdrive: ultrasound frequency (Hz)
            :param verbose: boolean stating whether to print out details
        '''
        # Initialize arguments
        self.neuron = neuron
        self.a = a  # m
        self.Fdrive = Fdrive  # Hz
        self.verbose = verbose
        self.mechname = neuron.name

        # Load mechanisms and set function tables of appropriate membrane mechanism
        self.loadMechanisms()
        self.setFuncTables(self.a, self.Fdrive)

        # Create section and set membrane mechanism
        self.section = self.createSection('point')
        self.section.insert(self.mechname)

        if self.verbose:
            print('Creating model: {}'.format(self))

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return 'SONIC0D_{}_{}m_{}Hz'.format(self.neuron.name, *si_format([self.a, self.Fdrive], 2))

    def pprint(self):
        ''' Pretty-print naming of the model instance. '''
        return '{} point-neuron, a = {}m, f = {}Hz'.format(
            self.neuron.name, *si_format([self.a, self.Fdrive], space=' '))

    def loadMechanisms(self):
        ''' Locate NMODL directory, check for untracked modifications in NMODL source files, and
            load compiled mechanisms file.
        '''
        nmodl_dir = getNmodlDir()
        mod_file = os.path.join(nmodl_dir, '{}.mod'.format(self.mechname))
        dll_file = os.path.join(nmodl_dir, 'nrnmech.dll')
        if not os.path.isfile(dll_file) or os.path.getmtime(mod_file) > os.path.getmtime(dll_file):
            raise Warning('"{}.mod" file more recent than compiled dll'.format(self.mechname))
        if not isAlreadyLoaded(dll_file):
            h.nrn_load_dll(dll_file)

    def createSection(self, id):
        ''' Create morphological section.

            :param id: name of the section.
        '''
        return h.Section(name=id, cell=self)

    def setFuncTables(self, a, Fdrive):
        ''' Set neuron-specific, sonophore diameter and US frequency dependent 2D interpolation tables
            in the (amplitude, charge) space, and link them to FUNCTION_TABLEs in the MOD file of the
            corresponding membrane mechanism.

            :param a: sonophore diameter (m)
            :param Fdrive: US frequency (Hz)
        '''

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

    def setAdrive(self, Adrive):
        ''' Set US stimulation amplitude (and set modality to "US").

            :param Adrive: acoustic pressure amplitude (Pa)
        '''
        if self.verbose:
            print('Setting acoustic stimulus amplitude: Adrive = {}Pa'
                  .format(si_format(Adrive * 1e3, space=' ')))
        setattr(self.section, 'Adrive_{}'.format(self.mechname), Adrive)
        self.modality = 'US'

    def setAstim(self, Astim):
        ''' Set electrical stimulation amplitude (and set modality to "elec")

            :param Astim: injected current density (mA/m2).
        '''
        self.Iinj = Astim * self.section(0.5).area() * 1e-6  # nA
        if self.verbose:
            print('Setting electrical stimulus amplitude: Iinj = {}A'
                  .format(si_format(self.Iinj * 1e-9, 2, space=' ')))
        self.iclamp = h.IClamp(self.section(0.5))
        self.iclamp.delay = 0  # we want to exert control over amp starting at 0 ms
        self.iclamp.dur = 1e9  # dur must be long enough to span all our changes
        self.modality = 'elec'

    def setStimON(self, value):
        ''' Set US or electrical stimulation ON or OFF by updating the appropriate
            mechanism/object parameter.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        setattr(self.section, 'stimon_{}'.format(self.mechname), value)
        if self.modality == 'elec':
            self.iclamp.amp = value * self.Iinj
        return value

    def toggleStim(self):
        ''' Toggle US or electrical stimulation and set appropriate next toggle event. '''
        # OFF -> ON at pulse onset
        if self.stimon == 0:
            # print('t = {:.2f} ms: switching stim ON and setting next OFF event at {:.2f} ms'
            #       .format(h.t, min(self.tstim, h.t + self.Ton)))
            self.stimon = self.setStimON(1)
            self.cvode.event(min(self.tstim, h.t + self.Ton), self.toggleStim)
        # ON -> OFF at pulse offset
        else:
            self.stimon = self.setStimON(0)
            if (h.t + self.Toff) < self.tstim - h.dt:
                # print('t = {:.2f} ms: switching stim OFF and setting next ON event at {:.2f} ms'
                #       .format(h.t, h.t + self.Toff))
                self.cvode.event(h.t + self.Toff, self.toggleStim)
            # else:
            #     print('t = {:.2f} ms: switching stim OFF'.format(h.t))

        # Re-initialize cvode if active
        if self.cvode.active():
            self.cvode.re_init()
        else:
            h.fcurrent()

    def integrate(self, tstop, tstim, PRF, DC, dt, atol):
        ''' Integrate the model differential variables for a given duration, while updating the
            value of the boolean parameter stimon during ON and OFF periods throughout the numerical
            integration, according to stimulus parameters.

            Integration uses an adaptive time step method by default.

            :param tstop: duration of numerical integration (s)
            :param tstim: stimulus duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step (s). If provided, the fixed time step method is used.
            :param atol: absolute error tolerance (default = 1e-3). If provided, the adaptive
                time step method is used.
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

        # Set pulsing parameters used in CVODE events
        self.Ton = DC / PRF
        self.Toff = (1 - DC) / PRF
        self.tstim = tstim

        # Set integration parameters
        h.secondorder = 2
        self.cvode = h.CVode()
        if dt is not None:
            h.dt = dt
            self.cvode.active(0)
            print('fixed time step integration (dt = {} ms)'.format(h.dt))
        else:
            self.cvode.active(1)
            if atol is not None:
                def_atol = self.cvode.atol()
                self.cvode.atol(atol)
                print('adaptive time step integration (atol = {})'.format(self.cvode.atol()))

        # Initialize
        h.finitialize(self.neuron.Vm0)
        self.stimon = self.setStimON(1)
        self.cvode.event(self.Ton, self.toggleStim)

        # Integrate
        while h.t < tstop:
            h.fadvance()

        # Set absolute error tolerance back to default value if changed
        if atol is not None:
            self.cvode.atol(def_atol)

        return 0

    def simulate(self, tstim, toffset, PRF, DC, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
        '''
        # Set recording vectors
        tprobe = setTimeProbe()
        stimprobe = setStimProbe(self.section, self.mechname)
        vprobe = setRangeProbe(self.section, 'v')
        Vmeffprobe = setRangeProbe(self.section, 'Vmeff_{}'.format(self.mechname))
        statesprobes = [setRangeProbe(self.section, '{}_{}'.format(alias(state), self.mechname))
                        for state in self.neuron.states_names]

        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Retrieve output variables
        t = Vec2array(tprobe) * 1e-3  # s
        stimon = Vec2array(stimprobe)
        Qprobe = Vec2array(vprobe) * 1e-5  # nC/cm2
        Vmeffprobe = Vec2array(Vmeffprobe)  # mV
        statesprobes = list(map(Vec2array, statesprobes))
        y = np.vstack([Qprobe, Vmeffprobe, np.array(statesprobes)])

        # return output variables
        return (t, y, stimon)
