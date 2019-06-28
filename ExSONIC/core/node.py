# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 17:26:47
# @Author: Theo Lemaire
# @Date:   2018-08-27 09:23:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-07 13:59:37

import pickle
import abc
import numpy as np
import pandas as pd
from neuron import h

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import si_format, timer, logger

from .pyhoc import *
from ..utils import getNmodlDir
from ..constants import *


class Node(metaclass=abc.ABCMeta):
    ''' Generic node interface. '''

    @property
    @abc.abstractmethod
    def modality(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    def __init__(self, pneuron):
        ''' Initialization.

            :param pneuron: point-neuron model
        '''
        # Initialize arguments
        self.pneuron = pneuron
        logger.debug('Creating {} model'.format(self))

        # Load mechanisms and set function tables of appropriate membrane mechanism
        load_mechanisms(getNmodlDir(), self.pneuron.name)
        self.setFuncTables()

        # Create section and set membrane mechanism
        self.section = self.createSection('node0')
        self.section.insert(self.pneuron.name)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pneuron)

    def strBiophysics(self):
        return '{} neuron'.format(self.pneuron.name)

    def createSection(self, id):
        ''' Create morphological section.

            :param id: name of the section.
        '''
        return h.Section(name=id, cell=self)

    def getLookup(self):
        lkp = self.pneuron.getLookup()
        lkp.refs['A'] = np.array([0.])
        for k, v in lkp.items():
            lkp[k] = np.array([v])
        return lkp

    def setFuncTables(self):
        ''' Set neuron-specific interpolation tables along the charge dimension,
            and link them to FUNCTION_TABLEs in the MOD file of the corresponding
            membrane mechanism.
        '''
        logger.debug('loading %s membrane dynamics lookup tables', self.pneuron.name)

        # Get Lookup
        lkp = self.getLookup()

        # Convert lookups independent variables to hoc vectors
        self.Aref = h.Vector(lkp.refs['A'] * 1e-3)  # kPa
        self.Qref = h.Vector(lkp.refs['Q'] * 1e5)   # nC/cm2

        # Convert lookup tables to hoc matrices
        # !!! hoc lookup dictionary must be a member of the class,
        # otherwise the assignment below does not work properly !!!
        self.lkp = {'V': array_to_matrix(lkp['V'])}  # mV
        for rate in self.pneuron.rates:
            self.lkp[rate] = array_to_matrix(lkp[rate] * 1e-3)  # ms-1

        # Assign hoc matrices to 2D interpolation tables in membrane mechanism
        for k, v in self.lkp.items():
            setFuncTable(self.pneuron.name, k, v, self.Aref, self.Qref)

    def printStimAmp(self, value):
        logger.debug('Stimulus amplitude: {} = {}{}'.format(
            self.modality['name'],
            si_format(value * self.modality['factor'], space=' ', precision=2),
            self.modality['unit']))

    @property
    @abc.abstractmethod
    def setStimAmp(self, value):
        raise NotImplementedError

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        setattr(self.section, 'stimon_{}'.format(self.pneuron.name), value)
        return value

    def toggleStim(self):
        ''' Toggle stimulation and set appropriate next toggle event. '''
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
        h.finitialize(self.pneuron.Qm0 * 1e5)  # nC/cm2
        self.stimon = self.setStimON(1)
        self.cvode.event(self.Ton, self.toggleStim)

        # Integrate
        while h.t < tstop:
            h.fadvance()

        # Set absolute error tolerance back to default value if changed
        if atol is not None:
            self.cvode.atol(def_atol)

        return 0

    @timer
    def simulate(self, A, tstim, toffset, PRF, DC, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param A: stimulus amplitude (in modality units)
            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
        '''

        logger.info(
            '%s: simulation @ %s = %s%s, t = %ss (%ss offset)%s',
            self, self.modality['name'],
            si_format(A * self.modality['factor'], space=' ', precision=2), self.modality['unit'],
            *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(
                si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

        # Set recording vectors
        t = setTimeProbe()
        stim = setStimProbe(self.section, self.pneuron.name)
        Qm = setRangeProbe(self.section, 'v')
        Vm = setRangeProbe(self.section, 'Vm_{}'.format(self.pneuron.name))
        states = {k: setRangeProbe(self.section, '{}_{}'.format(alias(k), self.pneuron.name))
                  for k in self.pneuron.statesNames()}

        # Set stimulus amplitude
        self.setStimAmp(A)

        # Integrate model
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Store output in dataframe
        data = pd.DataFrame({
            't': vec_to_array(t) * 1e-3,  # s
            'stimstate': vec_to_array(stim),
            'Qm': vec_to_array(Qm) * 1e-5,  # C/cm2
            'Vm': vec_to_array(Vm)         # mV
        })
        for k, v in states.items():
            data[k] = vec_to_array(v)

        # Return dataframe
        return data

    def titrate(self, tstim, toffset, PRF, DC, dt, atol, Arange=None):
        ''' Use a binary search to determine the threshold excitation amplitude.

            :param tstim: stimulus duration (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param dt: integration time step
            :param atol: integration error tolerance
            :param Arange: amplitude search interval
            :return: threshold excitation amplitude (kPa)
        '''

        # Determine amplitude interval if needed
        if Arange is None:
            Arange = (0, self.Aref.max())  # kPa
        A = (Arange[0] + Arange[1]) / 2  # kPa

        # Run simulation and detect spikes on ith trace
        t, y, stimon = self.simulate(A, tstim, toffset, PRF, DC, dt, atol)
        Qm = y[0, :]
        ipeaks, *_ = findPeaks(Qm, mph=SPIKE_MIN_QAMP, mpp=SPIKE_MIN_QPROM)
        nspikes = ipeaks.size

        # If accurate threshold is found, return simulation results
        if (Arange[1] - Arange[0]) <= DELTA_US_AMP_MIN and nspikes == 1:
            print('threshold amplitude: {}Pa'.format(si_format(A * 1e3, 2, space=' ')))
            return A

        # Otherwise, refine titration interval and iterate recursively
        else:
            if nspikes == 0:
                # if Adrive too close to max then stop
                if (self.Aref.max() - A) <= DELTA_US_AMP_MIN:
                    print('no threshold amplitude found within (0-{:.0f}) kPa search interval'.format(
                        self.Aref.max()))
                    return np.nan
                Arange = (A, Arange[1])
            else:
                Arange = (Arange[0], A)
            return self.titrate(tstim, toffset, PRF, DC, dt, atol, Arange=Arange)

    @property
    @abc.abstractmethod
    def filecode(self, *args):
        raise NotImplementedError

    def runAndSave(self, outdir, *args):
        ''' Simulate the model for specific parameters and save the results
            in a specific output directory. '''
        args = self.pneuron.checkAmplitude(args)
        data, tcomp = self.simulate(*args)
        meta = self.pneuron.meta(*args)
        meta['tcomp'] = tcomp
        fpath = '{}/{}.pkl'.format(outdir, self.filecode(*args))
        with open(fpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', fpath)
        return fpath


class IintraNode(Node):
    ''' Node used for simulations with intracellular current. '''

    modality = {
        'name': 'I_intra',
        'unit': 'A/m2',
        'factor': 1e-3
    }

    def setStimAmp(self, Astim):
        ''' Set electrical stimulation amplitude

            :param Astim: injected current density (mA/m2).
        '''
        self.printStimAmp(Astim)
        self.Iinj = Astim * self.section(0.5).area() * 1e-6  # nA
        self.iclamp = h.IClamp(self.section(0.5))
        self.iclamp.delay = 0  # we want to exert control over amp starting at 0 ms
        self.iclamp.dur = 1e9  # dur must be long enough to span all our changes

    def setStimON(self, value):
        value = super().setStimON(value)
        self.iclamp.amp = value * self.Iinj
        return value

    def filecode(self, *args):
        return self.pneuron.filecode(*args) + '_NEURON'


class VextNode(Node):
    ''' Node used for simulations with extracellular potential. '''

    modality = {
        'name': 'V_ext',
        'unit': 'V',
        'factor': 1e-3
    }

    def setStimAmp(self, Vext):
        ''' Insert extracellular mechanism into section and set extracellular potential value.

            :param Vext: extracellular potential (mV).
        '''
        self.printStimAmp(Vext)
        insertVext(self.section)
        self.Vext = Vext

    def setStimON(self, value):
        value = super().setStimON(value)
        self.section.e_extracellular = value * self.Vext
        return value


class SonicNode(Node):
    ''' Node used for simulations with US stimulus. '''

    modality = {
        'name': 'A_US',
        'unit': 'Pa',
        'factor': 1e3
    }

    def __init__(self, pneuron, a=32., Fdrive=500., fs=1.):
        ''' Initialization.

            :param pneuron: point-neuron model
            :param a: sonophore diameter (nm)
            :param Fdrive: ultrasound frequency (kHz)
            :param fs: sonophore membrane coverage fraction (-)
        '''

        if fs > 1. or fs < 0.:
            raise ValueError('fs ({}) must be within [0-1]'.format(fs))
        self.nbls = NeuronalBilayerSonophore(a * 1e-9, pneuron, Fdrive * 1e3)
        self.a = a
        self.fs = fs
        self.Fdrive = Fdrive
        super().__init__(pneuron)

    def __repr__(self):
        return '{}({:.1f} nm, {}, {:.0f} kHz)'.format(
            self.__class__.__name__, self.a, self.pneuron, self.Fdrive)

    def strBiophysics(self):
        return super().strBiophysics() + ', a = {}m{}, f = {}Hz'.format(
            si_format(self.a * 1e-9, space=' '),
            ', fs = {:.0f}%'.format(self.fs * 1e2) if self.fs is not None else '',
            si_format(self.Fdrive * 1e3, space=' '))

    def getLookup(self):
        return self.nbls.getLookup2D(self.Fdrive * 1e3, self.fs)

    def setStimAmp(self, Adrive):
        ''' Set US stimulation amplitude (and set modality to "US").

            :param Adrive: acoustic pressure amplitude (kPa)
        '''
        self.printStimAmp(Adrive)
        setattr(self.section, 'Adrive_{}'.format(self.pneuron.name), Adrive)

    def filecode(self, *args):
        return self.nbls.filecode(self.Fdrive * 1e3, *args, self.fs * 1e2, 'NEURON')