# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 09:23:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-05 01:12:07

import pickle
import abc
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from neuron import h

from PySONIC.constants import *
from PySONIC.core import Model, PointNeuron, NeuronalBilayerSonophore
from PySONIC.utils import si_format, timer, logger, binarySearch, plural, debug

from .pyhoc import *
# from ..utils import getNmodlDir
from ..constants import *


class Node(metaclass=abc.ABCMeta):
    ''' Generic node interface. '''

    @property
    @abc.abstractmethod
    def modality(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    def __init__(self, pneuron, id=None, auto_nmodl=False):
        ''' Initialization.

            :param pneuron: point-neuron model
        '''
        # Initialize arguments
        self.pneuron = pneuron
        if id is None:
            id = self.__repr__()
        self.id = id
        logger.debug('Creating {} model'.format(self))

        # Load mechanisms and set function tables of appropriate membrane mechanism
        self.auto_nmodl = auto_nmodl
        self.mechname = self.pneuron.name
        if self.auto_nmodl:
            self.mechname += 'auto'
        load_mechanisms(self.getNmodlDir(), self.mechname)
        self.setFuncTables()

        # Create section and set membrane mechanism
        self.section = self.createSection(self.id)
        self.section.insert(self.mechname)

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

    def getNmodlDir(self):
        ''' Return path to directory containing MOD files and compiled mechanisms files. '''
        selfdir = os.path.dirname(os.path.realpath(__file__))
        pardir = os.path.abspath(os.path.join(selfdir, os.pardir))
        if self.auto_nmodl:
            return os.path.join(pardir, 'auto_nmodl')
        else:
            return os.path.join(pardir, 'nmodl')

    def setFuncTables(self):
        ''' Set neuron-specific interpolation tables along the charge dimension,
            and link them to FUNCTION_TABLEs in the MOD file of the corresponding
            membrane mechanism.
        '''
        logger.debug('loading %s membrane dynamics lookup tables', self.mechname)

        # Get Lookup
        lkp = self.getLookup()

        # Convert lookups independent variables to hoc vectors
        self.Aref = h.Vector(lkp.refs['A'] * 1e-3)  # kPa
        self.Qref = h.Vector(lkp.refs['Q'] * 1e5)   # nC/cm2

        # Convert lookup tables to hoc matrices
        # !!! hoc lookup dictionary must be a member of the class,
        # otherwise the assignment below does not work properly !!!
        self.lkp = {'V': array_to_matrix(lkp['V'])}  # mV
        for ratex in self.pneuron.alphax_list.union(self.pneuron.betax_list):
            self.lkp[ratex] = array_to_matrix(lkp[ratex] * 1e-3)  # ms-1
        for taux in self.pneuron.taux_list:
            self.lkp[taux] = array_to_matrix(lkp[taux] * 1e3)  # ms
        for xinf in self.pneuron.xinf_list:
            self.lkp[xinf] = array_to_matrix(lkp[xinf])  # (-)

        # Assign hoc matrices to 2D interpolation tables in membrane mechanism
        for k, v in self.lkp.items():
            setFuncTable(self.mechname, k, v, self.Aref, self.Qref)

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
        setattr(self.section, 'stimon_{}'.format(self.mechname), value)
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
            logger.debug('fixed time step integration (dt = {} ms)'.format(h.dt))
        else:
            self.cvode.active(1)
            if atol is not None:
                def_atol = self.cvode.atol()
                self.cvode.atol(atol)
                logger.debug('adaptive time step integration (atol = {})'.format(self.cvode.atol()))

        # Initialize
        self.stimon = self.setStimON(0)
        print(f'A = {getattr(self.section(0.5), self.mechname).Adrive} kPa')
        print('finitialize')
        h.finitialize(self.pneuron.Qm0() * 1e5)  # nC/cm2
        self.stimon = self.setStimON(1)
        self.cvode.event(self.Ton, self.toggleStim)

        # Integrate
        i = 0
        while h.t < tstop:
            # if i < 5:
            #     print(f't = {h.t} ms')
            #     print(f'Qm = {self.section(0.5).v} nC/cm2')
            #     print(f'Vmeff = {getattr(self.section(0.5), self.mechname).Vm} mV')
            #     print(f'iCaT = {getattr(self.section(0.5), self.mechname).iCaT} mA/cm2')
            #     print('')
            h.fadvance()
            i += 1

        # Set absolute error tolerance back to default value if changed
        if atol is not None:
            self.cvode.atol(def_atol)

        return 0

    def setProbes(self):
        ''' Set recording vectors. '''
        t = setTimeProbe()
        stim = setStimProbe(self.section, self.mechname)
        Qm = setRangeProbe(self.section, 'v')
        Vm = setRangeProbe(self.section, 'Vm_{}'.format(self.mechname))
        states = {k: setRangeProbe(self.section, '{}_{}'.format(alias(k), self.mechname))
                  for k in self.pneuron.statesNames()}
        return t, stim, Qm, Vm, states

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    @Model.logNSpikes
    @Model.checkTitrate('A')
    @Model.addMeta
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
        t, stim, Qm, Vm, states = self.setProbes()

        # Set stimulus amplitude and integrate model
        self.setStimAmp(A)
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

        # Resample data to regular sampling rate
        return self.resample(data, DT_EFFECTIVE)

    def meta(self, A, tstim, toffset, PRF, DC):
        return self.pneuron.meta(A, tstim, toffset, PRF, DC)

    @staticmethod
    def resample(data, dt):
        ''' Resample dataframe at regular time step. '''
        t = data['t'].values
        n = int(np.ptp(t) / dt) + 1
        tnew = np.linspace(t.min(), t.max(), n)
        new_data = {}
        for key in data:
            kind = 'nearest' if key == 'stimstate' else 'linear'
            new_data[key] = interp1d(t, data[key].values, kind=kind)(tnew)
        return pd.DataFrame(new_data)

    def titrate(self, tstim, toffset, PRF=100., DC=1., xfunc=None):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given duration, PRF and duty cycle.

            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param method: integration method
            :param xfunc: function determining whether condition is reached from simulation output
            :return: determined threshold amplitude (Pa)
        '''
        # Default output function
        if xfunc is None:
            xfunc = self.pneuron.titrationFunc

        return binarySearch(
            lambda x: xfunc(self.simulate(*x)[0]),
            [tstim, toffset, PRF, DC], 0, self.Arange, self.A_conv_thr)

    @property
    @abc.abstractmethod
    def filecode(self, *args):
        raise NotImplementedError

    def simAndSave(self, outdir, *args):
        ''' Simulate the model and save the results in a specific output directory. '''
        data, meta = self.simulate(*args)
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
    Arange = (0., 2 * AMP_UPPER_BOUND_ESTIM)
    A_conv_thr = THRESHOLD_CONV_RANGE_ESTIM



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
    Arange = None
    A_conv_thr = None

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

    def filecode(self, *args):
        return 'Vext_' + self.pneuron.filecode(*args) + '_NEURON'


class SonicNode(Node):
    ''' Node used for simulations with US stimulus. '''

    modality = {
        'name': 'A_US',
        'unit': 'Pa',
        'factor': 1e0
    }
    A_conv_thr = THRESHOLD_CONV_RANGE_ASTIM

    def __init__(self, pneuron, id=None, a=32e-9, Fdrive=500e3, fs=1., auto_nmodl=False):
        ''' Initialization.

            :param pneuron: point-neuron model
            :param a: sonophore diameter (m)
            :param Fdrive: ultrasound frequency (Hz)
            :param fs: sonophore membrane coverage fraction (-)
        '''
        if fs > 1. or fs < 0.:
            raise ValueError('fs ({}) must be within [0-1]'.format(fs))
        self.nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)
        self.a = a
        self.fs = fs
        self.Fdrive = Fdrive
        super().__init__(pneuron, id=id, auto_nmodl=auto_nmodl)
        self.Arange = (0., self.getLookup().refs['A'].max())

    def __repr__(self):
        return '{}({:.1f} nm, {}, {:.0f} kHz)'.format(
            self.__class__.__name__, self.a * 1e9, self.pneuron, self.Fdrive * 1e-3)

    def strBiophysics(self):
        return super().strBiophysics() + ', a = {}m{}, f = {}Hz'.format(
            si_format(self.a, space=' '),
            ', fs = {:.0f}%'.format(self.fs * 1e2) if self.fs is not None else '',
            si_format(self.Fdrive, space=' '))

    def getLookup(self):
        return self.nbls.getLookup2D(self.Fdrive, self.fs)

    def setStimAmp(self, Adrive):
        ''' Set US stimulation amplitude.

            :param Adrive: acoustic pressure amplitude (Pa)
        '''
        self.printStimAmp(Adrive)
        setattr(self.section, 'Adrive_{}'.format(self.mechname), Adrive * 1e-3)

    def filecode(self, *args):
        return self.nbls.filecode(self.Fdrive, *args, self.fs, 'NEURON')

    def meta(self, A, tstim, toffset, PRF, DC):
        return self.nbls.meta(self.Fdrive, A, tstim, toffset, PRF, DC, self.fs, 'NEURON')
