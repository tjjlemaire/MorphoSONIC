# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 09:23:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-20 22:37:17

import pickle
import abc
from inspect import signature
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from neuron import h

from PySONIC.constants import *
from PySONIC.core import Model, PointNeuron, NeuronalBilayerSonophore
from PySONIC.utils import si_format, timer, logger, binarySearch, plural, debug, logCache, filecode, simAndSave
from PySONIC.postpro import prependDataFrame

from .pyhoc import *
from ..utils import getNmodlDir
from ..constants import *


class Node(metaclass=abc.ABCMeta):
    ''' Generic node interface. '''

    tscale = 'ms'  # relevant temporal scale of the model
    titration_var = 'A'  # name of the titration parameter

    @property
    @abc.abstractmethod
    def modality(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    def __init__(self, pneuron, id=None, cell=None, pylkp=None):
        ''' Initialization.

            :param pneuron: point-neuron model
        '''
        self.cell = cell
        if self.cell is None:
            self.cell = self
        self.pylkp = pylkp

        # Initialize arguments
        self.pneuron = pneuron
        if id is None:
            id = self.__repr__()
        self.id = id
        if cell == self:
            logger.debug('Creating {} model'.format(self))

        # Load mechanisms and set function tables of appropriate membrane mechanism
        self.modfile = f'{self.pneuron.name}.mod'
        self.mechname = f'{self.pneuron.name}auto'
        load_mechanisms(getNmodlDir(), self.modfile)
        self.setFuncTables()

        # Create section and set membrane mechanism
        self.section = h.Section(name=id, cell=self.cell)
        self.section.insert(self.mechname)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pneuron)

    def str_biophysics(self):
        return '{} neuron'.format(self.pneuron.name)

    def clear(self):
        del self.section

    def getPyLookup(self):
        pylkp = self.pneuron.getLookup()
        pylkp.refs['A'] = np.array([0.])
        for k, v in pylkp.items():
            pylkp[k] = np.array([v])
        return pylkp

    def getModLookup(self):
        # Get Lookup
        if self.pylkp is None:
            self.pylkp = self.getPyLookup()

        # Convert lookups independent variables to hoc vectors
        self.Aref = h.Vector(self.pylkp.refs['A'] * 1e-3)  # kPa
        self.Qref = h.Vector(self.pylkp.refs['Q'] * 1e5)   # nC/cm2

        # Convert lookup tables to hoc matrices
        # !!! hoc lookup dictionary must be a member of the class,
        # otherwise the assignment below does not work properly !!!
        self.lkp = {'V': array_to_matrix(self.pylkp['V'])}  # mV
        for ratex in self.pneuron.alphax_list.union(self.pneuron.betax_list):
            self.lkp[ratex] = array_to_matrix(self.pylkp[ratex] * 1e-3)  # ms-1
        for taux in self.pneuron.taux_list:
            self.lkp[taux] = array_to_matrix(self.pylkp[taux] * 1e3)  # ms
        for xinf in self.pneuron.xinf_list:
            self.lkp[xinf] = array_to_matrix(self.pylkp[xinf])  # (-)

    def setFuncTables(self):
        ''' Set neuron-specific interpolation tables along the charge dimension,
            and link them to FUNCTION_TABLEs in the MOD file of the corresponding
            membrane mechanism.
        '''
        if self.cell == self:
            logger.debug('loading %s membrane dynamics lookup tables', self.mechname)

        # Get Lookup
        self.getModLookup()

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
        return toggleStim(self)

    def setProbesDict(self, sec):
        return {
            **{
                'Qm': setRangeProbe(sec, 'v'),
                'Vm': setRangeProbe(sec, 'Vm_{}'.format(self.mechname))
            },
            **{
                k: setRangeProbe(sec, '{}_{}'.format(alias(k), self.mechname))
                for k in self.pneuron.statesNames()
            }
        }

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    def simulate(self, A, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param A: stimulus amplitude (in modality units)
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
            :return: output dataframe
        '''
        logger.info(self.desc(self.meta(A, pp)))

        # Set recording vectors
        t = setTimeProbe()
        stim = setStimProbe(self.section, self.mechname)
        probes = self.setProbesDict(self.section)

        # Set stimulus amplitude and integrate model
        self.setStimAmp(A)
        integrate(self, pp, dt, atol)

        # Store output in dataframe
        data = pd.DataFrame({
            't': vec_to_array(t) * 1e-3,  # s
            'stimstate': vec_to_array(stim)
        })
        for k, v in probes.items():
            data[k] = vec_to_array(v)
        data.loc[:,'Qm'] *= 1e-5  # C/m2

        # Prepend initial conditions (prior to stimulation)
        data = prependDataFrame(data)

        return data

    def meta(self, A, pp):
        meta = self.pneuron.meta(A, pp)
        return meta

    def desc(self, meta):
        m = self.modality
        Astr = f'{m["name"]} = {si_format(meta[m["name"]] * m["factor"], 2)}{m["unit"]}'
        return f'{self}: simulation @ {Astr}, {meta["pp"].pprint()}'

    def titrate(self, pp, xfunc=None):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param pp: pulsed protocol object
            :param xfunc: function determining whether condition is reached from simulation output
            :return: determined threshold amplitude (Pa)
        '''
        # Default output function
        if xfunc is None:
            xfunc = self.pneuron.titrationFunc

        return binarySearch(
            lambda x: xfunc(self.simulate(*x)[0]),
            [pp], 0, self.Arange, self.A_conv_thr)

    @property
    @abc.abstractmethod
    def filecodes(self, *args):
        return NotImplementedError

    def filecode(self, *args):
        return filecode(self, *args)

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)


class IintraNode(Node):
    ''' Node used for simulations with intracellular current. '''

    modality = {
        'name': 'Astim',
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
        logger.debug(f'Equivalent injected current: {self.Iinj:.1f} nA')
        self.iclamp = h.IClamp(self.section(0.5))
        self.iclamp.delay = 0  # we want to exert control over amp starting at 0 ms
        self.iclamp.dur = 1e9  # dur must be long enough to span all our changes

    def setStimON(self, value):
        value = super().setStimON(value)
        self.iclamp.amp = value * self.Iinj
        return value

    def filecodes(self, *args):
        codes = self.pneuron.filecodes(*args)
        codes['method'] = 'NEURON'
        return codes


class SonicNode(Node):
    ''' Node used for simulations with US stimulus. '''

    modality = {
        'name': 'Adrive',
        'unit': 'Pa',
        'factor': 1e0
    }
    A_conv_thr = THRESHOLD_CONV_RANGE_ASTIM

    def __init__(self, pneuron, *args, **kwargs):
        ''' Initialization.

            :param pneuron: point-neuron model
            :param a: sonophore diameter (m)
            :param Fdrive: ultrasound frequency (Hz)
            :param fs: sonophore membrane coverage fraction (-)
        '''
        self.a = kwargs.pop('a', 32e-9)  # m
        self.Fdrive = kwargs.pop('Fdrive', 500e3)  # Hz
        self.fs = kwargs.pop('fs', 1.)  # -
        if self.fs > 1. or self.fs < 0.:
            raise ValueError('fs ({}) must be within [0-1]'.format(self.fs))
        if 'nbls' in kwargs:
            self.nbls = kwargs.pop('nbls')
        else:
            self.nbls = NeuronalBilayerSonophore(self.a, pneuron, self.Fdrive)
        super().__init__(pneuron, *args, **kwargs)
        self.Arange = (0., self.pylkp.refs['A'].max())

    def __repr__(self):
        return '{}({:.1f} nm, {}, {:.0f} kHz, fs={})'.format(
            self.__class__.__name__, self.a * 1e9, self.pneuron, self.Fdrive * 1e-3, self.fs)

    def str_biophysics(self):
        return super().str_biophysics() + ', a = {}m{}, f = {}Hz'.format(
            si_format(self.a, space=' '),
            ', fs = {:.0f}%'.format(self.fs * 1e2) if self.fs is not None else '',
            si_format(self.Fdrive, space=' '))

    def getPyLookup(self):
        return self.nbls.getLookup2D(self.Fdrive, self.fs)

    def setStimAmp(self, Adrive):
        ''' Set US stimulation amplitude.

            :param Adrive: acoustic pressure amplitude (Pa)
        '''
        self.printStimAmp(Adrive)
        setattr(self.section, 'Adrive_{}'.format(self.mechname), Adrive * 1e-3)

    def filecodes(self, *args):
        return self.nbls.filecodes(self.Fdrive, *args, self.fs, 'NEURON', None)

    def meta(self, A, pp):
        meta = self.nbls.meta(self.Fdrive, A, pp, self.fs, 'NEURON', None)
        return meta

    def getPltVars(self, *args, **kwargs):
        return self.nbls.getPltVars(*args, **kwargs)

    def getPltScheme(self, *args, **kwargs):
        return self.nbls.getPltScheme(*args, **kwargs)

    @logCache(os.path.join(os.path.split(__file__)[0], 'sonicnode_titrations.log'))
    def titrate(self, pp, xfunc=None):
        return super().titrate(pp, xfunc=xfunc)

