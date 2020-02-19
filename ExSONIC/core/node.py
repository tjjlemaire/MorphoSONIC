# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 09:23:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-19 22:05:30

import os
import abc
from inspect import signature
import numpy as np
import pandas as pd

from PySONIC.constants import *
from PySONIC.core import Model, PointNeuron, NeuronalBilayerSonophore
from PySONIC.utils import si_format, logger, logCache, filecode, simAndSave
from PySONIC.threshold import threshold
from PySONIC.postpro import prependDataFrame

from ..utils import getNmodlDir, load_mechanisms
from ..constants import *
from .nmodel import NeuronModel


class Node(NeuronModel):
    ''' Generic node interface. '''

    tscale = 'ms'  # relevant temporal scale of the model

    def __init__(self, pneuron, id=None, construct=True):
        ''' Initialization.

            :param pneuron: point-neuron model
        '''
        # Initialize arguments
        self.pneuron = pneuron
        if id is None:
            id = self.__repr__()
        self.id = id
        logger.debug('Creating {} model'.format(self))

        # Load mechanisms and set appropriate membrane mechanism
        load_mechanisms(getNmodlDir(), self.modfile)

        # Construct model section and set membrane mechanism
        if construct:
            self.section = self.createSection(self.id)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pneuron)

    def str_biophysics(self):
        return '{} neuron'.format(self.pneuron.name)

    def clear(self):
        del self.section

    def getAreaNormalizationFactor(self):
        ''' Return area normalization factor '''
        A0 = self.section(0.5).area() * 1e-12  # section area (m2)
        A = self.pneuron.area                  # neuron membrane area (m2)
        return A0 / A

    @abc.abstractmethod
    def setPyLookup(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def setDrive(self, value):
        raise NotImplementedError

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        self.section.setStimON(value)
        return value

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    def simulate(self, drive, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param drive: drive object
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
            :return: output dataframe
        '''
        logger.info(self.desc(self.meta(drive, pp)))

        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.section.setStimProbe()
        probes = self.section.setProbesDict()

        # Set drive and integrate model
        self.setDrive(drive)
        self.integrate(pp, dt, atol)

        # Store output in dataframe
        data = pd.DataFrame({
            't': t.to_array() * 1e-3,  # s
            'stimstate': stim.to_array()
        })
        for k, v in probes.items():
            data[k] = v.to_array()
        data.loc[:,'Qm'] *= 1e-5  # C/m2

        # Prepend initial conditions (prior to stimulation)
        data = prependDataFrame(data)

        return data

    def meta(self, drive, pp):
        return self.pneuron.meta(drive, pp)

    def desc(self, meta):
        return f'{self}: simulation @ {meta["drive"].desc}, {meta["pp"].desc}'

    def titrate(self, drive, pp, xfunc=None):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param drive: unresolved drive object
            :param pp: pulsed protocol object
            :param xfunc: function determining whether condition is reached from simulation output
            :return: determined threshold amplitude (Pa)
        '''
        # Default output function
        if xfunc is None:
            xfunc = self.pneuron.titrationFunc

        return threshold(
            lambda x: xfunc(self.simulate(drive.updatedX(x), pp)[0]),
            self.Arange, x0=self.A_conv_initial,
            eps_thr=self.A_conv_thr, rel_eps_thr=self.A_conv_rel_thr,
            precheck=self.A_conv_precheck)

    @property
    @abc.abstractmethod
    def filecodes(self, *args):
        raise NotImplementedError

    def filecode(self, *args):
        return filecode(self, *args)

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)


class IintraNode(Node):
    ''' Node used for simulations with intracellular current. '''

    A_conv_initial = ESTIM_AMP_INITIAL
    A_conv_rel_thr = ESTIM_REL_CONV_THR
    A_conv_thr = None
    A_conv_precheck = False

    @property
    def Arange(self):
        return self.pneuron.Arange

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set invariant function tables
        self.setFuncTables()

    def clear(self):
        super().clear()
        if hasattr(self, 'iclamp'):
            del self.iclamp

    def setPyLookup(self):
        self.pylkp = self.pneuron.getLookup()
        self.pylkp.refs['A'] = np.array([0.])
        for k, v in self.pylkp.items():
            self.pylkp[k] = np.array([v])

    def setDrive(self, drive):
        ''' Set electrical stimulation amplitude

            :param drive: electric drive object.
        '''
        logger.debug(f'Stimulus: {drive}')
        Iinj = drive.I * self.section(0.5).area() * 1e-6  # nA
        logger.debug(f'Equivalent injected current: {Iinj:.1f} nA')
        self.iclamp = IClamp(self.section(0.5), Iinj)

    def setStimON(self, value):
        value = super().setStimON(value)
        self.iclamp.toggle(value)
        return value

    def filecodes(self, *args):
        codes = self.pneuron.filecodes(*args)
        codes['method'] = 'NEURON'
        return codes


class SonicNode(Node):
    ''' Node used for simulations with US stimulus. '''

    A_conv_initial = ASTIM_AMP_INITIAL
    A_conv_rel_thr = 1e0
    A_conv_thr = ASTIM_ABS_CONV_THR
    A_conv_precheck = True

    def __init__(self, pneuron, *args, **kwargs):
        ''' Initialization.

            :param pneuron: point-neuron model
            :param a: sonophore diameter (m)
            :param fs: sonophore membrane coverage fraction (-)
        '''
        self.a = kwargs.pop('a', 32e-9)  # m
        self.fs = kwargs.pop('fs', 1.)   # -
        if self.fs > 1. or self.fs < 0.:
            raise ValueError('fs ({}) must be within [0-1]'.format(self.fs))
        if 'nbls' in kwargs:
            self.nbls = kwargs.pop('nbls')
        else:
            self.nbls = NeuronalBilayerSonophore(self.a, pneuron)
        self.fref = None
        self.pylkp = None
        super().__init__(pneuron, *args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.a * 1e9:.1f} nm, {self.pneuron}, fs={self.fs:.2f})'

    def str_biophysics(self):
        fs_str = f', fs = {self.fs * 1e2:.0f}%' if self.fs is not None else ''
        return f'{super().str_biophysics()}, a = {si_format(self.a)}m{fs_str}'

    @property
    def Arange(self):
        return (0., self.pylkp.refs['A'].max())

    def setPyLookup(self, f):
        if self.pylkp is None or f != self.fref:
            self.pylkp = self.nbls.getLookup2D(f, self.fs)
            self.fref = f

    def setDrive(self, drive):
        ''' Set US drive. '''
        logger.debug(f'Stimulus: {drive}')
        self.setFuncTables(drive.f)
        self.section.setMechValue('Adrive', drive.A * 1e-3)

    def filecodes(self, *args):
        return self.nbls.filecodes(*args, self.fs, 'NEURON', None)

    def meta(self, drive, pp):
        return self.nbls.meta(drive, pp, self.fs, 'NEURON', None)

    def getPltVars(self, *args, **kwargs):
        return self.nbls.getPltVars(*args, **kwargs)

    @property
    def pltScheme(self):
        return self.nbls.pltScheme

    @logCache(os.path.join(os.path.split(__file__)[0], 'sonicnode_titrations.log'))
    def titrate(self, drive, pp, xfunc=None):
        self.setFuncTables(drive.f)  # pre-loading lookups to have a defined Arange
        return super().titrate(drive, pp, xfunc=xfunc)


class DrivenSonicNode(SonicNode):

    def __init__(self, pneuron, Idrive, *args, **kwargs):
        self.Idrive = Idrive
        SonicNode.__init__(self, pneuron, *args, **kwargs)
        self.setDrive()
        Qrange = self.pylkp.refs['Q']

    def setDrive(self):
        logger.debug(f'setting {self.Idrive:.2f} mA/m2 driving current')
        Iinj = self.Idrive * self.section(0.5).area() * 1e-6  # nA
        logger.debug(f'Equivalent injected current: {Iinj:.1f} nA')
        self.iclamp = IClamp(self.section(0.5), Iinj)
        self.iclamp.toggle(1)

    def __repr__(self):
        return super().__repr__()[:-1] + f', Idrive = {self.Idrive:.2f} mA/m2)'

    def filecodes(self, *args):
        codes = SonicNode.filecodes(self, *args)
        codes['Idrive'] = f'Idrive{self.Idrive:.1f}mAm2'
        return codes

    def meta(self, A, pp):
        meta = super().meta(A, pp)
        meta['Idrive'] = self.Idrive
        return meta
