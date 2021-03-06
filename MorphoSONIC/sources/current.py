# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-23 09:43:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-14 16:29:40

import numpy as np

from PySONIC.utils import isIterable

from ..constants import *
from .source import XSource, SectionSource, ExtracellularSource


class CurrentSource(XSource):

    xkey = 'I'
    polarities = ('anode', 'cathode')

    def __init__(self, I, mode=None):
        ''' Constructor.

            :param I: current amplitude (A)
            :param mode: polarity mode ("cathode" or "anode")
        '''
        self.mode = mode
        self.I = I

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in self.polarities:
            raise ValueError(f'Unknown polarity: {value} (should be in one of {self.polarities})')
        self._mode = value

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, value):
        if value is not None:
            value = self.checkFloat('I', value)
            if self.mode == 'cathode':
                self.checkNegativeOrNull('I', value)
            else:
                self.checkPositiveOrNull('I', value)
        self._I = value

    @property
    def xvar(self):
        return self.I

    @xvar.setter
    def xvar(self, value):
        self.I = value

    @property
    def is_cathodal(self):
        return self.mode == 'cathode'

    @staticmethod
    def inputs():
        return {
            'I': {
                'desc': 'current amplitude',
                'label': 'I',
                'unit': 'A',
                'precision': 1
            },
            'mode': {
                'desc': 'polarity mode',
                'label': 'mode'
            }
        }


class IntracellularCurrent(SectionSource, CurrentSource):

    conv_factor = A_TO_NA
    key = 'Iintra'

    def __init__(self, sec_id, I=None, mode='anode'):
        SectionSource.__init__(self, sec_id)
        CurrentSource.__init__(self, I, mode)

    def computeDistributedAmps(self, model):
        if not self.is_resolved:
            raise ValueError('Cannot compute field distribution: unresolved source')
        return {k: v * self.conv_factor
                for k, v in SectionSource.computeDistributedAmps(self, model).items()}

    def copy(self):
        return self.__class__(self.sec_id, self.I, mode=self.mode)

    @staticmethod
    def inputs():
        return {**SectionSource.inputs(), **CurrentSource.inputs()}


class ExtracellularCurrent(ExtracellularSource, CurrentSource):

    conv_factor = 1 / MA_TO_A
    key = 'Iextra'

    def __init__(self, x, I=None, mode='cathode', rho=300.0):
        ''' Initialization.

            :param rho: extracellular medium resistivity (Ohm.cm)
        '''
        ExtracellularSource.__init__(self, x)
        CurrentSource.__init__(self, I, mode)
        self.rho = rho

    @property
    def rho(self):
        if all([x == self._rho[0] for x in self._rho]):
            return self._rho[0]
        else:
            return self._rho

    @rho.setter
    def rho(self, value):
        if not isIterable(value):
            value = tuple([value] * self.nx)
        if len(value) != self.nx:
            raise ValueError(f'rho must be either a scalar or a {self.nx}-elements vector like x')
        value = [self.checkFloat('rho', v) for v in value]
        for v in value:
            self.checkStrictlyPositive('rho', v)
        if self.nx == 3 and value[1] != value[2]:
            raise ValueError('transverse resistivites must be equal')
        self._rho = value

    def copy(self):
        return self.__class__(self.x, self.I, mode=self.mode, rho=self.rho)

    @staticmethod
    def inputs():
        return {
            **ExtracellularSource.inputs(), **CurrentSource.inputs(),
            'rho': {
                'desc': 'extracellular resistivity',
                'label': 'rho',
                'unit': 'Ohm.cm',
                'precision': 1
            }
        }

    def conductance(self, d):
        ''' Compute the conductance resulting from integrating the medium's resistivity
            along the electrode-target path.

            :param d: vectorial distance(s) to target point(s) (m)
            :return: integrated conductance (S)
        '''
        d = d.T
        # square_S = (np.linalg.norm(d, axis=0) / self.rho)**2
        square_S = d[0]**2 / self._rho[-1]**2
        for dd, rho in zip(d[1:], self._rho[1:]):
            square_S += dd**2 / (rho * self._rho[0])
        return 4 * np.pi * np.sqrt(square_S) * M_TO_CM  # S

    def Vext(self, I, d):
        ''' Compute the extracellular electric potential generated by a given point-current source
            at a given distance in a homogenous, isotropic medium.

            :param I: point-source current amplitude (A)
            :param d: vectorial distance(s) to target point(s) (m)
            :return: extracellular potential(s) (mV)
        '''
        return I / self.conductance(d) * V_TO_MV  # mV

    def Iext(self, V, d):
        ''' Compute the point-current source amplitude that generates a given extracellular
            electric potential at a given distance in a homogenous, isotropic medium.

            :param V: electric potential (mV)
            :param d: vectorial distance(s) to target point(s) (m)
            :return: point-source current amplitude (A)
        '''
        return V * self.conductance(d) / V_TO_MV  # A

    def computeDistributedAmps(self, fiber):
        ''' Compute extracellular potential value at all fiber sections. '''
        if not self.is_resolved:
            raise ValueError('Cannot compute field distribution: unresolved source')
        return {k: self.Vext(self.I, d) for k, d in self.vDistances(fiber).items()}  # mV

    def computeSourceAmp(self, fiber, Ve):
        ''' Compute the current needed to generate the extracellular potential value
            at closest node. '''
        xnodes = fiber.getXZCoords()['node']  # fiber nodes coordinates
        i_closestnode = self.euclidianDistance(xnodes).argmin()  # index of closest fiber node
        return self.Iext(Ve, self.vectorialDistance(xnodes[i_closestnode]))  # A
