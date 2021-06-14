# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-23 09:43:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-14 16:24:59

from .source import XSource, UniformSource, GaussianSource


class VoltageSource(XSource):

    conv_factor = 1e0  # mV
    xkey = 'Ve'
    key = 'Vext'
    polarities = ('anode', 'cathode')

    def __init__(self, Ve, mode=None):
        ''' Constructor.

            :param Ve: extracellular voltage amplitude (mV)
            :param mode: polarity mode ("cathode" or "anode")
        '''
        self.mode = mode
        self.Ve = Ve

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in self.polarities:
            raise ValueError(f'Unknown polarity: {value} (should be in one of {self.polarities})')
        self._mode = value

    @property
    def Ve(self):
        return self._Ve

    @Ve.setter
    def Ve(self, value):
        if value is not None:
            value = self.checkFloat('Ve', value)
            if self.mode == 'cathode':
                self.checkNegativeOrNull('Ve', value)
            else:
                self.checkPositiveOrNull('Ve', value)
        self._Ve = value

    @property
    def is_cathodal(self):
        return self.mode == 'cathode'

    @staticmethod
    def inputs():
        return {
            'Ve': {
                'desc': 'extracellular voltage amplitude',
                'label': 'Ve',
                'unit': 'V',
                'factor': 1e-3,
                'precision': 1
            },
            'mode': {
                'desc': 'polarity mode',
                'label': 'mode',
            }
        }

    @property
    def xvar(self):
        return self.Ve

    @xvar.setter
    def xvar(self, value):
        self.Ve = value


class UniformVoltageSource(UniformSource, VoltageSource):

    def __init__(self, Ve=None, mode='cathode'):
        ''' Constructor.

            :param Ve: Extracellular voltage (mV)
        '''
        VoltageSource.__init__(self, Ve, mode=mode)

    def copy(self):
        return self.__class__(Ve=self.Ve, mode=self.mode)

    def computeDistributedAmps(self, fiber):
        return {k: v * self.conv_factor
                for k, v in UniformSource.computeDistributedAmps(self, fiber).items()}

    @staticmethod
    def inputs():
        return VoltageSource.inputs()

    def computeMaxNodeAmp(self, fiber):
        return self.Ve


class GaussianVoltageSource(GaussianSource, VoltageSource):

    def __init__(self, x0, sigma, Ve=None, mode='cathode'):
        ''' Constructor.

            :param Ve: Extracellular voltage (mV)
        '''
        GaussianSource.__init__(self, x0, sigma)
        VoltageSource.__init__(self, Ve, mode=mode)

    def copy(self):
        return self.__class__(self.x0, self.sigma, Ve=self.Ve, mode=self.mode)

    def computeDistributedAmps(self, fiber):
        return {k: v * self.conv_factor
                for k, v in GaussianSource.computeDistributedAmps(self, fiber).items()}

    @staticmethod
    def inputs():
        return {**GaussianSource.inputs(), **VoltageSource.inputs()}

    def computeMaxNodeAmp(self, fiber):
        return self.Ve
