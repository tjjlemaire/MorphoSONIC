# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-23 09:43:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-22 11:40:57

import abc
import numpy as np
from scipy.optimize import brentq

from PySONIC.utils import logger, gaussian
from PySONIC.core.stimobj import StimObject
from PySONIC.core.drives import *

from ..constants import *


class Source(StimObject):

    @abc.abstractmethod
    def computeDistributedAmps(self, fiber):
        raise NotImplementedError

    @abc.abstractmethod
    def computeSourceAmp(self, fiber):
        raise NotImplementedError

    @property
    def is_searchable(self):
        return True

    @property
    def is_cathodal(self):
        return False


class XSource(Source):

    xvar_precheck = False

    @property
    @abc.abstractmethod
    def xvar(self):
        raise NotImplementedError

    @xvar.setter
    @abc.abstractmethod
    def xvar(self, value):
        raise NotImplementedError

    def updatedX(self, value):
        other = self.copy()
        other.xvar = value
        return other

    @property
    def is_searchable(self):
        return True

    @property
    def is_resolved(self):
        return self.xvar is not None


class SectionSource(XSource):

    def __init__(self, sec_id):
        ''' Constructor.

            :param sec_id: section ID
        '''
        self.sec_id = sec_id

    @property
    def sec_id(self):
        return self._sec_id

    @sec_id.setter
    def sec_id(self, value):
        if not isinstance(value, str):
            raise ValueError('section ID must be a string')
        self._sec_id = value

    @staticmethod
    def inputs():
        return {
            'sec_id': {
                'desc': 'section ID',
                'label': 'ID'
            }
        }

    def computeDistributedAmps(self, model):
        d = {}
        match = False
        for sectype, secdict in model.sections.items():
            d[sectype] = np.zeros(len(secdict))
            if self.sec_id in secdict.keys():
                i = list(secdict.keys()).index(self.sec_id)
                d[sectype][i] = self.xvar
                match = True
        if not match:
            raise ValueError(f'{self.sec_id} section ID not found in {model}')
        return d

    def computeSourceAmp(self, model, A):
        return A


class UniformSource(XSource):

    def computeDistributedAmps(self, model):
        return {k: np.ones(len(v)) * self.xvar for k, v in model.sections.items()}

    def computeSourceAmp(self, model, A):
        return A


class GaussianSource(XSource):

    # Ratio between RMS width and full width at half maximum
    sigma_to_fwhm = 2 * np.sqrt(2 * np.log(2))

    def __init__(self, x0, sigma):
        ''' Constructor.

            :param x0: gaussian center coordinate (m)
            :param sigma: gaussian RMS width (m)
        '''
        self.x0 = x0
        self.sigma = sigma

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        value = self.checkFloat('center', value)
        self._x0 = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        value = self.checkFloat('width', value)
        self.checkStrictlyPositive('width', value)
        self._sigma = value

    @classmethod
    def from_FWHM(cls, w):
        return w / cls.sigma_to_fwhm

    @property
    def FWHM(self):
        ''' Full width at half maximum. '''
        return self.sigma * self.sigma_to_fwhm

    @staticmethod
    def inputs():
        return {
            'x0': {
                'desc': 'center coordinate',
                'label': 'x0',
                'unit': 'm',
                'precision': 1
            },
            'sigma': {
                'desc': 'width',
                'label': 'sigma',
                'unit': 'm',
                'precision': 1
            }
        }

    def getField(self, x):
        return gaussian(x, mu=self.x0, sigma=self.sigma, A=self.xvar)

    def computeDistributedAmps(self, fiber):
        if fiber.length < MIN_FIBERL_FWHM_RATIO * self.FWHM:
            logger.warning('fiber is too short w.r.t stimulus FWHM')
        return {k: self.getField(v) for k, v in fiber.getXCoords().items()}

    def computeSourceAmp(self, fiber, A):
        return A


class GammaSource(XSource):

    def __init__(self, gamma_dict, f=None):
        self.gamma_dict = gamma_dict
        self.f = f

    def copy(self):
        return self.__class__(self.gamma_dict, f=self.f)

    @property
    def gamma_dict(self):
        return self._gamma_dict

    @gamma_dict.setter
    def gamma_dict(self, value):
        self._gamma_dict = value

    @property
    def gamma_range(self):
        gamma_min = min(v.min() for v in self.gamma_dict.values())
        gamma_max = max(v.max() for v in self.gamma_dict.values())
        return gamma_min, gamma_max

    @property
    def xvar(self):
        return self.gamma_range[1]

    @staticmethod
    def inputs():
        return {
            'gamma_range': {
                'desc': 'gamma range',
                'label': 'gamma',
                'unit': '',
                'precision': 2
            }
        }

    def computeDistributedAmps(self, fiber):
        return self.gamma_dict

    def computeSourceAmp(self, fiber, A):
        return A


class ExtracellularSource(XSource):

    def __init__(self, x):
        ''' Constructor.

            :param x: n-dimensional position vector.
        '''
        self.x = x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if value is not None:
            value = tuple([self.checkFloat('x', v) for v in value])
        self._x = value

    @property
    def y(self):
        if self.nx <= 2:
            raise ValueError('Cannot return y-component for a 2D position (XZ)')
        return self._x[1]

    @property
    def z(self):
        return self._x[-1]

    @property
    def xz(self):
        return (self._x[0], self._x[-1])

    @property
    def nx(self):
        return len(self.x)

    def strPos(self):
        return self.paramStr('x')

    @staticmethod
    def inputs():
        return {
            'x': {
                'desc': 'position',
                'label': 'x',
                'unit': 'm',
                'precision': 1
            }
        }

    @property
    def zstr(self):
        d = self.inputs()['x']
        return f'z{self.z * d.get("factor", 1.):.1f}{d["unit"]}'

    def vectorialDistance(self, x):
        ''' Vectorial distance(s) to target point(s).

            :param x: target point(s) location (m)
            :return: vectorial distance(s) (m)
        '''
        return np.asarray(self.x) - np.asarray(x)

    def euclidianDistance(self, x):
        ''' Euclidian distance(s) to target point(s).

            :param x: target point(s) location (m)
            :return: Euclidian distance(s) (m)
        '''
        return np.linalg.norm(self.vectorialDistance(x), axis=-1)

    def vDistances(self, fiber):
        return {k: self.vectorialDistance(v) for k, v in fiber.getXZCoords().items()}  # m

    def eDistances(self, fiber):
        return {k: self.euclidianDistance(v) for k, v in fiber.getXCoords().items()}  # m

    def getMinNodeDistance(self, fiber):
        return min(self.eDistances(fiber)['node'])  # m


class SourceArray:

    def __init__(self, psources, rel_amps):
        if len(rel_amps) != len(psources):
            raise ValueError('number of point-sources does not match number of relative amplitudes')
        self.rel_amps = rel_amps
        self.psources = psources

    def strAmp(self, A):
        return ', '.join([p.strAmp(A * r) for p, r in zip(self.psources, self.rel_amps)])

    def computeDistributedAmps(self, fiber, A):
        amps = np.array([
            p.computeDistributedAmps(fiber, A * r) for p, r in zip(self.psources, self.rel_amps)])
        return amps.sum(axis=0)

    def computeSourceAmp(self, fiber, A):
        # Compute the individual source amplitudes required for each point source
        # to reach the desired output amplitude at their closest fiber node
        source_amps = np.abs([p.computeSourceAmp(fiber, A) / r
                              for p, r in zip(self.psources, self.rel_amps)])

        # Define an exploration range for the combined effect of all sources
        # to reach the target amplitude at a fiber node
        Amin, Amax = min(source_amps) * 1e-3, max(source_amps) * 1e3

        # Search for source array amplitude that matches the target amplitude
        Asource = brentq(
            lambda x: self.computeDistributedAmps(fiber, x).max() - A, Amin, Amax, xtol=1e-16)
        return Asource

    def filecodes(self, A):
        keys = self.psources[0].filecodes(A)
        return {key: '_'.join([p.filecodes(A * r)[key]
                               for p, r in zip(self.psources, self.rel_amps)])
                for key in keys}

    def strPos(self):
        return '({})'.format(', '.join([p.strPos() for p in self.psources]))
