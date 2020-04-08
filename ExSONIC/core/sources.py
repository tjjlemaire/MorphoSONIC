# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-23 09:43:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-07 17:34:06

import abc
import numpy as np
from scipy.optimize import brentq
from scipy.signal import unit_impulse

from PySONIC.utils import si_format, rotAroundPoint2D, StimObject, gaussian, isIterable
from PySONIC.core.drives import *

from .grids import getCircle2DGrid
from ..constants import *


class Source(StimObject):

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

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
                'label': 'ID',
                'unit': ''
            }
        }

    def __repr__(self):
        return f'{self.__class__.__name__}({self.sec_id})'

    def filecodes(self):
        return {'sec_id': self.sec_id}

    @property
    def quickcode(self):
        return self.sec_id

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


class GaussianSource(XSource):

    def __init__(self, x0, sigma):
        ''' Constructor.

            :param x0: gaussian center coordinate (m)
            :param sigma: gaussian width (m)
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

    @staticmethod
    def inputs():
        return {
            'x0': {
                'desc': 'center coordinate',
                'label': 'x0',
                'unit': 'mm',
                'factor': M_TO_MM,
                'precision': 1
            },
            'sigma': {
                'desc': 'width',
                'label': 'sigma',
                'unit': 'mm',
                'factor': M_TO_MM,
                'precision': 1
            }
        }

    def __repr__(self):
        params = [f'{key} = {value}' for key, value in self.filecodes().items()]
        return f'{self.__class__.__name__}({", ".join(params)})'

    def filecodes(self):
        codes = {}
        for key in ['x0', 'sigma']:
            d = self.inputs()[key]
            codes[key] = f'{getattr(self, key) * d.get("factor", 1.):.3f}{d["unit"]}'
        return codes

    @property
    def quickcode(self):
        return f'sigma{self.sigma * M_TO_MM:.3f}mm'

    def computeDistributedAmps(self, fiber):
        return {k: gaussian(v, mu=self.x0, sigma=self.sigma, A=self.xvar)
                for k, v in fiber.getXCoords().items()}

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
        d = self.inputs()["x"]
        x_mm = [f'{xx * d.get("factor", 1.):.1f}' for xx in self.x]
        return f'({",".join(x_mm)}){d["unit"]}'

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.strPos()})'

    @staticmethod
    def inputs():
        return {
            'x': {
                'desc': 'position',
                'label': 'x',
                'unit': 'mm',
                'factor': M_TO_MM,
                'precision': 1
            }
        }

    def filecodes(self):
        return {'position': self.strPos().replace(',', '_').replace('(', '').replace(')', '')}

    @property
    def zstr(self):
        d = self.inputs()['x']
        return f'z{self.z * d.get("factor", 1.):.1f}{d["unit"]}'

    @property
    def quickcode(self):
        return f'x{self.strPos()}'

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

    def Istr(self):
        return f'{si_format(self.I, 2)}{self.inputs()["I"]["unit"]}'

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
                'label': 'mode',
                'unit': '',
            }
        }

    def filecodes(self):
        unit = self.inputs()["I"]["unit"]
        prefix = si_format(1 / self.conv_factor)[-1]
        return {'I': f'{(self.I * self.conv_factor):.2f}{prefix}{unit}'}

    @property
    def xvar(self):
        return self.I

    @xvar.setter
    def xvar(self, value):
        self.I = value


class IntracellularCurrent(SectionSource, CurrentSource):

    conv_factor = A_TO_NA
    key = 'Iintra'

    def __init__(self, sec_id, I=None, mode='anode'):
        SectionSource.__init__(self, sec_id)
        CurrentSource.__init__(self, I, mode)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['sec_id', 'mode', 'I']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{SectionSource.__repr__(self)[:-1]}, {self.mode}'
        if self.I is not None:
            s = f'{s}, {self.Istr()}'
        return f'{s})'

    def computeDistributedAmps(self, model):
        if not self.is_resolved:
            raise ValueError('Cannot compute field distribution: unresolved source')
        return {k: v * self.conv_factor
                for k, v in SectionSource.computeDistributedAmps(self, model).items()}

    def filecodes(self):
        return {**SectionSource.filecodes(self), **CurrentSource.filecodes(self)}

    def copy(self):
        return self.__class__(self.sec_id, self.I, mode=self.mode)

    @staticmethod
    def inputs():
        return {**SectionSource.inputs(), **CurrentSource.inputs()}

    @property
    def quickcode(self):
        return f'{self.key}_{SectionSource.quickcode.fget(self)}'


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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['x', 'mode', 'I', 'rho']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{ExtracellularSource.__repr__(self)[:-1]}, {self.strRho}, {self.mode}'
        if self.I is not None:
            s = f'{s}, {self.Istr()}'
        return f'{s})'

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

    @property
    def strRho(self):
        if isIterable(self.rho):
            s = f'({",".join(f"{x:.2f}" for x in self.rho)})'
        else:
            s = f'{self.rho:.0f}'
        return f'{s}{self.inputs()["rho"]["unit"]}'

    def filecodes(self):
        return {
            **ExtracellularSource.filecodes(self), **CurrentSource.filecodes(self),
            'rho': self.strRho.replace(',', '_').replace('(', '').replace(')', '')
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

    @property
    def quickcode(self):
        return f'{self.key}_{ExtracellularSource.quickcode.fget(self)}'


class VoltageSource(XSource):

    conv_factor = 1e0  # mV
    xkey = 'Ve'
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
                'unit': 'mV',
                'precision': 1
            },
            'mode': {
                'desc': 'polarity mode',
                'label': 'mode',
                'unit': '',
            }
        }

    @property
    def xvar(self):
        return self.Ve

    @xvar.setter
    def xvar(self, value):
        self.Ve = value

    def Vstr(self):
        return f'{self.Ve:.1f} {self.inputs()["Ve"]["unit"]}'

    def filecodes(self):
        unit = self.inputs()["Ve"]["unit"]
        prefix = si_format(1 / self.conv_factor)[-1]
        return {'Ve': f'{(self.Ve * self.conv_factor):.2f}{prefix}{unit}'}


class GaussianVoltageSource(GaussianSource, VoltageSource):

    def __init__(self, x0, sigma, Ve=None, mode='cathode'):
        ''' Constructor.

            :param Ve: Extracellular voltage (mV)
        '''
        GaussianSource.__init__(self, x0, sigma)
        VoltageSource.__init__(self, Ve, mode=mode)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['x0', 'sigma', 'Ve', 'mode']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{GaussianSource.__repr__(self)[:-1]}, {self.mode}'
        if self.Ve is not None:
            s = f'{s}, {self.Vstr()}'
        return f'{s})'

    def copy(self):
        return self.__class__(self.x0, self.sigma, Ve=self.Ve, mode=self.mode)

    def computeDistributedAmps(self, fiber):
        return {k: v * self.conv_factor
                for k, v in GaussianSource.computeDistributedAmps(self, fiber)}

    def filecodes(self):
        return {**GaussianSource.filecodes(self), **VoltageSource.filecodes(self)}

    @staticmethod
    def inputs():
        return {**GaussianSource.inputs(), **VoltageSource.inputs()}

    def computeMaxNodeAmp(self, fiber):
        return self.Ve


class AcousticSource(XSource):

    conv_factor = PA_TO_KPA

    def __init__(self, f):
        ''' Constructor.

            :param f: US frequency (Hz).
        '''
        self.f = f

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        value = self.checkFloat('f', value)
        self.checkStrictlyPositive('f', value)
        self._f = value

    @staticmethod
    def inputs():
        return {
            'f': {
                'desc': 'US drive frequency',
                'label': 'f',
                'unit': 'Hz',
                'factor': HZ_TO_KHZ,
                'precision': 0
            }
        }

    def filecodes(self):
        return {'f': f'{self.f * HZ_TO_KHZ:.0f}kHz'}

    @property
    def quickcode(self):
        return f'f{si_format(self.f, 0, space="")}{self.inputs()["f"]["unit"]}'


class SectionAcousticSource(SectionSource, AcousticSource):

    def __init__(self, sec_id, f, A=None):
        ''' Constructor.

            :param A: Acoustic amplitude (Pa)
        '''
        SectionSource.__init__(self, sec_id)
        AcousticSource.__init__(self, f)
        self.A = A

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        if value is not None:
            value = self.checkFloat('A', value)
            self.checkPositiveOrNull('A', value)
        self._A = value

    @property
    def xvar(self):
        return self.A

    @xvar.setter
    def xvar(self, value):
        self.A = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['sec_id', 'f', 'A']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{SectionSource.__repr__(self)[:-1]}, {si_format(self.f, 1, space="")}Hz'
        if self.A is not None:
            s = f'{s}, {si_format(self.A, 1, space="")}Pa'
        return f'{s})'

    def computeDistributedAmps(self, fiber):
        return SectionSource.computeDistributedAmps(self, fiber)

    def filecodes(self):
        return {
            **SectionSource.filecodes(self), **AcousticSource.filecodes(self),
            'A': f'{(self.A * self.conv_factor):.2f}kPa'
        }

    @property
    def quickcode(self):
        return '_'.join([
            AcousticSource.quickcode.fget(self),
            SectionSource.quickcode.fget(self)
        ])

    def copy(self):
        return self.__class__(self.sec_id, self.f, A=self.A)

    @staticmethod
    def inputs():
        return {
            **SectionSource.inputs(), **AcousticSource.inputs(),
            'A': {
                'desc': 'US pressure amplitude',
                'label': 'A',
                'unit': 'kPa',
                'factor': PA_TO_KPA,
                'precision': 2
            }
        }

    def computeMaxNodeAmp(self, fiber):
        return self.A


class GaussianAcousticSource(GaussianSource, AcousticSource):

    def __init__(self, x0, sigma, f, A=None):
        ''' Constructor.

            :param A: Acoustic amplitude (Pa)
        '''
        GaussianSource.__init__(self, x0, sigma)
        AcousticSource.__init__(self, f)
        self.A = A

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        if value is not None:
            value = self.checkFloat('A', value)
            self.checkPositiveOrNull('A', value)
        self._A = value

    @property
    def xvar(self):
        return self.A

    @xvar.setter
    def xvar(self, value):
        self.A = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['x0', 'sigma', 'f', 'A']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{GaussianSource.__repr__(self)[:-1]}, {si_format(self.f, 1, space="")}Hz'
        if self.A is not None:
            s = f'{s}, {si_format(self.A, 1, space="")}Pa'
        return f'{s})'

    def computeDistributedAmps(self, fiber):
        return GaussianSource.computeDistributedAmps(self, fiber)

    def filecodes(self):
        codes = {**GaussianSource.filecodes(self), **AcousticSource.filecodes(self)}
        if self.A is not None:
            codes['A'] = f'{(self.A * self.conv_factor):.2f}kPa'
        return codes

    @property
    def quickcode(self):
        return '_'.join([
            AcousticSource.quickcode.fget(self),
            GaussianSource.quickcode.fget(self)
        ])

    def copy(self):
        return self.__class__(self.x0, self.sigma, self.f, A=self.A)

    @staticmethod
    def inputs():
        return {
            **GaussianSource.inputs(), **AcousticSource.inputs(),
            'A': {
                'desc': 'US pressure amplitude',
                'label': 'A',
                'unit': 'kPa',
                'factor': PA_TO_KPA,
                'precision': 2
            }
        }

    def computeMaxNodeAmp(self, fiber):
        return self.A


class PlanarDiskTransducerSource(ExtracellularSource, AcousticSource):
    ''' Acoustic source coming from a distant disk planar transducer.
        For now, acoustic propagation is only computed along the transducer normal axis.
        The rest of the field is computed assuming radial symmetry.
    '''

    conv_factor = 1e0
    source_density = 217e6  # points/m2
    min_focus = 1e-4    # m

    def __init__(self, x, f, u=None, rho=1e3, c=1500., r=2e-3, theta=0):
        ''' Initialization.

            :param u: particle velocity normal to the transducer surface (m/s)
            :param rho: medium density (kg/m3)
            :param c: medium speed of sound (m/s)
            :param r: transducer radius (m)
            :param theta: transducer angle of incidence (radians)
        '''
        self.rho = rho  # default value from Kyriakou 2015
        self.c = c      # default value from Kyriakou 2015
        self.r = r
        self.theta = theta
        self.u = u
        AcousticSource.__init__(self, f)
        ExtracellularSource.__init__(self, x)

    def __repr__(self):
        s = f'{ExtracellularSource.__repr__(self)[:-1]}, {si_format(self.f, 1, space="")}Hz'
        if self.u is not None:
            s = f'{s}, {si_format(self.u, 1, space="")}m/s'
        return f'{s})'

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if isinstance(value, tuple):
            value = list(value)
        if value[-1] == 'focus':
            value[-1] = self.getFocalDistance()
        value = tuple([self.checkFloat('x', v) for v in value])
        self._x = value

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        value = self.checkFloat('r', value)
        self.checkStrictlyPositive('r', value)
        self._r = value

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        value = self.checkFloat('theta', value)
        self._theta = value

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        value = self.checkFloat('rho', value)
        self.checkStrictlyPositive('rho', value)
        self._rho = value

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        value = self.checkFloat('c', value)
        self._c = value

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        if value is not None:
            value = self.checkFloat('u', value)
            self.checkPositiveOrNull('u', value)
        self._u = value

    @property
    def kf(self):
        ''' Angular wave number. '''
        return 2 * np.pi * self.f / self.c

    def copy(self):
        return self.__class__(self.x, self.f, self.u, rho=self.rho, c=self.c,
                              r=self.r, theta=self.theta)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['x', 'y', 'z', 'f', 'u', 'rho', 'c', 'r', 'theta']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    @staticmethod
    def inputs():
        return {
            **ExtracellularSource.inputs(),
            'r': {
                'desc': 'transducer radius',
                'label': 'r',
                'unit': 'm',
                'precision': 1
            },
            'theta': {
                'desc': 'transducer angle of incidence',
                'label': 'theta',
                'unit': 'rad',
                'precision': 2
            },
            'rho': {
                'desc': 'medium density',
                'label': 'rho',
                'unit': 'kg/m3',
                'precision': 1
            },
            'c': {
                'desc': 'medium speed of sound',
                'label': 'c',
                'unit': 'm/s',
                'precision': 1
            },
            **AcousticSource.inputs(),
            'u': {
                'desc': 'particle velocity normal to the transducer surface',
                'label': 'u',
                'unit': 'm/s',
                'precision': 1
            }
        }

    def filecodes(self):
        d = {
            **ExtracellularSource.filecodes(self),
            'r': f'{si_format(self.r, 2, space="")}{self.inputs()["r"]["unit"]}',
            'theta': f'{self.theta:.0f}{self.inputs()["theta"]["unit"]}',
            'rho': f'{self.rho:.0f}{self.inputs()["rho"]["unit"]}',
            'c': f'{self.c:.0f}{self.inputs()["c"]["unit"]}',
            **AcousticSource.filecodes(self)}
        if self.u is not None:
            d['u'] = f'{self.u:.0f}{self.inputs()["u"]["unit"]}'
        for k, v in d.items():
            d[k] = v.replace('/', '_per_')
        return d

    @property
    def quickcode(self):
        return '_'.join([
            AcousticSource.quickcode.fget(self),
            f'r{si_format(self.r, 2, space="")}{self.inputs()["r"]["unit"]}',
            ExtracellularSource.quickcode.fget(self)
        ])

    @property
    def xvar(self):
        return self.u

    @xvar.setter
    def xvar(self, value):
        self.u = value

    def distance(self, x):
        return np.linalg.norm(np.asarray(x) - np.asarray(self.xz))

    def getFocalDistance(self):
        ''' Get transducer focal distance, according to the equation of the normal axis amplitude.
        '''
        d = self.f * self.r**2 / self.c - self.c / (4 * self.f)
        return max(d, self.min_focus)

    def getFocalWidth(self, xmax, n):
        ''' Compute the width of the beam at the focal distance (-6 dB focal diameter),
            we assume the beam centered in z=0 and symmetric with x and y.

            param xmax: maximum x value of the interval used to search for the limit of the beam
            param n: number of evaluation points in the interval [0, xmax]
        '''

        # Get the pressure amplitudes in the focal zone along x
        xx = np.linspace(0, xmax, n)
        amps = self.DPSMxy(xx, np.array([0]), self.z - self.getFocalDistance())

        # Compute the conversion of the amplitudes into decibels
        ampsdB = 20 * np.log10(amps / amps[0])

        # Find the width of the x interval in which the reduction is less than 6 dB
        i = 0
        while ampsdB[i] > -6:
            i = i + 1
        x_6dB = xx[i] - (xx[i] - xx[i - 1]) * (ampsdB[i] + 6) / (ampsdB[i] - ampsdB[i - 1])
        return np.float(2 * x_6dB)

    def area(self):
        ''' Transducer surface area. '''
        return np.pi * self.r**2

    def relNormalAxisAmp(self, z):
        ''' Compute the relative acoustic amplitude at a given coordinate along
            the transducer normal axis.

            :param z: coordinate on transducer normal axis (m)
            :return: acoustic amplitude per particle velocity (Pa.s/m)
        '''
        deltaz = abs(z - self.z)
        j = complex(0, 1)  # imaginary number
        ez = np.exp(j * self.kf * deltaz)
        ezr = np.exp(j * self.kf * np.sqrt(deltaz**2 + self.r**2))
        return np.abs(self.rho * self.c * (ez - ezr))

    def normalAxisAmp(self, z):
        ''' Compute the acoustic amplitude at a given coordinate along the transducer normal axis.

            :param z: coordinate on transducer normal axis (m)
            :return: acoustic amplitude (Pa)
        '''
        return self.u * self.relNormalAxisAmp(z)

    def DPSM_squaredsources(self, m):
        return getCircle2DGrid(self.r, m, 'square')

    def DPSM_concentricsources(self, m):
        return getCircle2DGrid(self.r, m, 'concentric')

    def DPSM_sunflowersources(self, m, alpha=1):
        return getCircle2DGrid(r, m, 'sunflower')

    def DPSM_point(self, x, y, z, xsource, ysource, mact):
        ''' Compute acoustic amplitude in the point (x,z), given the transducer normal particle
            velocity and the distribution of point sources used to approximate the transducer.
            It follows the Distributed Point Source Method (DPSM) from Yamada 2009 (eq. 15).

            :param x: x coordinate of the point for which compute the acustic amplitude (m)
            :param z: z coordinate of the point for which compute the acustic amplitude (m)
            :param xsource: x coordinates of the point sources (m)
            :param ysource: y coordinates of the point sources, y perpendicular to x
                and parallel to the transducer surface (m)
            :param meff: number of point sources actually used
            :return: acoustic amplitude (Pa)
        '''
        j = complex(0, 1)        # imaginary number
        ds = self.area() / mact  # surface associated at each point source
        deltax = xsource + self.x[0] * np.ones(mact) - x * np.ones(mact)
        deltay = ysource + self.y * np.ones(mact) - y * np.ones(mact)
        deltaz = (self.z - z) * np.ones(mact)
        # distances of the point (x,z) to the point sources on the transducer surface
        distances = np.sqrt(deltax**2 + deltay**2 + deltaz**2)
        exp_sum = sum(np.exp(j * self.kf * distances) / distances)
        return np.abs(-j * self.rho * self.f * ds * self.u * exp_sum)

    def DPSM2d(self, x, z, m=None, d='concentric'):
        ''' Compute acoustic amplitude in the 2D space xz, given the transducer normal particle
            velocity and the transducer approximation to use.
            It follows the Distributed Point Source Method (DPSM) from Yamada 2009 (eq. 15).

            :param x: axis parallel to a fixed diameter of the transducer (m)
            :param z: transducer normal axis (m)
            :param m: number of point sources we want to use to approximate the transducer surface
            :param d: type of point sources distribution used
            :return: acoustic amplitude matrix (Pa)
        '''
        if m is None:
            m = int(np.ceil(self.source_density * self.area()))
        nx = len(x)
        nz = len(z)
        results = np.zeros((nx, nz))
        DPSM_method = {
            'sunflower': self.DPSM_sunflowersources,
            'squared': self.DPSM_squaredsources,
            'concentric': self.DPSM_concentricsources
        }[d]
        xsource, ysource = DPSM_method(m)
        mact = len(xsource)
        for i in range(nx):
            for k in range(nz):
                results[i, k] = self.DPSM_point(x[i], 0, z[k], xsource, ysource, mact)
        return results

    def DPSMxy(self, x, y, z, m=None, d='concentric'):
        ''' Compute acoustic amplitude in the 2D space xz, given the transducer normal particle
            velocity and the transducer approximation to use.
            It follows the Distributed Point Source Method (DPSM) from Yamada 2009 (eq. 15).

            :param x: axis parallel to a fixed diameter of the transducer (m)
            :param y: axis parallel to the transducer and perpendiculr to x (m)
            :param z: transducer normal axis (m)
            :param m: number of point sources we want to use to approximate the transducer surface
            :param d: type of point sources distribution used
            :return: acoustic amplitude matrix (Pa)
        '''
        if m is None:
            m = int(np.ceil(self.source_density * self.area()))

        nx = len(x)
        ny = len(y)
        results = np.zeros((nx, ny))
        if d == 'sunflower':
            xsource, ysource = self.DPSM_sunflowersources(m)
        if d == 'squared':
            xsource, ysource = self.DPSM_squaredsources(m)
        if d == 'concentric':
            xsource, ysource = self.DPSM_concentricsources(m)
        mact = len(xsource)
        for i in range(nx):
            for j in range(ny):
                results[i, j] = self.DPSM_point(x[i], y[j], z, xsource, ysource, mact)
        return results

    def computeDistributedAmps(self, fiber):
        ''' Compute acoustic amplitude value at all fiber nodes, given
            a transducer normal particle velocity.

            :param fiber: fiber model object
            :return: vector of acoustic amplitude at the nodes (Pa)
        '''
        # Get fiber nodes coordinates
        node_xcoords = fiber.getNodeCoords()
        node_coords = np.array([node_xcoords, np.zeros(fiber.nnodes)])

        # Rotate around source incident angle
        node_coords = rotAroundPoint2D(node_coords, self.theta, self.xz)

        # Compute amplitudes
        node_amps = self.DPSM2d(node_xcoords, np.array([0]))
        return node_amps.ravel()   # Pa

    def computeMaxNodeAmp(self, fiber):
        return max(self.computeDistributedAmps(fiber))  # Pa

    def computeSourceAmp(self, fiber, A):
        ''' Compute transducer particle velocity amplitude from target acoustic amplitude
            felt by the closest fiber node along the normal axis.

            :param fiber: fiber model object
            :param A: target acoustic amplitude (Pa)
            :return: particle velocity normal to the transducer surface.
        '''

        # Compute the particle velocity needed to generate the acoustic amplitude value at this node
        return A / self.relNormalAxisAmp(0.)  # m/s


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
