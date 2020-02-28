# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-23 09:43:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-28 17:50:43

import abc
import numpy as np
from scipy.optimize import brentq

from PySONIC.utils import si_format, rotAroundPoint2D, StimObject, gaussian
from PySONIC.core.drives import *
from .grids import getCircle2DGrid


class Source(StimObject):

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @abc.abstractmethod
    def computeNodesAmps(self, fiber):
        raise NotImplementedError

    @abc.abstractmethod
    def computeSourceAmp(self, fiber):
        raise NotImplementedError

    @property
    def is_searchable(self):
        return True


class XSource(Source):

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


class NodeSource(XSource):

    def __init__(self, inode):
        ''' Constructor.

            :param inode: node index
        '''
        self.inode = inode

    @property
    def inode(self):
        return self._inode

    @inode.setter
    def inode(self, value):
        value = self.checkInt('inode', value)
        self.checkPositiveOrNull('inode', value)
        self._inode = value

    @staticmethod
    def inputs():
        return {
            'inode': {
                'desc': 'node index',
                'label': 'i_node',
                'unit': ''
            }
        }

    def __repr__(self):
        return f'{self.__class__.__name__}(node {self.inode})'

    def filecodes(self):
        return {'inode': f'node{self.inode}'}

    @property
    def quickcode(self):
        return f'node{self.inode}'

    def computeNodesAmps(self, fiber):
        amps = np.zeros(fiber.nnodes)
        amps[self.inode] = self.xvar
        return amps

    def computeSourceAmp(self, fiber, A):
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
                'factor': 1e3,
                'precision': 1
            },
            'sigma': {
                'desc': 'width',
                'label': 'sigma',
                'unit': 'mm',
                'factor': 1e3,
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
            codes[key] = f'{getattr(self, key) * d.get("factor", 1.):.1f}{d["unit"]}'
        return codes

    @property
    def quickcode(self):
        return f'sigma{self.sigma * 1e3:.3f}mm'

    def computeNodesAmps(self, fiber):
        xcoords = fiber.getNodeCoords()  # m
        amps = gaussian(xcoords, mu=self.x0, sigma=self.sigma, A=self.xvar)
        return amps

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
        if len(self.x) <= 2:
            raise ValueError('Cannot return y-component for a 2D position (XZ)')
        return self._x[1]

    @property
    def z(self):
        return self._x[-1]

    @property
    def xz(self):
        return (self._x[0], self._x[-1])

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
                'factor': 1e3,
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
        return self.zstr

    def distance(self, x):
        return np.linalg.norm(np.asarray(x) - np.asarray(self.x))

    def distances(self, fiber):
        return np.array([self.distance((item, 0.)) for item in fiber.getNodeCoords()])  # m

    def getMinNodeDistance(self, fiber):
        return min(self.distances(fiber))  # m


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
        return f'{si_format(self.I, 1)}{self.inputs()["I"]["unit"]}'

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
                'factor': 1e0,
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


class IntracellularCurrent(NodeSource, CurrentSource):

    conv_factor = 1e9  # A to nA

    def __init__(self, inode, I=None, mode='anode'):
        NodeSource.__init__(self, inode)
        CurrentSource.__init__(self, I, mode)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['inode', 'mode', 'I']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{NodeSource.__repr__(self)[:-1]}, {self.mode}'
        if self.I is not None:
            s = f'{s}, {self.Istr()}'
        return f'{s})'

    def computeNodesAmps(self, fiber):
        return NodeSource.computeNodesAmps(self, fiber) * self.conv_factor

    def filecodes(self):
        return {**NodeSource.filecodes(self), **CurrentSource.filecodes(self)}

    def copy(self):
        return self.__class__(self.inode, self.I, mode=self.mode)

    @staticmethod
    def inputs():
        return {**NodeSource.inputs(), **CurrentSource.inputs()}


class ExtracellularCurrent(ExtracellularSource, CurrentSource):

    conv_factor = 1e3  # A to mA

    def __init__(self, x, I=None, mode='cathode', rho=300.0):
        ''' Initialization.

            :param rho: extracellular medium resistivity (Ohm.cm)
        '''
        ExtracellularSource.__init__(self, x)
        CurrentSource.__init__(self, I, mode)
        self.rho = rho

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        value = self.checkFloat('rho', value)
        self.checkStrictlyPositive('rho', value)
        self._rho = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in ['x', 'mode', 'I', 'rho']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{ExtracellularSource.__repr__(self)[:-1]}, {self.mode}'
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
                'factor': 1e0,
                'precision': 1
            }
        }

    def filecodes(self):
        return {
            **ExtracellularSource.filecodes(self), **CurrentSource.filecodes(self),
            'rho': f'{self.rho:.0f}{self.inputs()["rho"]["unit"]}'
        }


    def Vext(self, I, r):
        ''' Compute the extracellular electric potential generated by a given point-current source
            at a given distance in a homogenous, isotropic medium.

            :param I: point-source current amplitude (A)
            :param r: euclidian distance(s) between the source and the point(s) of interest (m)
            :return: extracellular potential(s) (mV)
        '''
        return self.rho * I / (4 * np.pi * r) * 1e1  # mV

    def Iext(self, V, r):
        ''' Compute the point-current source amplitude that generates a given extracellular
            electric potential at a given distance in a homogenous, isotropic medium.

            :param V: electric potential (mV)
            :param r: euclidian distance(s) between the source and the point(s) of interest (m)
            :return: point-source current amplitude (A)
        '''
        return 4 * np.pi * r * V / self.rho * 1e-1  # mV

    def computeNodesAmps(self, fiber):
        ''' Compute extracellular potential value at all fiber nodes. '''
        return self.Vext(self.I, self.distances(fiber))  # mV

    def computeSourceAmp(self, fiber, Ve):
        ''' Compute the current needed to generate the extracellular potential value at closest node. '''
        return self.Iext(Ve, self.getMinNodeDistance(fiber))  # A


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
                'factor': 1e0,
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

    def computeNodesAmps(self, fiber):
        return GaussianSource.computeNodesAmps(self, fiber) * self.conv_factor

    def filecodes(self):
        return {**GaussianSource.filecodes(self), **VoltageSource.filecodes(self)}

    @staticmethod
    def inputs():
        return {**GaussianSource.inputs(), **VoltageSource.inputs()}

    def computeMaxNodeAmp(self, fiber):
        return self.Ve


class AcousticSource(XSource):

    conv_factor = 1e-3  # Pa to kPa

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
                'factor': 1e-3,
                'precision': 0
            }
        }

    def filecodes(self):
        return {'f': f'{self.f * 1e-3:.0f}kHz'}

    @property
    def quickcode(self):
        return f'f{si_format(self.f, 0, space="")}{self.inputs()["f"]["unit"]}'


class NodeAcousticSource(NodeSource, AcousticSource):

    def __init__(self, inode, f, A=None):
        ''' Constructor.

            :param A: Acoustic amplitude (Pa)
        '''
        NodeSource.__init__(self, inode)
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
        for k in ['inode', 'f', 'A']:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        s = f'{NodeSource.__repr__(self)[:-1]}, {si_format(self.f, 1, space="")}Hz'
        if self.A is not None:
            s = f'{s}, {si_format(self.A, 1, space="")}Pa'
        return f'{s})'

    def computeNodesAmps(self, fiber):
        return NodeSource.computeNodesAmps(self, fiber) * self.conv_factor

    def filecodes(self):
        return {
            **NodeSource.filecodes(self), **AcousticSource.filecodes(self),
            'A': f'{(self.A * self.conv_factor):.2f}kPa'
        }

    @property
    def quickcode(self):
        return '_'.join([
            AcousticSource.quickcode.fget(self),
            NodeSource.quickcode.fget(self)
        ])

    def copy(self):
        return self.__class__(self.inode, self.f, A=self.A)

    @staticmethod
    def inputs():
        return {
            **NodeSource.inputs(), **AcousticSource.inputs(),
            'A': {
                'desc': 'US pressure amplitude',
                'label': 'A',
                'unit': 'kPa',
                'factor': 1e-3,
                'precision': 2
            }
        }

    def computeNodesAmps(self, fiber):
        return NodeSource.computeNodesAmps(self, fiber)

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

    def computeNodesAmps(self, fiber):
        return GaussianSource.computeNodesAmps(self, fiber) * self.conv_factor

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
                'factor': 1e-3,
                'precision': 2
            }
        }

    def computeNodesAmps(self, fiber):
        return GaussianSource.computeNodesAmps(self, fiber)

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
                'factor': 1e0,
                'precision': 1
            },
            'theta': {
                'desc': 'transducer angle of incidence',
                'label': 'theta',
                'unit': 'rad',
                'factor': 1e0,
                'precision': 2
            },
            'rho': {
                'desc': 'medium density',
                'label': 'rho',
                'unit': 'kg/m3',
                'factor': 1e0,
                'precision': 1
            },
            'c': {
                'desc': 'medium speed of sound',
                'label': 'c',
                'unit': 'm/s',
                'factor': 1e0,
                'precision': 1
            },
            **AcousticSource.inputs(),
            'u': {
                'desc': 'particle velocity normal to the transducer surface',
                'label': 'u',
                'unit': 'm/s',
                'factor': 1e0,
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
        ''' Get transducer focal distance. '''
        d = self.f * self.r**2 / self.c - self.c / (4 * self.f)
        return max(d, self.min_focus)

    def area(self):
        ''' Transducer surface area. '''
        return np.pi * self.r**2

    def relNormalAxisAmp(self, z):
        ''' Compute the relative acoustic amplitude at a given coordinate along the transducer normal axis.

            :param z: coordinate on transducer normal axis (m)
            :return: acoustic amplitude per particle velocity (Pa.s/m)
        '''
        deltaz = z - self.z
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
            :param ysource: y coordinates of the point sources, y perpendicular to x and parallel to the transducer surface (m)
            :param meff: number of point sources actually used
            :return: acoustic amplitude (Pa)
        '''
        j = complex(0, 1)                          # imaginary number
        ds = self.area() / mact              # surface associated at each point source
        deltax = xsource + self.x[0] * np.ones(mact) - x * np.ones(mact)
        deltay = ysource + self.y * np.ones(mact) - y * np.ones(mact)
        deltaz = (self.z - z) * np.ones(mact)
        distance = np.sqrt(deltax**2 + deltay**2 + deltaz**2)      # distances of the point (x,z) to the point sources on the transducer surface
        return np.abs(- j * self.rho * self.f * ds * self.u * sum( np.exp(j * self.kf * distance) / distance))

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

    def computeNodesAmps(self, fiber):
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
        return max(self.computeNodesAmps(fiber))  # Pa

    def computeSourceAmp(self, fiber, A):
        ''' Compute transducer particle velocity amplitude from target acoustic amplitude
            felt by the closest fiber node along the normal axis.

            :param fiber: fiber model object
            :param A: target acoustic amplitude (Pa)
            :return: particle velocity normal to the transducer surface.
        '''

        # Compute the particle velocity needed to generate the acoustic amplitude value at this node
        return A / self.relNormalAxisAmp(self.getMinNodeDistance(fiber))  # m/s


class SourceArray:

    def __init__(self, psources, rel_amps):
        if len(rel_amps) != len(psources):
            raise ValueError('number of point-sources does not match number of relative amplitudes')
        self.rel_amps = rel_amps
        self.psources = psources

    def strAmp(self, A):
        return ', '.join([p.strAmp(A * r) for p, r in zip(self.psources, self.rel_amps)])

    def computeNodesAmps(self, fiber, A):
        amps = np.array([
            p.computeNodesAmps(fiber, A * r) for p, r in zip(self.psources, self.rel_amps)])
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
        Asource = brentq(lambda x: self.computeNodesAmps(fiber, x).max() - A, Amin, Amax, xtol=1e-16)
        return Asource

    def filecodes(self, A):
        keys = self.psources[0].filecodes(A)
        return {key: '_'.join([p.filecodes(A * r)[key] for p, r in zip(self.psources, self.rel_amps)])
                for key in keys}

    def strPos(self):
        return '({})'.format(', '.join([p.strPos() for p in self.psources]))