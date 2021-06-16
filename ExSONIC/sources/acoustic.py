# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-23 09:43:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-16 11:42:54

import numpy as np

from PySONIC.utils import logger, rotAroundPoint2D
from PySONIC.core.drives import *
from PySONIC.core.batches import Batch

from ..grids import getCircle2DGrid
from ..constants import *
from .source import XSource, SectionSource, GaussianSource, ExtracellularSource


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
                'precision': 0
            }
        }


class AbstractAcousticSource(AcousticSource):

    key = 'A'

    def __init__(self, f, A=None):
        ''' Constructor.

            :param A: Acoustic amplitude (Pa)
        '''
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

    def copy(self):
        return self.__class__(self.f, A=self.A)

    @staticmethod
    def inputs():
        return {
            **AcousticSource.inputs(),
            'A': {
                'desc': 'US pressure amplitude',
                'label': 'A',
                'unit': 'Pa',
                'precision': 2
            }
        }

    def computeMaxNodeAmp(self, fiber):
        return self.A


class SectionAcousticSource(SectionSource, AbstractAcousticSource):

    def __init__(self, sec_id, f, A=None):
        SectionSource.__init__(self, sec_id)
        AbstractAcousticSource.__init__(self, f, A=A)

    def computeDistributedAmps(self, fiber):
        return SectionSource.computeDistributedAmps(self, fiber)

    def copy(self):
        return self.__class__(self.sec_id, self.f, A=self.A)

    @staticmethod
    def inputs():
        return {**SectionSource.inputs(), **AbstractAcousticSource.inputs()}


class GaussianAcousticSource(GaussianSource, AbstractAcousticSource):

    def __init__(self, x0, sigma, f, A=None):
        GaussianSource.__init__(self, x0, sigma)
        AbstractAcousticSource.__init__(self, f, A=A)

    def computeDistributedAmps(self, fiber):
        return GaussianSource.computeDistributedAmps(self, fiber)

    def copy(self):
        return self.__class__(self.x0, self.sigma, self.f, A=self.A)

    @staticmethod
    def inputs():
        return {**GaussianSource.inputs(), **AbstractAcousticSource.inputs()}


class PlanarDiskTransducerSource(ExtracellularSource, AcousticSource):
    ''' Acoustic source coming from a distant disk planar transducer.
        For now, acoustic propagation is only computed along the transducer normal axis.
        The rest of the field is computed assuming radial symmetry.
    '''

    conv_factor = 1e0
    source_density = 217e6  # points/m2
    min_focus = 1e-4    # m
    MAX_COMBS = int(2e7)  # max number of source-target combinations for DPSM computations

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

    @property
    def xvar(self):
        return self.u

    @xvar.setter
    def xvar(self, value):
        self.u = value

    def copy(self):
        return self.__class__(self.x, self.f, self.u, rho=self.rho, c=self.c,
                              r=self.r, theta=self.theta)

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
                'unit': 'g/m3',
                'factor': 1e3,
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

    def getXYSources(self, m=None, d='concentric'):
        ''' Wrapper around the method to obtain a source distribution for this transducer.

            :param m: number of point sources we want to use to approximate the transducer surface
            :param d: type of point sources distribution used
        '''
        if m is None:
            m = int(np.ceil(self.source_density * self.area()))
        return getCircle2DGrid(self.r, m, d)

    def getAcousticPressure(self, dmat):
        ''' Compute the complex acoustic pressure field using the Rayleigh-Sommerfeld integral,
            given a source-target distance matrix.

            :param d: (nsource x ntargets) distance matrix
            :return complex acoustic pressure
        '''
        ds = self.area() / dmat.shape[0]  # surface associated at each point source
        j = complex(0, 1)                 # imaginary number
        # Compute complex exponentials matrix
        expmat = np.exp(j * self.kf * dmat) / dmat
        # Sum along source dimension
        expsum = np.sum(expmat, axis=0)
        # Return RSI output
        return -j * self.rho * self.f * ds * self.u * expsum

    def DPSM_serialized(self, x, y, z, subcall=False, **kwargs):
        ''' Compute acoustic amplitude the points (x,z), given the distribution of
            point sources used to approximate the transducer. It follows the
            Distributed Point Source Method (DPSM) from Yamada 2009 (eq. 15).

            :param x: x coordinate of the point(s) for which compute the acoustic amplitude (m)
            :param y: y coordinate of the point(s) for which compute the acoustic amplitude (m)
            :param z: z coordinate of the point(s) for which compute the acoustic amplitude (m)
            :param m: number of point sources used
            :return: vector of complex acoustic amplitudes
        '''
        x, y, z = [np.atleast_1d(xx) for xx in [x, y, z]]
        if not x.size == y.size == z.size:
            raise ValueError('position vectors differ in size')

        # Get point sources
        xs, ys = self.getXYSources(**kwargs)

        # Ultimately system size is determined by number of source-target combinations
        npoints = x.size
        ncombs = npoints * xs.size
        # If number of combinations is too large, split work into different slices
        if ncombs > self.MAX_COMBS:
            # If asked to split during a subcall -> raise error
            if subcall:
                raise ValueError(
                    f'splitted work is too large ({npoints} points, i.e. {ncombs} combinations)')

            # Compute number of slices and number of jobs per slice
            nslices = ncombs // self.MAX_COMBS
            if ncombs % self.MAX_COMBS != 0:
                nslices += 1
            nperslice = npoints // nslices
            if npoints % nperslice != 0:
                nslices += 1
            logger.debug(
                f'Splitting {npoints} points job into {nslices} slices of {nperslice} points each')

            # Define batch function to call function on a slice
            def runSlice(i, inds, *args, **kwargs):
                logger.debug(
                    f'computing slice {i + 1} / {nslices} (indexes {inds.start} - {inds.stop - 1})')
                return self.DPSM_serialized(*args, subcall=True, **kwargs)

            # Run batch job to enable multiprocessing
            queue = []
            for i in range(nslices):
                inds = slice(i * nperslice, min((i + 1) * nperslice, npoints))
                queue.append([i, inds, x[inds], y[inds], z[inds]])
            queue = [(x, kwargs) for x in queue]
            batch = Batch(runSlice, queue)
            return np.hstack(batch.run(mpi=True, loglevel=logger.getEffectiveLevel()))

        # Get meshgrids for each dimension and compute multidimensional distance matrix
        X, XS = np.meshgrid(x, xs + self.x[0])
        Y, YS = np.meshgrid(y, ys + self.x[1])
        Z, ZS = np.meshgrid(z, np.ones_like(xs) * self.x[2])
        dmat = np.sqrt((XS - X)**2 + (YS - Y)**2 + (ZS - Z)**2)

        # Return complex acoustic pressure
        return self.getAcousticPressure(dmat)

    def DPSM(self, x, y, z, **kwargs):
        ''' Compute acoustic amplitude field for a collection of x, y and z coordinates.

            :param x: x-coordinates (m)
            :param y: y-coordinates (m)
            :param z: z-coordinates (m)
            :return: matrix of complex acoustic pressures
        '''
        x, y, z = [np.atleast_1d(xx) for xx in [x, y, z]]
        X, Y, Z = np.meshgrid(x, y, z)
        Pac = self.DPSM_serialized(X.flatten(), Y.flatten(), Z.flatten(), **kwargs)
        return np.squeeze(np.reshape(Pac, (x.size, y.size, z.size)))

    def DPSM_amps(self, *args, **kwargs):
        return np.abs(self.DPSM(*args, **kwargs))

    def DPSM_phases(self, *args, **kwargs):
        return np.angle(self.DPSM(*args, **kwargs))

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

        # Compute acoustic amplitudes
        node_amps = self.DPSM_amps(node_xcoords, 0., 0.)  # Pa
        return {'node': node_amps.ravel()}

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
