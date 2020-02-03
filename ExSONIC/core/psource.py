# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-23 09:43:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-20 14:54:09

import abc
import numpy as np
from scipy.optimize import brentq

from PySONIC.utils import si_format, rotAroundPoint2D


class PointSource(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def modality(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    # @property
    # @abc.abstractmethod
    # def attrkeys(self):
    #     ''' attributes to compare for object equality assessment. '''
    #     raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return sum([getattr(self, k) != getattr(other, k) for k in self.attrkeys]) == 0

    def strAmp(self, amp):
        return '{} = {}{}'.format(
            self.modality['name'],
            si_format(amp * self.modality.get('factor', 1.), precision=2),
            self.modality['unit']
        )

    @property
    @abc.abstractmethod
    def strPos(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def computeNodesAmps(self, fiber, A):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def computeSourceAmp(self, fiber, A):
        return NotImplementedError


class IntracellularPointSource(PointSource):

    def __init__(self, inode):
        self.inode = inode
        self.attrkeys = ['inode']

    def __repr__(self):
        return f'{self.__class__.__name__}({self.strPos()})'

    def computeNodesAmps(self, fiber, A):
        amps = np.zeros(fiber.nnodes)
        amps[self.inode] = A
        return amps

    def computeSourceAmp(self, fiber, A):
        return A

    def strPos(self):
        return f'node {self.inode}'


class ExtracellularPointSource(PointSource):

    def __init__(self, x):
        self.x = x
        self.attrkeys = ['x']

    def __repr__(self):
        return f'{self.__class__.__name__}(x = {self.strPos()})'

    def distance(self, x):
        return np.linalg.norm(np.asarray(x) - np.asarray(self.x))

    def distances(self, fiber):
        return np.array([self.distance((item, 0.)) for item in fiber.getNodeCoords()])  # m

    def strPos(self):
        x_mm = ['{:.1f}'.format(x * 1e3) for x in self.x]
        return '({})mm'.format(','.join(x_mm))

    def getMinNodeDistance(self, fiber):
        return min(self.distances(fiber))  # m


class CurrentPointSource(PointSource):

    modality = {'name': 'I', 'unit': 'A'}

    def __init__(self, mode):
        if mode == 'cathode':
            self.is_cathodal = True
        elif mode == 'anode':
            self.is_cathodal = False
        else:
            raise ValueError(f'Unknown polarity: {mode} (should be "cathode" or "anode")')
        self.attrkeys.append('is_cathodal')

    def isCathodal(self, amp):
        return amp <= 0


class ExtracellularCurrent(ExtracellularPointSource, CurrentPointSource):

    def __init__(self, x, mode='cathode', rho=300.0):
        ''' Initialization.

            :param rho: extracellular medium resistivity (Ohm.cm)
        '''
        self.rho = rho
        ExtracellularPointSource.__init__(self, x)
        CurrentPointSource.__init__(self, mode)
        self.attrkeys.append('rho')

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

    def computeNodesAmps(self, fiber, I):
        ''' Compute extracellular potential value at all fiber nodes. '''
        return self.Vext(I, self.distances(fiber))  # mV

    def computeSourceAmp(self, fiber, Ve):
        # Compute the current needed to generate the extracellular potential value at closest node
        return self.Iext(Ve, self.getMinNodeDistance(fiber))  # A

    def filecodes(self, A):
        return {
            'psource': f'ps{self.strPos()}',
            'A': f'{(A * 1e3):.2f}mA'
        }


class IntracellularCurrent(IntracellularPointSource, CurrentPointSource):

    conv_factor = 1e9  # A to nA

    def __init__(self, inode, mode='anode'):
        IntracellularPointSource.__init__(self, inode)
        CurrentPointSource.__init__(self, mode)

    def computeNodesAmps(self, fiber, A):
        return IntracellularPointSource.computeNodesAmps(self, fiber, A) * self.conv_factor

    def filecodes(self, A):
        return {
            'psource': f'ps(node{self.inode})',
            'A': f'{(A * self.conv_factor):.2f}nA'
        }


class AcousticPointSource(PointSource):

    modality = {'name': 'A', 'unit': 'Pa'}
    conv_factor = 1e-3  # Pa to kPa

    def __init__(self, Fdrive):
        self.Fdrive = Fdrive  # Hz

    def filecodes(self, A):
        return {
            'f': f'{self.Fdrive * 1e-3:.0f}kHz',
            'A': f'{(A * self.conv_factor):.2f}kPa'
        }


class NodeAcousticSource(IntracellularPointSource, AcousticPointSource):

    def __init__(self, inode, Fdrive):
        IntracellularPointSource.__init__(self, inode)
        AcousticPointSource.__init__(self, Fdrive)

    def __repr__(self):
        return f'{IntracellularPointSource.__repr__(self)[:-1]}, {self.Fdrive * 1e-3:.0f} kHz)'

    def computeNodesAmps(self, fiber, A):
        assert self.Fdrive == fiber.Fdrive, 'US frequency mismatch'
        return IntracellularPointSource.computeNodesAmps(self, fiber, A)

    def filecodes(self, A):
        return {**{'psource': f'ps(node{self.inode})'}, **super().filecodes(A)}

    def computeMaxNodeAmp(self, fiber, A):
        return A    

class PlanarDiskTransducerSource(ExtracellularPointSource):
    ''' Acoustic source coming from a distant disk planar transducer.
        For now, acoustic propagation is only computed along the transducer normal axis.
        The rest of the field is computed assuming radial symmetry.
    '''

    modality = {'name': 'u', 'unit': 'm/s'}
    conv_factor = 1e0
    source_density = 217e6  # points/m2
    min_focus = 1e-4    # m

    def __init__(self, x, y, z, Fdrive, rho=10e3, c=1500., r=2e-3, theta=0):
        ''' Initialization.

            :param rho: medium density (kg/m3)
            :param c: medium speed of sound (m/s)
            :param theta: transducer angle of incidence (radians)
            :param r: transducer radius (m)
        '''
        super().__init__(x)
        self.x = x   # m
        self.y = y   # m
        self.Fdrive = Fdrive  # Hz
        self.rho = rho   #defaul value from Kyriakou 2015
        self.c = c       #defaul value from Kyriakou 2015
        self.r = r
        self.theta = theta
        for k in ['rho', 'c', 'theta']:
            self.attrkeys.append(k)

        # Angular wave number
        self.kf = 2 * np.pi * self.Fdrive / self.c
        
        if z == 'focus':
            z = self.getFocus()
        self.z = z   # m
    
    def getFocus(self):
        return max(self.Fdrive * self.r**2 / self.c - self.c / (4 * self.Fdrive), self.min_focus)
        
    def area(self):
        return np.pi * self.r**2

    def relNormalAxisAmp(self, z):
        ''' Compute the relative acoustic amplitude at a given distance along the transducer normal axis.

            :param z: distance from transducer (m)
            :return: acoustic amplitude per particle velocity (Pa.s/m)
        '''
        j = complex(0, 1)  # imaginary number
        ez = np.exp(j * self.kf * z)
        ezr = np.exp(j * self.kf * np.sqrt(z**2 + self.r**2))
        return np.abs(self.rho * self.c * (ez - ezr))

    def normalAxisAmp(self, z, u):
        ''' Compute the acoustic amplitude at a given distance along the transducer normal axis.

            :param z: distance from transducer (m)
            :param u: particle velocity normal to the transducer surface (m/s)
            :return: acoustic amplitude (Pa)
        '''
        return u * self.relNormalAxisAmp(z)

    def DPSM_squaredsources(self, m):
        ''' Generate a population of uniformly distributed 2D data points
            in a circle with radius r with squared disposition.

            :param m: number of data points
            :return: 2D matrix of Cartesian (x, y) positions
        '''
        xs = []
        ys = []
        A = self.area() / m                     # area associated at each point source
        s = np.sqrt(A)                   # side of the square associated at each point source
        y = - round(self.r/s - 1/2) * s                                       # initial y value
        while y <= round(self.r/s -1/2) * s + s * 1e-3:
            x = - round( np.sqrt( self.r**2 - y**2)/s -1/2) * s               # initial x value for every y iteration
            while x <= round( np.sqrt( self.r**2 - y**2)/s -1/2) * s + s*1e-3:
                xs.append(x)
                ys.append(y)
                x = x + s                # shift the x component
            y = y + s                    # shift the y component
        xsource = np.array(xs)
        ysource = np.array(ys)
        return xsource, ysource

    def DPSM_concentricsources(self, m):
        ''' Generate a population of uniformly distributed 2D data points
            in a circle with radius r with concentric disposition.

            :param m: number of data points
            :return: 2D matrix of Cartesian (x, y) positions
        '''
        radius = [0]
        angle = [0]
        nl = np.int((3 * np.pi -1 + np.sqrt( 9*np.pi**2 - 14*np.pi + 1 +4*np.pi*m)) / (2 * np.pi))  # Number of concentric layers
        d = self.r / (nl - 1/2)                  # distance between layers
        a = [0, 0]
        for i in range(nl - 1):
            ml = round(2 * np.pi * (i + 1))      # number of point sources in the layer
            rl = (i + 1) * d                     # radius of the concentric layer
            r = rl * np.ones(ml)
            a = (a[0] + a[1]) / 2 + 2 * np.pi * np.arange(ml) / ml
            radius = np.concatenate((radius, r), axis=None)    # point sources radius array
            angle = np.concatenate((angle, a), axis=None)      # point sources angle array
        xsource = radius * np.cos(angle)         # x coordinates of point sources
        ysource = radius * np.sin(angle)         # y coordinates of point sources
        return xsource, ysource

    def DPSM_sunflowersources(self, m, alpha=1):
        ''' Generate a population of uniformly distributed 2D data points
            in a circle with radius r.

            :param m: number of data points
            :param alpha: coefficient determining evenness of the boundary
            :return: 2D matrix of Cartesian (x, y) positions
        '''
        nbounds = np.round(alpha * np.sqrt(m))    # number of boundary points
        phi = (np.sqrt(5) + 1) / 2                # golden ratio
        k = np.arange(1, m + 1)                   # index array
        theta = 2 * np.pi * k / phi**2            # angle array
        r = np.sqrt((k - 1) / (m - nbounds - 1))  # radius array
        r[r > 1] = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.r * np.vstack((x, y))

    def DPSM_linearsources(self, m):
        ''' Generate a population of uniformly distributed data points
            in a line of length r.

            :param m: number of data points
            :return: 2D matrix of Cartesian (x, y) positions
        '''
        xs = []
        ys = []
        s = self.r * 2 / m
        y = 0
        x = - round(np.sqrt(self.r**2) / s - 1 / 2) * s
        while x <= round(np.sqrt(self.r**2) / s -1 / 2) * s + s * 1e-3:
            xs.append(x)
            ys.append(y)
            x = x + s
        xsource = np.array(xs)
        ysource = np.array(ys)
        return xsource, ysource

    def DPSM_point(self, x, y, z, u, xsource, ysource, mact):
        ''' Compute acoustic amplitude in the point (x,z), given the transducer normal particle
        velocity and the distribution of point sources used to approximate the transducer.
        It follows the Distributed Point Source Method (DPSM) from Yamada 2009 (eq. 15).

            :param x: x coordinate of the point for which compute the acustic amplitude (m)
            :param z: z coordinate of the point for which compute the acustic amplitude (m)
            :param u: particle velocity normal to the transducer surface (m/s)
            :param xsource: x coordinates of the point sources (m)
            :param ysource: y coordinates of the point sources, y perpendicular to x and parallel to the transducer surface (m)
            :param meff: number of point sources actually used
            :return: acoustic amplitude (Pa)
        '''
        j = complex(0, 1)                          # imaginary number
        ds = self.area() / mact              # surface associated at each point source
        deltax = xsource + self.x * np.ones(mact) - x * np.ones(mact)
        deltay = ysource + self.y * np.ones(mact) - y * np.ones(mact)
        deltaz = (self.z - z) * np.ones(mact)
        distance = np.sqrt(deltax**2 + deltay**2 + deltaz**2)      # distances of the point (x,z) to the point sources on the transducer surface
        return np.abs(- j * self.rho * self.Fdrive * ds * u * sum( np.exp(j * self.kf * distance) / distance))

    def DPSM2d(self, x, z, u, m=None, d='concentric'):
        ''' Compute acoustic amplitude in the 2D space xz, given the transducer normal particle
        velocity and the transducer approximation to use.
        It follows the Distributed Point Source Method (DPSM) from Yamada 2009 (eq. 15).

            :param x: axis parallel to a fixed diameter of the transducer (m)
            :param z: transducer normal axis (m)
            :param u: particle velocity normal to the transducer surface (m/s)
            :param m: number of point sources we want to use to approximate the transducer surface
            :param d: type of point sources distribution used
            :return: acoustic amplitude matrix (Pa)
        '''
        if m is None:
            m = int(np.ceil(self.source_density * self.area()))
        # ref_dict[d][m][tuple(x.tolist())][tuple(z.tolist())]

        nx = len(x)
        nz = len(z)
        results = np.zeros((nx, nz))
        DPSM_method = {
            'sunflower': self.DPSM_sunflowersources,
            'squared': self.DPSM_squaredsources,
            'concentric': self.DPSM_concentricsources,
            'linear': self.DPSM_linearsources
        }[d]
        xsource, ysource = DPSM_method(m)
        mact = len(xsource)
        for i in range(nx):
            for k in range(nz):
                results[i, k] = self.DPSM_point(x[i], 0, z[k], u, xsource, ysource, mact)
        return results

    def DPSMxy(self, x, y, z, u, m=None, d='concentric'):
        ''' Compute acoustic amplitude in the 2D space xz, given the transducer normal particle
        velocity and the transducer approximation to use.
        It follows the Distributed Point Source Method (DPSM) from Yamada 2009 (eq. 15).

            :param x: axis parallel to a fixed diameter of the transducer (m)
            :param y: axis parallel to the transducer and perpendiculr to x (m)
            :param z: transducer normal axis (m)
            :param u: particle velocity normal to the transducer surface (m/s)
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
        if d == 'linear':
            xsource, ysource = self.DPSM_linearsources(m)
        mact = len(xsource)
        for i in range(nx):
            for j in range(ny):
                results[i, j] = self.DPSM_point (x[i], y[j], z, u, xsource, ysource, mact)
        return results

    def computeNodesAmps(self, fiber, u):
        ''' Compute acoustic amplitude value at all fiber nodes, given
            a transducer normal particle velocity.

            :param fiber: fiber model object
            :param u: particle velocity normal to the transducer surface.
            :return: vector of acoustic amplitude at the nodes (Pa)
        '''
        # Get fiber nodes coordinates
        node_coords = np.array([fiber.getNodeCoords(), np.zeros(fiber.nnodes)])
        node_xcoords = np.array(fiber.getNodeCoords())

        # Rotate around source incident angle
        node_coords = rotAroundPoint2D(node_coords, self.theta, self.x)

        # Compute amplitudes 
        node_amps = self.DPSM2d(node_xcoords, np.array([0]), u)
        return node_amps.ravel() # Pa

    def computeMaxNodeAmp(self, fiber, u):
        return max(self.computeNodesAmps(fiber, u)) # Pa

    def computeSourceAmp(self, fiber, A):
        ''' Compute transducer particle velocity amplitude from target acoustic amplitude
            felt by the closest fiber node along the normal axis.

            :param fiber: fiber model object
            :param A: target acoustic amplitude (Pa)
            :return: particle velocity normal to the transducer surface.
        '''

        # Compute the particle velocity needed to generate the acoustic amplitude value at this node
        return A / self.relNormalAxisAmp(self.getMinNodeDistance(fiber))  # m/s

    def filecodes(self, A):
        pos_mm = ','.join(['{:.1f}'.format(x * 1e3) for x in [self.x, self.y, self.z]])
        return {
            'psource': f'ps({pos_mm})mm',
            'f': f'{self.Fdrive * 1e-3:.0f}kHz',
            'A': f'{si_format(A*1e-3, 2)}kPa'
        }
        
    def strPos(self):
        pos_mm = ['{:.1f}'.format(x * 1e3) for x in [self.x, self.y, self.z]]
        return '({})mm'.format(','.join(pos_mm))


class PointSourceArray:

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