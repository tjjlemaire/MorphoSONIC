# -*- coding: utf-8 -*-
# @Author: Andy Bonnetto
# @Email: andy.bonnetto@epfl.ch
# @Date:   2021-05-21 08:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-22 14:46:27

from tqdm import tqdm
import random
import pickle
import numpy as np
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
from neuron import h

from PySONIC.utils import logger, TqdmHandler, my_log_formatter, setHandler, si_format
from ..models import SennFiber, UnmyelinatedFiber, getModel
from ..core import getAllSecs


def circleContour(r, n=10, closed=False):
    ''' Get a set of contour points for a circle.

        :param r: circle radius
        :param n: number of points
        :param closed: bollean indicating whether the resulting path must be closed
        :return: n x 2 matrix of contour coordinates
    '''
    if not closed:
        n += 1
    theta = np.linspace(0, 2 * np.pi, n)
    if not closed:
        theta = theta[:-1]
    return np.vstack((r * np.cos(theta), r * np.sin(theta))).T


def getXYBounds(pts):
    ''' Find the bounds of a 2-dimensional set of points along each dimension.

        :param pts: list of 2D points
        :return: (xmin, xmax), (ymin, ymax) list of bounds
    '''
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    return np.array([[xmin, xmax], [ymin, ymax]])


class Bundle:
    ''' Bundle with myelinated and unmyelinated fibers. '''

    MAX_PRATIO = 0.55  # packing ratio upper limit
    MAX_NTRIALS = 500  # maximum number of fiber placing trials
    MIN_INTERSPACE = 5e-8  # minimum space between fibers (m)

    def __init__(self, contours, length, fibers=None, fiberD_hists=None,
                 pratio=0.3, un_to_my_ratio=2.45, **fiber_kwargs):
        ''' Initialization.
            :param contours: set of sorted 2D points describing the contour coordinates (in m)
            :param length: bundle length (m)
            :param fibers: list of (position, fiber kernel) tuples
            :param fiberD_hists: reference histograms of fiber diameter distributions per fiber type
            :param pratio: overall fiber packing ratio in the bundle
            :param un_to_my_ratio: ratio of unmyelinated to myelinated fibers in the bundle
            :param fiber_kwargs: common keyword arguments used to initialize fibers
        '''
        # Initialize attributes
        self.contours = contours
        self.length = length
        self.fiberD_hists = fiberD_hists
        self.target_pratio = pratio
        self.target_un_to_my_ratio = un_to_my_ratio
        self.fiber_kwargs = fiber_kwargs
        # Assign fibers if provided
        if fibers is not None:
            self.fibers = fibers
        else:
            # Otherwise populate the bundle
            self.populate()

    def __repr__(self):
        s = f'{self.__class__.__name__}(A = {self.area * 1e6:.2f}mm2, L = {si_format(self.length)}m'
        nfibers = {
            'myelinated': len(self.myelinated_fibers),
            'unmyelinated': len(self.unmyelinated_fibers),
            'total': len(self.fibers)
        }
        nfibers_str = ', '.join([f'n_{k} = {v}' for k, v in nfibers.items()])
        s = f'{s}, {nfibers_str}'
        return f'{s})'

    @property
    def contours(self):
        return self._contours

    @contours.setter
    def contours(self, value):
        self._contours = value
        self.contour_path = Path(self.contours)
        self.xy_bounds = getXYBounds(self.contours)

    @property
    def fiberD_hists(self):
        return self._fiberD_hists

    @fiberD_hists.setter
    def fiberD_hists(self, value):
        if value is not None:
            self.fiberD_rv_samplers = {k: rv_histogram(v) for k, v in value.items()}
        self._fiberD_hists = value

    @property
    def target_pratio(self):
        return self._target_pratio

    @target_pratio.setter
    def target_pratio(self, value):
        if value is not None:
            if value <= 0:
                raise ValueError('packing ratio must be positive')
            if value > self.MAX_PRATIO:
                raise ValueError(f'The target packing ratio must not exceed {self.MAX_PRATIO}')
        self._target_pratio = value

    @property
    def area(self):
        ''' Calculate cross-section area (in m2) using the shoelace formula '''
        x, y = self.contours.T
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @property
    def pratio(self):
        fiberDs = np.array([fk[0].fiberD for fk in self.fibers])
        return np.sum(np.pi * fiberDs**2 / 4) / self.area

    @property
    def density(self):
        return len(self.fibers) / self.area  # fibers/m^2

    @property
    def center(self):
        ''' Find cross-section center for relative positioning '''
        # Find extrema coordinates along each dimension
        maxs = np.max(self.contours, axis=0)
        mins = np.min(self.contours, axis=0)
        # Define center as mid-point of extrema coorindates
        return maxs - (maxs - mins) / 2

    @property
    def diameter(self):
        # Compute distance from center to every contour coordinate
        distances = np.linalg.norm([x - self.center for x in self.contours], axis=1)
        # Set diameter as twice the max distance
        return 2 * np.max(distances)

    def sampleFiberDiameter(self, is_myelinated):
        ''' Sample diameters from a gaussian distribution for both
            myelinated and unmyelinated fibers
        '''
        k = {True: 'MY', False: 'UN'}[is_myelinated]
        return self.fiberD_rv_samplers[k].rvs()

    def getFiberKernel(self, is_myelinated, fiberD=None):
        ''' Create a fiber "kernel" (i.e. with set parameters but no constructed sections)
            for a specific fiber type and diameter.

            :param is_myelinated: boolean stating whether the fiber is myelinated
            :param fiberD: required fiber diameter
        '''
        if fiberD is None:
            fiberD = self.sampleFiberDiameter(is_myelinated)
        fclass = {True: SennFiber, False: UnmyelinatedFiber}[is_myelinated]
        return fclass(fiberD, fiberL=self.length, construct=False, **self.fiber_kwargs)

    def sampleFibers(self):
        ''' Get a list of fiber "kernels" that approaches the target packing ratio
            and fiber type ratio.
        '''
        logger.info('Creating fiber collection')
        # Initialize parameters
        nMY = 0
        pratio = 0.
        next_myelinated = True
        # Populate "kernels" list until target packing ratio is reached
        fkernels = []
        while pratio < self.target_pratio:
            # Sample fiber from appropriate type and add it to kernel list
            fkernel = self.getFiberKernel(next_myelinated)
            fkernels.append(fkernel)
            # Update process parameters and determine next fiber type
            nMY += int(next_myelinated)
            pratio += (np.pi * fkernel.fiberD**2 / 4) / self.area
            nUN = (len(fkernels) - nMY)
            next_myelinated = nUN / nMY > self.target_un_to_my_ratio
        assert len(fkernels) == nMY + nUN, 'fiber count not matching'
        # Assign kernel list sorted by decreasing fiber diameter
        self.virtual_fkernels = sorted(fkernels, key=lambda x: x.fiberD, reverse=True)

    def isInside(self, position, fiberD):
        ''' Check if position of the fiber is outside the faciscle or overlaps its boundary. '''
        # Generate collection of points outlinging the fiber contour
        fpoints = circleContour(fiberD, n=100) + position
        # Use matplotlib Path to find if points are inside the contour
        is_inside = self.contour_path.contains_points(fpoints)
        # Return true only if all points lie inside the contour path
        return np.all(is_inside)

    def randomYZ(self):
        ''' Get random YZ position within the bundle bounding limits. '''
        return random.uniform(*self.xy_bounds[0]), random.uniform(*self.xy_bounds[1])

    def randomPosition(self, fkernel):
        ''' Position fiber randomly in the bundle volume. '''
        dx = fkernel.node_to_node_L
        x = random.uniform(-dx / 2, dx / 2)
        yz = self.randomYZ()
        while not self.isInside(yz, fkernel.fiberD):
            yz = self.randomYZ()
        return np.array([x, *yz])

    def isOverlapping(self, fkernel, position):
        ''' Check if candidate fiber overlaps with any placed fiber. '''
        for fk, pos in self.fibers:
            np.linalg.norm(pos[1:] - position[1:])
            d_center_to_center = np.linalg.norm(pos[1:] - position[1:])
            d_edge_to_edge = d_center_to_center - 0.5 * (fkernel.fiberD + fk.fiberD)
            if d_edge_to_edge <= self.MIN_INTERSPACE:
                return True
        return False

    def placeFibers(self):
        # Place fibers within the bundle
        nfibers = len(self.virtual_fkernels)
        logger.info(f'Placing {nfibers} fibers...')
        self.fibers = []
        setHandler(logger, TqdmHandler(my_log_formatter))
        pbar = tqdm(total=nfibers)
        for fkernel in self.virtual_fkernels:
            pbar.update()
            is_overlapping = True
            count = 0
            # Repat random position assignment until a free spot is found
            while is_overlapping and count <= self.MAX_NTRIALS:
                xyz = self.randomPosition(fkernel)
                is_overlapping = self.isOverlapping(fkernel, xyz)
                count += 1
            # Raise error if not spot was found after a large number of trials
            if count == self.MAX_NTRIALS + 1:
                raise ValueError(
                    f'could not place fiber number {len(self.fibers)} (diameter = {fkernel.fiberD * 1e6:.2f} um')
            # Append fiber to fibers list
            self.fibers.append((fkernel, xyz))
        pbar.close()

    @property
    def myelinated_fibers(self):
        return list(filter(lambda x: x[0].is_myelinated, self.fibers))

    @property
    def unmyelinated_fibers(self):
        return list(filter(lambda x: not x[0].is_myelinated, self.fibers))

    def populate(self):
        ''' Generate fibers inside the bundle using a "brute force" algorithm. '''
        # Get fibers kernel list
        self.sampleFibers()
        self.placeFibers()
        nMY = len(self.myelinated_fibers)
        nUN = len(self.myelinated_fibers)
        logger.info(
            f'{self.__class__.__name__} populated with {len(self.fibers)} fibers ({nMY} MY. and {nUN} UN.)')
        logger.info(
            f'Packing ratio: {self.pratio:.2f} ({self.pratio / self.target_pratio * 1e2:.2f}% of target)')
        logger.info(
            f'UN:MY ratio: {nUN / nMY:.2f} ({(nUN / nMY) / self.target_un_to_my_ratio * 1e2:.2f}% of target)')
        logger.info(f'density: {self.density * 1e-6:.2f} fibers / mm^2')

    def plotDiameterDistribution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            for sk in ['top', 'right']:
                ax.spines[sk].set_visible(False)
            ax.set_title('diameter distribution')
            ax.set_xlabel('diameter (um)')
            ax.set_ylabel('frequency')
        else:
            fig = None
        is_myelinated = np.array([fk[0].is_myelinated for fk in self.fibers])
        fiberDs = np.array([fk[0].fiberD for fk in self.fibers])
        for b in [False, True]:
            valid_fibers = is_myelinated == b
            k = {True: 'MY', False: 'UN'}[b]
            ax.hist(fiberDs[valid_fibers] * 1e6, label=f'{k} (n = {sum(valid_fibers)})', bins=50)
        if fig is not None:
            ax.legend(frameon=False)
        return fig

    def plotCrossSection(self, unit='um', ax=None):
        ''' Plot bundle cross-section with contained fibers. '''
        factor = {'um': 1e6, 'mm': 1e3, 'm': 1e0}[unit]
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title('bundle cross section')
            ax.set_xlabel('y (um)')
            ax.set_ylabel('z (um)')
            ax.set_aspect(1.)
        else:
            fig = None
        ylist = []
        zlist = []
        if self.fibers is not None:
            for fkernel, xyz in self.fibers:
                _, y, z = xyz * factor  # um
                fc = {True: 'C1', False: 'C0'}[fkernel.is_myelinated]
                ax.add_patch(Circle((y, z), fkernel.fiberD / 2 * factor, fc=fc, ec='None'))
                ylist.append(y)
                zlist.append(z)
            ax.scatter(y, z, c='y', marker='')
        ax.add_patch(Polygon(self.contours * factor, closed=True, fill=False, color='k'))
        return fig

    def toDict(self):
        return {
            'contours': self.contours,
            'length': self.length,
            'fibers': [(x[0].meta, x[1]) for x in self.fibers]
        }

    @classmethod
    def fromDict(cls, d):
        fibers = [(getModel(x[0]), x[1]) for x in d['fibers']]
        return cls(d['contours'], d['length'], fibers=fibers)

    def toPickle(self, fpath):
        with open(fpath, 'wb') as fh:
            pickle.dump(self.toDict(), fh)

    @classmethod
    def fromPickle(cls, fpath):
        with open(fpath, 'rb') as fh:
            d = pickle.load(fh)
        return cls.fromDict(d)

    def simulate(self, source, pp):
        ''' Simulate each constituent fiber for a givne source and pulsing protocol.

            :param source: stimulation source object
            :param pp: pulsing protocol object
        '''
        output = []
        for i, (fiber, pos) in enumerate(self.fibers):
            fiber.construct()
            source.x0 = - pos[0]
            data, meta = fiber.simulate(source, pp)
            tspikes = fiber.getEndSpikeTrain(data)
            fiber.clear()
            h.allobjects()
            output.append(tspikes)
        return output

    def rasterPlot(self, output):
        fig, ax = plt.subplots()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('fiber index')
        ax.set_ylim(0, len(self.fibers))
        for i, ((fk, _), tspikes) in enumerate(zip(self.fibers, output)):
            c = {True: 'C1', False: 'C0'}[fk.is_myelinated]
            if tspikes is not None:
                ax.vlines(tspikes * 1e3, i, i + 1, colors=c)
        return fig
