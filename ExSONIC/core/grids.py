# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-04 21:24:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-04 22:25:16

import abc
import numpy as np


class CircleGrid(metaclass=abc.ABCMeta):
    ''' Generic interface to create populations of uniformly distributed data points
        within a 2D circle with specifc radius.
    '''

    def __init__(self, r, m):
        ''' Constructor.

            :param r: radius
            :param m: number of data points
        '''
        self.r = r
        self.m = m

    def area(self):
        return np.pi * self.r**2

    def areaPerPoint(self):
        ''' Area associated at each point. '''
        return self.area() / self.m

    @abc.abstractmethod
    def generate():
        ''' Generate 2D grid

            :return: 2D matrix of Cartesian (x, y) positions
        '''
        raise NotImplementedError


class CircleGridSquare(CircleGrid):
    ''' Squared disposition. '''

    rel_offset = 1e-3

    def lateralExtentPerPoint(self):
        ''' Side of the square associated at each point. '''
        return np.sqrt(self.areaPerPoint())

    @property
    def rel_ymax(self):
        ''' Maximal relative y deviation from grid center. '''
        s = self.lateralExtentPerPoint()
        return round(self.r / s - 1 / 2)

    def rel_xmax(self, y):
        ''' Maximal relative x deviation from grid center at height y. '''
        s = self.lateralExtentPerPoint()
        return round(np.sqrt(self.r**2 - y**2) / s - 1 / 2)

    def generate(self):
        xs = []
        ys = []
        s = np.sqrt(self.areaPerPoint())  # side of the square associated at each point source
        y = -self.rel_ymax * s  # initial y value
        while y <= (self.rel_ymax + self.rel_offset) * s:
            x = -self.rel_xmax(y) * s  # initial x value for every y iteration
            while x <= (self.rel_xmax(y) + self.rel_offset) * s:
                xs.append(x)
                ys.append(y)
                x = x + s  # shift the x component
            y = y + s  # shift the y component
        return np.array(xs), np.array(ys)


class CircleGridConcentric(CircleGrid):
    ''' Concentric disposition. '''

    def nlayers(self):
        ''' # Number of concentric layers. '''
        num = 3 * np.pi - 1 + np.sqrt(9 * np.pi**2 - 14 * np.pi + 1 + 4 * np.pi * self.m)
        return np.int(num / (2 * np.pi))

    def generate(self):
        radius = [0]
        angle = [0]
        nl = self.nlayers()      # number of concentric layers
        d = self.r / (nl - 1/2)  # radial distance between layers
        a = [0, 0]
        for i in range(nl - 1):
            ml = round(2 * np.pi * (i + 1))      # number of point sources in the layer
            rl = (i + 1) * d                     # radius of the concentric layer
            r = rl * np.ones(ml)
            a = (a[0] + a[1]) / 2 + 2 * np.pi * np.arange(ml) / ml
            radius = np.concatenate((radius, r), axis=None)    # point sources radius array
            angle = np.concatenate((angle, a), axis=None)      # point sources angle array
        return radius * np.cos(angle), radius * np.sin(angle)


class CircleGridSunflower(CircleGrid):
    ''' Sunflower disposition. '''

    def generate(self, alpha=1):
        nbounds = np.round(alpha * np.sqrt(self.m))    # number of boundary points
        phi = (np.sqrt(5) + 1) / 2                     # golden ratio
        k = np.arange(1, self.m + 1)                   # index array
        theta = 2 * np.pi * k / phi**2                 # angle array
        r = np.sqrt((k - 1) / (self.m - nbounds - 1))  # radius array
        r[r > 1] = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return self.r * np.vstack((x, y))


def getCircle2DGrid(r, m, dist_type):
    grid_classes = {
        'square': CircleGridSquare,
        'concentric': CircleGridConcentric,
        'sunflower': CircleGridSunflower
    }
    try:
        return grid_classes[dist_type](r, m).generate()
    except KeyError as err:
        raise ValueError(f'Unknown distribution (available dispositions are: {list(grid_classses.keys())}')
