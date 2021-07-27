# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-21 15:52:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-27 17:49:00

import matplotlib.pyplot as plt
from matplotlib.path import Path

from .bundle import *


class Nerve(Bundle):
    ''' Nerve object with constituent fascicles. '''

    def __init__(self, contours, length, fascicles=None, f_contours=None, f_kwargs=None):
        '''
            :param contours: set of sorted 2D points describing the nerve contours (m)
            :param length: length of the nerve section (m)
            :param fasciles: fascicles dictionary
            :param f_contours: dictionary of contours per fascicle
            :param f_kwargs: dictionary of intialization keyword arguments per fascicle
        '''
        self.contours = contours
        self.length = length
        if fascicles is not None:
            self.fascicles = fascicles
        else:
            for k, v in f_contours.items():
                if not self.contour_path.contains_path(Path(v)):
                    raise ValueError(f'fascicle {k} extends outside nerve')
            self.fascicles = {
                k: Bundle(v, length, **f_kwargs[k]) for k, v in f_contours.items()}

    def plotDiameterDistribution(self):
        fig, axes = plt.subplots(len(self.fascicles), 1)
        fig.suptitle('diameter distribution')
        fig.supxlabel('diameter (um)')
        fig.supylabel('frequency')
        for ax, (k, v) in zip(axes, self.fascicles.items()):
            ax.set_title(k)
            for sk in ['top', 'right']:
                ax.spines[sk].set_visible(False)
            v.plotDiameterDistribution(ax=ax)
        return fig

    def plotCrossSection(self, unit='um'):
        '''2D plot of complete nerve'''
        factor = {'um': 1e6, 'mm': 1e3, 'm': 1e0}[unit]
        fig, ax = plt.subplots()
        ax.set_title('nerve cross section')
        ax.set_xlabel('y (um)')
        ax.set_ylabel('z (um)')
        ax.set_aspect(1.)
        for k, v in self.fascicles.items():
            v.plotCrossSection(ax=ax, unit=unit)
        ax.add_patch(Polygon(self.contours * factor, closed=True, fc='none', ec='k'))
        ax.add_patch(Polygon(self.contours * factor, closed=True, ec='none', fc='gray', alpha=0.1))
        return fig

    def toDict(self):
        return {
            'contours': self.contours,
            'length': self.length,
            'fascicles': {k: v.toDict() for k, v in self.fascicles.items()}
        }

    @classmethod
    def fromDict(cls, d):
        return cls(
            d['contours'], d['length'],
            {k: Bundle.fromDict(v) for k, v in d['fascicles'].items()}
        )
