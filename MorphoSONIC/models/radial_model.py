# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-14 11:32:12

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import si_format, logger
from PySONIC.postpro import detectSpikes

from ..constants import *
from ..core import SpatiallyExtendedNeuronModel, addSonicFeatures


@addSonicFeatures
class RadialModel(SpatiallyExtendedNeuronModel):
    ''' Radially-symmetric model with a center and a periphery. '''

    simkey = 'radial_model'
    # gmax = 1e-2  # S/cm2
    # use_explicit_iax = True

    def __init__(self, pneuron, innerR, outerR, rs, depth=100e-9, **kwargs):
        self.pneuron = pneuron
        self.rs = rs          # Ohm.cm
        self.innerR = innerR  # m
        self.outerR = outerR  # m
        self.depth = depth      # m
        super().__init__(**kwargs)

    @property
    def innerR(self):
        return self._innerR

    @innerR.setter
    def innerR(self, value):
        if value <= 0:
            raise ValueError('inner radius must be positive')
        if hasattr(self, '_outerR') and value >= self.outerR:
            raise ValueError('inner radius must be smaller than outer radius')
        self.set('innerR', value)

    @property
    def outerR(self):
        return self._outerR

    @outerR.setter
    def outerR(self, value):
        if value <= 0:
            raise ValueError('outer radius must be positive')
        if hasattr(self, '_innerR') and value <= self.innerR:
            raise ValueError('outer radius must be greater than inner radius')
        self.set('outerR', value)

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        if value <= 0:
            raise ValueError('depth must be positive')
        self.set('depth', value)

    @property
    def meta(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'innerR': self.innerR,
            'outerR': self.outerR,
            'rs': self.rs,
            'depth': self.depth
        }

    @staticmethod
    def getMetaArgs(meta):
        args = [getPointNeuron(meta['neuron'])] + [meta[k] for k in ['innerR', 'outerR', 'rs']]
        kwargs = {'depth': meta['depth']}
        return args, kwargs

    @property
    def morpho_attrs(self):
        return {
            'innerR': f'{self.innerR * M_TO_NM:.1f}nm',
            'outerR': f'{self.outerR * M_TO_NM:.1f}nm',
            'depth': f'{self.depth * M_TO_NM:.0f}nm',
            'rs': f'{self.rs:.1e}Ohm.cm'
        }

    @property
    def modelcodes(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            **{k: f'{k} = {v}' for k, v in self.morpho_attrs.items()}
        }

    def __repr__(self):
        morpho_str = ', '.join([f'{k}{v}' for k, v in self.morpho_attrs.items()])
        return f'{self.__class__.__name__}({self.pneuron}, {morpho_str})'

    def clear(self):
        del self.center
        del self.periphery

    def createSections(self):
        ''' Create morphological sections. '''
        self.center = self.createSection(
            'center', mech=self.mechname, states=self.pneuron.statesNames())
        self.periphery = self.createSection(
            'periphery', mech=self.mechname, states=self.pneuron.statesNames())

    @property
    def sections(self):
        return {'nodes': {
            'center': self.center,
            'periphery': self.periphery
        }}

    @property
    def IDs(self):
        return list(self.sections['nodes'].keys())

    @property
    def refsection(self):
        return self.center

    @property
    def seclist(self):
        return [self.center, self.periphery]

    @property
    def nonlinear_sections(self):
        return self.sections['nodes']

    @staticmethod
    def translateRadialGeometry(depth, r1, r2):
        ''' Return geometrical parameters of cylindrical sections to match quantities of
            membrane and axial currents in a radial configuration between a central
            and a peripheral section.

            :param depth: depth of the radial sections (m)
            :param r1: radius of central section (m)
            :param r2: outer radius of peripheral section (m)
            :return: 3-tuple with sections common diameter and their respective lengths (m)
        '''
        logger.debug('radial geometry: depth = {}m, r1 = {}m, r2 = {}m'.format(
            *si_format([depth, r1, r2], 2)))
        d = np.power(4 * depth * r2**2 / np.log((r1 + r2) / r1), 1 / 3)  # m
        L1 = r1**2 / d  # m
        L2 = r2**2 / d - L1  # m
        logger.debug('equivalent linear geometry: d = {}m, L1 = {}m, L2 = {}m'.format(
            *si_format([d, L1, L2], 2)))
        return d, L1, L2

    def setGeometry(self):
        ''' Set sections geometry. '''
        d, L1, L2 = self.translateRadialGeometry(self.depth, self.innerR, self.outerR)
        self.center.setGeometry(d, L1)
        self.periphery.setGeometry(d, L2)

    def setResistivity(self):
        ''' Set sections axial resistivity. '''
        for sec in self.seclist:
            sec.setResistivity(self.rs)

    def setTopology(self):
        self.center.connect(self.periphery)

    @staticmethod
    def isExcited(data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        nspikes_periphery = detectSpikes(data['periphery'])[0].size
        return nspikes_periphery > 0


def surroundedSonophore(pneuron, a, fs, *args, **kwargs):
    model = RadialModel(pneuron, a, a / np.sqrt(fs), *args, **kwargs)
    model.a = a
    return model
