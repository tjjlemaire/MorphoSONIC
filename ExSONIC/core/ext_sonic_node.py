# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-19 15:33:16

import os
import numpy as np
import pandas as pd

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import si_format, logger, logCache
from PySONIC.threshold import threshold
from PySONIC.constants import *
from PySONIC.core import Model, PointNeuron
from PySONIC.postpro import detectSpikes, prependDataFrame

from ..constants import *
from .node import SonicNode
from .connectors import SeriesConnector


class ExtendedSonicNode(SonicNode):

    simkey = 'nano_ext_SONIC'
    secnames = ['sonophore', 'surroundings']

    def __init__(self, pneuron, rs, a=32e-9, fs=0.5, deff=100e-9):

        # Assign attributes
        self.rs = rs  # Ohm.cm
        self.deff = deff  # m
        assert fs < 1., 'fs must be lower than 1'

        # Initialize parent class and delete nominal section
        super().__init__(pneuron, id=None, a=a, fs=1.)
        self.fs = fs
        del self.section

        # Construct model
        self.construct()

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{super().__repr__()[:-1]}, rs={self.rs:.2e} Ohm.cm, deff={self.deff * 1e9:.0f} nm)'

    @classmethod
    def initFromMeta(cls, meta):
        return cls(getPointNeuron(meta['neuron']), meta['rs'], a=meta['a'],
                   fs=meta['fs'], deff=meta['deff'])

    def setPyLookup(self, f):
        ''' Set lookups computing with fs = 1. '''
        if self.fref is None or f != self.fref:
            self.pylkp = self.nbls.getLookup2D(f, 1.)
            self.fref = f

    def construct(self):
        ''' Create and connect node sections with assigned membrane dynamics. '''
        self.createSections()
        self.setGeometry()  # must be called PRIOR to build_custom_topology()
        self.setResistivity()
        self.setTopology()

    def createSections(self):
        ''' Create morphological sections. '''
        self.sections = {id: self.createSection(id) for id in self.secnames}

    def clear(self):
        del self.sections

    def translateRadialGeometry(self, depth, r1, r2):
        ''' Return geometrical parameters of cylindrical sections to match quantities of
            membrane and axial currents in a radial configuration between a central
            and a peripheral section.

            :param depth: depth of the radial sections (um)
            :param r1: radius of central section (um)
            :param r2: outer radius of peripheral section (um)
            :return: 3-tuple with sections common diameter and their respective lengths (um)
        '''
        logger.debug('radial geometry: depth = {}m, r1 = {}m, r2 = {}m'.format(
                *si_format(np.array([depth, r1, r2]) * 1e-6, 2)))
        d = np.power(4 * depth * r2**2 / np.log((r1 + r2) / r1), 1 / 3)  # um
        L1 = r1**2 / d  # um
        L2 = r2**2 / d - L1  # um
        logger.debug('equivalent linear geometry: d = {}m, L1 = {}m, L2 = {}m'.format(
            *si_format(np.array([d, L1, L2]) * 1e-6, 2)))
        return d, L1, L2

    def setGeometry(self):
        ''' Set sections geometry. '''
        d, L1, L2 = self.translateRadialGeometry(
            self.deff * 1e6, self.a * 1e6, self.a / np.sqrt(self.fs) * 1e6)
        for sec in self.sections.values():
            sec.diam = d  # um
            sec.nseg = 1
        self.sections['sonophore'].L = L1  # um
        self.sections['surroundings'].L = L2  # um

    def setResistivity(self):
        ''' Set sections axial resistivity, corrected to account for internodes and membrane capacitance
            in the Q-based differentiation scheme. '''
        for sec in self.sections.values():
            sec.Ra = self.rs

    def setTopology(self):
        self.connector = SeriesConnector(vref='Vm_{}'.format(self.mechname), rmin=1e2)
        logger.debug('building custom {}-based topology'.format(self.connector.vref))
        list(map(self.connector.attach, self.sections.values()))
        self.connector.connect(self.sections['sonophore'], self.sections['surroundings'])

    def setDrive(self, drive):
        ''' Set US drive. '''
        logger.debug(f'Stimulus: {drive}')
        self.setFuncTables(drive.f)
        setattr(self.sections['sonophore'], 'Adrive_{}'.format(self.mechname), drive.A * 1e-3)
        setattr(self.sections['surroundings'], 'Adrive_{}'.format(self.mechname), 0.)

    def setStimON(self, value):
        for sec in self.sections.values():
            sec.setStimON(value)
        return value

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    def simulate(self, drive, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param drive: acoustic drive object
            :param pp: pulse protocol object
            :param dt: integration time step (s)
        '''
        logger.info(f'{self}: simulation @ {drive.desc}, {pp.desc}')

        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.sections[self.secnames[0]].setStimProbe()
        probes = {k: v.setProbesDict() for k, v in self.sections.items()}

        # Set stimulus amplitude and integrate model
        self.setDrive(drive)
        self.integrate(pp, dt, atol)

        # Store output in dataframes
        data = {}
        for id in self.sections.keys():
            data[id] = pd.DataFrame({
                't': t.to_array() * 1e-3,  # s
                'stimstate': stim.to_array()
            })
            for k, v in probes[id].items():
                data[id][k] = v.to_array()
            data[id].loc[:,'Qm'] *= 1e-5  # C/m2

        # Prepend initial conditions (prior to stimulation)
        data = {id: prependDataFrame(df) for id, df in data.items()}

        return data

    @staticmethod
    def getNSpikes(data):
        ''' Compute number of spikes in charge profile of simulation output.

            :param data: dataframe containing output time series
            :return: number of detected spikes
        '''
        return detectSpikes(data['sonophore'])[0].size

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        nspikes_sonophore = detectSpikes(data['sonophore'])[0].size
        nspikes_surroundings = detectSpikes(data['surroundings'])[0].size
        return nspikes_start > 0 and nspikes_end > 0

    @logCache(os.path.join(os.path.split(__file__)[0], 'nanoextsonic_titrations.log'))
    def titrate(self, drive, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param pp: pulse protocol object
            :param method: integration method
            :param xfunc: function determining whether condition is reached from simulation output
            :return: determined threshold amplitude (Pa)
        '''
        self.setFuncTables(drive.f)  # pre-loading lookups to have a defined Arange
        return threshold(
            lambda x: self.isExcited(self.simulate(drive.updatedX(x), pp)[0]),
            self.Arange, x0=ASTIM_AMP_INITIAL,
            eps_thr=ASTIM_ABS_CONV_THR, rel_eps_thr=1e0, precheck=True)

    def filecodes(self, drive, pp):
        # Get parent codes and supress irrelevant entries
        codes = self.nbls.filecodes(drive, pp, self.fs, 'NEURON', None)
        del codes['method']
        codes.update({
            'rs': f'rs{self.rs:.1e}Ohm.cm',
            'deff': f'deff{(self.deff * 1e9):.0f}nm'
        })
        return codes

    def meta(self, drive, pp):
        meta = super().meta(drive, pp)
        meta['fs'] = self.fs
        meta['rs'] = self.rs
        meta['deff'] = self.deff
        meta['simkey'] = self.simkey
        return meta

    @staticmethod
    def isExcited(data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        return PointNeuron.getNSpikes(data['sonophore']) > 0

    @staticmethod
    def inputs():
        return {
            'section': {
                'desc': 'section',
                'label': 'section',
                'unit': '',
                'factor': 1e0,
                'precision': 0
            }
        }

