# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-12-02 20:02:28

import numpy as np
import pandas as pd
from neuron import h

from PySONIC.utils import si_format, pow10_format, logger, debug, logCache
from PySONIC.threshold import threshold
from PySONIC.constants import *
from PySONIC.core import Model, PointNeuron
from PySONIC.postpro import detectSpikes, prependDataFrame

from .pyhoc import *
from .node import SonicNode
from .connectors import SeriesConnector
from ..constants import *


class ExtendedSonicNode(SonicNode):

    simkey = 'nano_ext_SONIC'
    titration_var = 'Adrive'  # name of the titration parameter

    def __init__(self, pneuron, rs, a=32e-9, Fdrive=500e3, fs=0.5, deff=100e-9):

        # Assign attributes
        self.rs = rs  # Ohm.cm
        self.deff = deff  # m
        assert fs < 1., 'fs must be lower than 1'

        # Initialize parent class and delete nominal section
        super().__init__(pneuron, id=None, a=a, Fdrive=Fdrive, fs=1.)
        self.fs = fs
        del self.section

        # Create sections and set their geometry, biophysics and topology
        self.createSections()
        self.setGeometry()  # must be called PRIOR to setTopology()
        self.setBiophysics()
        self.setResistivity()
        self.setTopology()

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return '{}({}, fs={}, deff={:.0f} nm)'.format(
            self.__class__.__name__, self.pneuron, self.fs, self.deff * 1e9)

    def getLookup(self):
        ''' Get lookups computing with fs = 1. '''
        return self.nbls.getLookup2D(self.Fdrive, 1.)

    def createSections(self):
        ''' Create morphological sections. '''
        self.sections = {id: h.Section(name=id, cell=self) for id in ['sonophore', 'surroundings']}

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

    def setBiophysics(self):
        ''' Set section-specific membrane properties with specific sonophore membrane coverage. '''
        logger.debug('defining membrane biophysics: {}'.format(self.str_biophysics()))
        for sec in self.sections.values():
            sec.insert(self.mechname)

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

    def setStimAmp(self, Adrive):
        ''' Set US stimulation amplitude.

            :param Adrive: acoustic pressure amplitude (Pa)
        '''
        setattr(self.sections['sonophore'], 'Adrive_{}'.format(self.mechname), Adrive * 1e-3)
        setattr(self.sections['surroundings'], 'Adrive_{}'.format(self.mechname), 0.)

    def setStimON(self, value):
        return setStimON(self, value)

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    def simulate(self, Adrive, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param A: acoustic pressure amplitude (Pa)
            :param pp: pulse protocol object
            :param dt: integration time step (s)
        '''
        m = self.modality
        Astr = f'{m["name"]} = {si_format(Adrive * m["factor"], 2)}{m["unit"]}'
        logger.info(f'{self}: simulation @ {Astr}, {pp.pprint()}')

        # Set recording vectors
        t = setTimeProbe()
        stim = setStimProbe(self.sections['sonophore'], self.mechname)
        probes = {k: self.setProbesDict(v) for k, v in self.sections.items()}

        # Set stimulus amplitude and integrate model
        self.setStimAmp(Adrive)
        integrate(self, pp, dt, atol)

        # Store output in dataframes
        data = {}
        for id in self.sections.keys():
            data[id] = pd.DataFrame({
                't': vec_to_array(t) * 1e-3,  # s
                'stimstate': vec_to_array(stim)
            })
            for k, v in probes[id].items():
                data[id][k] = vec_to_array(v)
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
    def titrate(self, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param pp: pulse protocol object
            :param method: integration method
            :param xfunc: function determining whether condition is reached from simulation output
            :return: determined threshold amplitude (Pa)
        '''
        return threshold(
            lambda x: self.isExcited(self.simulate(x, pp)[0]),
            self.Arange, x0=ASTIM_AMP_INITIAL,
            eps_thr=ASTIM_ABS_CONV_THR, rel_eps_thr=1e0, precheck=True)

    def filecodes(self, Adrive, pp):
        # Get parent codes and supress irrelevant entries
        codes = self.nbls.filecodes(self.Fdrive, Adrive, pp, self.fs, 'NEURON', None)
        del codes['method']
        codes.update({
            'rs': f'rs{self.rs:.1e}Ohm.cm',
            'deff': f'deff{(self.deff * 1e9):.0f}nm'
        })
        return codes

    def meta(self, Adrive, pp):
        meta = super().meta(Adrive, pp)
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

