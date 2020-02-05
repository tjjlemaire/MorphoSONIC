# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-05 16:52:08

import abc
import pickle
import csv
import os
import numpy as np
import pandas as pd
from inspect import signature
from scipy import stats
import scipy.signal

from PySONIC.neurons import getPointNeuron
from PySONIC.core import Model, PointNeuron, NeuronalBilayerSonophore
from PySONIC.utils import si_format, pow10_format, logger, plural, filecode, simAndSave
from PySONIC.threshold import threshold
from PySONIC.constants import *
from PySONIC.postpro import detectSpikes, prependDataFrame

from .pyhoc import *
from ..constants import *
from .node import IintraNode, SonicNode
from .connectors import SeriesConnector


class SennFiber(metaclass=abc.ABCMeta):
    ''' Generic single-cable, Spatially Extended Nonlinear Node (SENN) fiber model. '''

    tscale = 'ms'        # relevant temporal scale of the model
    titration_var = 'A'  # name of the titration parameter

    @property
    @abc.abstractmethod
    def simkey(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    def __init__(self, pneuron, nnodes, rs, nodeD, nodeL, interD, interL):
        ''' Constructor.

            :param pneuron: point-neuron model object
            :param fiberD: fiber outer diameter (m)
            :param nnodes: number of nodes
            :param rs: axoplasmic resistivity (Ohm.cm)
            :param nodeD: node diameter (m)
            :param nodeL: node length (m)
            :param interD: internode diameter (m)
            :param interL: internode length (m)
        '''
        if not isinstance(pneuron, PointNeuron):
            raise TypeError(f'{pneuron} is not a valid PointNeuron instance')
        if nnodes % 2 == 0:
            raise ValueError('Number of nodes must be odd')

        # Assign attributes
        self.pneuron = pneuron
        self.rs = rs          # Ohm.cm
        self.nnodes = nnodes
        self.nodeD = nodeD    # m
        self.nodeL = nodeL    # m
        self.interD = interD  # m
        self.interL = interL  # m
        self.mechname = self.pneuron.name + 'auto'
        self.cell = self

        # Compute nodal and internodal axial resistance ()
        self.R_node = self.resistance(self.nodeD, self.nodeL)  # Ohm
        self.R_inter = self.resistance(self.interD, self.interL)  # Ohm

        # Assign nodes IDs
        self.ids = [f'node{i}' for i in range(self.nnodes)]

        # Construct model
        self.construct()

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, value):
        if value <= 0:
            raise ValueError('axoplasmic resistivity must be positive')
        self._rs = value

    @property
    def nnodes(self):
        return self._nnodes

    @nnodes.setter
    def nnodes(self, value):
        if value % 2 == 0:
            raise ValueError('number of nodes must be odd')
        self._nnodes = value

    @property
    def nodeD(self):
        return self._nodeD

    @nodeD.setter
    def nodeD(self, value):
        if value <= 0:
            raise ValueError('node diameter must be positive')
        self._nodeD = value

    @property
    def nodeL(self):
        return self._nodeL

    @nodeL.setter
    def nodeL(self, value):
        if value <= 0:
            raise ValueError('node length must be positive')
        self._nodeL = value

    @property
    def interD(self):
        return self._interD

    @interD.setter
    def interD(self, value):
        if value <= 0:
            raise ValueError('internode diameter must be positive')
        self._interD = value

    @property
    def interL(self):
        return self._interL

    @interL.setter
    def interL(self, value):
        if value < 0:
            raise ValueError('internode diameter must be positive or null')
        self._interL = value

    @staticmethod
    def getSennArgs(meta):
        return [meta[x] for x in ['nnodes', 'rs', 'nodeD', 'nodeL', 'interD', 'interL']]

    @classmethod
    def initFromMeta(cls, meta):
        return cls(getPointNeuron(meta['neuron']), *cls.getSennArgs(meta))

    def construct(self):
        ''' Create and connect node sections with assigned membrane dynamics. '''
        self.createSections(self.ids)
        self.setGeometry()  # must be called PRIOR to build_custom_topology()
        self.setResistivity()
        self.setTopology()

    def clear(self):
        ''' delete all model sections. '''
        del self.sections

    def reset(self):
        ''' delete and re-construct all model sections. '''
        self.clear()
        self.construct()

    def str_biophysics(self):
        return f'{self.pneuron.name} neuron'

    def str_nodes(self):
        return f'{self.nnodes} node{plural(self.nnodes)}'

    def str_resistivity(self):
        return f'rs = {si_format(self.rs)}Ohm.cm'

    def str_geometry(self):
        ''' Format model geometrical parameters into string. '''
        params = {
            'nodeD': self.nodeD,
            'nodeL': self.nodeL,
            'interD': self.interD,
            'interL': self.interL
        }
        lbls = {key: f'{key} = {si_format(val, 1)}m' for key, val in params.items()}
        return ', '.join(lbls.values())

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return '{}({})'.format(self.__class__.__name__, ', '.join([
            self.str_biophysics(),
            self.str_nodes(),
            self.str_resistivity(),
            self.str_geometry()
        ]))

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

    def getNodeCoords(self):
        ''' Return vector of node coordinates along axial dimension, centered at zero (um). '''
        xcoords = (self.nodeL + self.interL) * np.arange(self.nnodes) + self.nodeL / 2
        return xcoords - xcoords[int((self.nnodes - 1) / 2)]

    def length(self):
        return self.nnodes * self.nodeL + (self.nnodes - 1) * self.interL

    @property
    @abc.abstractmethod
    def createSections(self, ids):
        ''' Create morphological sections. '''
        return NotImplementedError

    def setGeometry(self):
        ''' Set sections geometry. '''
        logger.debug(f'defining sections geometry: {self.str_geometry()}')
        for sec in self.sections.values():
            sec.diam = self.nodeD * 1e6  # um
            sec.L = self.nodeL * 1e6     # um
            sec.nseg = 1

    def resistance(self, d, L, rs=None):
        ''' Return resistance of cylindrical section based on its diameter and length.

            :param d: cylinder diameter (m)
            :param L: cylinder length (m)
            :param rs: axial resistivity (Ohm.cm)
            :return: resistance (Ohm)
        '''
        if rs is None:
            rs = self.rs
        return 4 * rs * L / (np.pi * d**2) * 1e-2  # Ohm

    def setResistivity(self): ####ASK!
        ''' Set sections axial resistivity, corrected to account for internodes and membrane capacitance
            in the Q-based differentiation scheme. '''

        logger.debug(f'nominal nodal resistivity: rs = {self.rs:.0f} Ohm.cm')
        rho_nodes = np.ones(self.nnodes) * self.rs  # Ohm.cm

        # Adding extra resistivity to account for half-internodal resistance
        # for each connected side of each node
        if self.R_inter > 0:
            logger.debug('adding extra-resistivity to account for internodal resistance')
            R_extra = np.hstack((
                [self.R_inter / 2],
                [self.R_inter] * (self.nnodes - 2),
                [self.R_inter / 2]
            ))  # Ohm
            rho_extra = R_extra * self.rs / self.R_node  # Ohm.cm
            rho_nodes += rho_extra  # Ohm.cm

        # In case the axial coupling variable is v (an alias for membrane charge density),
        # multiply resistivity by membrane capacitance to ensure consistency of Q-based
        # differential scheme, where Iax = dV / r = dQ / (r * cm)
        if self.connector is None or self.connector.vref == 'v':
            logger.debug('adjusting resistivities to account for Q-based differential scheme')
            rho_nodes *= self.pneuron.Cm0 * 1e2  # Ohm.cm

        # Assigning resistivities to sections
        for sec, rho in zip(self.sections.values(), rho_nodes):
            sec.Ra = rho

    def setTopology(self):
        ''' Connect the sections in series. '''
        sec_list = list(self.sections.values())
        if self.connector is None:
            logger.debug('building standard topology')
            for sec1, sec2 in zip(sec_list[:-1], sec_list[1:]):
                sec2.connect(sec1, 1, 0)
        else:
            logger.debug(f'building custom {self.connector.vref}-based topology')
            for sec in sec_list:
                self.connector.attach(sec)
            for sec1, sec2 in zip(sec_list[:-1], sec_list[1:]):
                self.connector.connect(sec1, sec2)

    def setFuncTables(self, *args, **kwargs):
        return setFuncTables(self, *args, **kwargs)

    def setModLookup(self, *args, **kwargs):
        return setModLookup(self, *args, **kwargs)

    @abc.abstractmethod
    def preProcessAmps(self, drives):
        ''' Convert stimulus to a model-friendly unit

            :param drives: model-sized vector of stimulus drives
            :return: model-sized vector of converted stimulus drives
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def setDrives(self, source):
        ''' Set distributed stimulus amplitudes. '''
        return NotImplementedError

    def setStimON(self, value):
        return setStimON(self, value)

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        h.finitialize(self.pneuron.Qm0 * 1e5)  # nC/cm2

    def toggleStim(self):
        return toggleStim(self)

    @Model.checkTitrate
    @Model.addMeta
    def simulate(self, source, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param source: source object
            :param pp: pulsed protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
            :return: output dataframe
        '''
        logger.info(self.desc(self.meta(source, pp)))

        # Set recording vectors
        t = setTimeProbe()
        stim = setStimProbe(self.sections[self.ids[0]], self.mechname)
        probes = {k: self.nodes[k].setProbesDict(v) for k, v in self.sections.items()}

        # Set distributed drives
        self.setDrives(source)

        # Integrate model
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

    def modelMeta(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nnodes': self.nnodes,
            'rs': self.rs,
            'nodeD': self.nodeD,
            'nodeL': self.nodeL,
            'interD': self.interD,
            'interL': self.interL,
        }

    def meta(self, source, pp):
        return {**self.modelMeta(), **{
            'source': source,
            'pp': pp
        }}

    def desc(self, meta):
        return f'{self}: simulation @ {meta["source"]}, {meta["pp"].desc}'

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        nspikes_start = detectSpikes(data[self.ids[0]])[0].size
        nspikes_end = detectSpikes(data[self.ids[-1]])[0].size
        return nspikes_start > 0 and nspikes_end > 0

    def getArange(self, source):
        return [source.computeSourceAmp(self, x) for x in self.A_range]

    def getAstart(self, source):
        return source.computeSourceAmp(self, self.A_start)

    def titrationFunc(self, *args):
        data, _ = self.simulate(*args)
        return self.isExcited(data)

    @abc.abstractmethod
    def titrate(self, source, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param source: source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
        raise NotImplementedError

    def getPltVars(self, *args, **kwargs):
        return self.pneuron.getPltVars(*args, **kwargs)

    @property
    def pltScheme(self):
        return self.pneuron.pltScheme

    @property
    def modelcodes(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nnodes': f'{self.nnodes}node{plural(self.nnodes)}',
            'rs': f'rs{self.rs:.0f}ohm.cm',
            'nodeD': f'nodeD{(self.nodeD * 1e6):.1f}um',
            'nodeL': f'nodeL{(self.nodeL * 1e6):.1f}um',
            'interD': f'interD{(self.interD * 1e6):.1f}um',
            'interL': f'interL{(self.interL * 1e6):.1f}um'
        }

    def filecodes(self, source, pp):
        return {
            **self.modelcodes,
            **source.filecodes(),
            'nature': pp.nature,
            **pp.filecodes
        }

    def filecode(self, *args):
        return filecode(self, *args)

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)

    def getSpikesTimings(self, data, zcross=True, spikematch='majority'):
        ''' Return an array containing occurence times of spikes detected on a collection of nodes.

            :param data: simulation output dataframe
            :param zcross: boolean stating whether to consider ascending zero-crossings preceding peaks
                as temporal reference for spike occurence timings
            :return: dictionary of spike occurence times (s) per node.
        '''
        tspikes = {}
        nspikes = np.zeros(len(data.items()))

        for i, (id, df) in enumerate(data.items()):

            # Detect spikes on current trace
            ispikes, *_ = detectSpikes(df, key='Vm', mph=SPIKE_MIN_VAMP, mpt=SPIKE_MIN_DT,
                                       mpp=SPIKE_MIN_VPROM)
            nspikes[i] = ispikes.size

            if ispikes.size > 0:
                # Extract time vector
                t = df['t'].values  # s
                if zcross:
                    # Consider spikes as time of zero-crossing preceding each peak
                    Vm = df['Vm'].values  # mV
                    i_zcross = np.where(np.diff(np.sign(Vm)) > 0)[0]  # detect ascending zero-crossings
                    if i_zcross.size > ispikes.size:
                        # If mismatch, remove irrelevant zero-crossings by taking only the ones preceding
                        # each detected peak
                        i_zcross = np.array([i_zcross[(i_zcross - i1) < 0].max() for i1 in ispikes])
                    slopes = (Vm[i_zcross + 1] - Vm[i_zcross]) / (t[i_zcross + 1] - t[i_zcross])  # slopes (mV/s)
                    tzcross = t[i_zcross] - (Vm[i_zcross] / slopes)
                    errmsg = 'Ascending zero crossing #{} (t = {:.2f} ms) not preceding peak #{} (t = {:.2f} ms)'
                    for ispike, (tzc, tpeak) in enumerate(zip(tzcross, t[ispikes])):
                        assert tzc < tpeak, errmsg.format(ispike, tzc * 1e3, ispike, tpeak * 1e3)
                    tspikes[id] = tzcross
                else:
                    tspikes[id] = t[ispikes]

        if spikematch == 'strict':
            # Assert consistency of spikes propagation
            assert np.all(nspikes == nspikes[0]), 'Inconsistent number of spikes in different nodes'
            if nspikes[0] == 0:
                logger.warning('no spikes detected')
                return None
        else:
            # Use majority voting
            nfrequent = np.int(stats.mode(nspikes).mode)
            tspikes = {k: v for k, v in tspikes.items() if len(v) == nfrequent}

        return pd.DataFrame(tspikes)

    def getConductionVelocity(self, data, ids=None, out='median'):
        ''' Compute average conduction speed from simulation results.

            :param data: simulation output dataframe
            :return: conduction speed output (m/s).
        '''
        # By default, consider all fiber nodes
        if ids is None:
            ids = self.ids.copy()

        # Remove end nodes from calculations if present
        for x in [0, -1]:
            if self.ids[x] in ids:
                ids.remove(self.ids[x])

        # Compute spikes timing dataframe (based on nspikes majority voting) and
        # update list of relevant sections accordingly
        tspikes = self.getSpikesTimings({id: data[id] for id in ids})  # (nspikes x nnodes)
        ids = list(tspikes.columns.values)  # (nnodes)

        # Get coordinates of relevant nodes
        indexes = [self.ids.index(id) for id in ids]  # (nnodes)
        xcoords = self.getNodeCoords()[indexes]  # (nnodes)

        # Compute distances across consecutive nodes only, and associated spiking delays for first spike only
        distances, delays = [], []  # (nnodes - 1)
        for i in range(len(ids)-1):
            d = xcoords[i]-xcoords[i-1]
            if np.isclose(d, (self.nodeL + self.interL)):
                distances.append(d)
                dt = np.abs(tspikes.values[0][i]-tspikes.values[0][i-1])
                delays.append(dt)   # node-to-node delay

        # Compute conduction velocities for each considered node pair
        # distances = np.tile(distances, (1, delays.shape[0]))   # dimension matching multi-dimensional delay array (nnodes x nspikes)
        velocities = np.array(distances) / np.array(delays)  # m/s

        # Return specific output metrics
        if out == 'range':
            return velocities.min(), velocities.max()
        elif out == 'median':
            return np.median(velocities)
        elif out == 'mean':
            return np.mean(velocities)
        else:
            raise AttributeError(f'invalid out option: {out}')

    def getSpikeAmp(self, data, ids=None, key='Vm', out='range'):
        # By default, consider all fiber nodes
        if ids is None:
            ids = self.ids.copy()
        amps = np.array([np.ptp(data[id][key].values) for id in ids])
        if out == 'range':
            return amps.min(), amps.max()
        elif out == 'median':
            return np.median(amps)
        elif out == 'mean':
            return np.mean(amps)
        else:
            raise AttributeError(f'invalid out option: {out}')


class EStimFiber(SennFiber):

    def __init__(self, *args, **kwargs):
        self.connector = None
        super().__init__(*args, **kwargs)

    def createSections(self, ids):
        ''' Create morphological sections. '''
        self.nodes = {id: IintraNode(self.pneuron, id, cell=self) for id in ids}
        self.sections = {id: node.section for id, node in self.nodes.items()}

    def setDrives(self, source):
        ''' Set distributed stimulation drives. '''
        amps = source.computeNodesAmps(self)
        self.Iinj = self.preProcessAmps(amps)
        logger.debug('injected intracellular currents: Iinj = [{}] nA'.format(
            ', '.join([f'{I:.2f}' for I in self.Iinj])))

        # Assign current clamps
        self.iclamps = []
        for i, sec in enumerate(self.sections.values()):
            iclamp = h.IClamp(sec(0.5))
            iclamp.delay = 0  # we want to exert control over amp starting at 0 ms
            iclamp.dur = 1e9  # dur must be long enough to span all our changes
            self.iclamps.append(iclamp)

    def setStimON(self, value):
        value = super().setStimON(value)
        for iclamp, Iinj in zip(self.iclamps, self.Iinj):
            iclamp.amp = value * Iinj
        return value

    def titrate(self, source, pp):
        Ithr = threshold(
            lambda x: self.titrationFunc(source.updatedX(-x if source.is_cathodal else x), pp),
            self.getArange(source), rel_eps_thr=1e-2, precheck=False)
        if source.is_cathodal:
            Ithr = -Ithr
        return Ithr


class IextraFiber(EStimFiber):

    simkey = 'senn_Iextra'
    A_range = (1e0, 1e5)  # mV

    def preProcessAmps(self, Ve):
        ''' Convert array of extracellular potentials into equivalent intracellular injected currents.

            :param Ve: model-sized vector of extracellular potentials (mV)
            :return: model-sized vector of intracellular injected currents (nA)
        '''
        logger.debug('Extracellular potentials: Ve = [{}] mV'.format(
            ', '.join([f'{v:.2f}' for v in Ve])))
        Iinj =  np.diff(Ve, 2) / (self.R_node + self.R_inter) * 1e6  # nA
        Iinj = np.pad(Iinj, (1, 1), 'constant')  # zero-padding on both extremities
        logger.debug('Equivalent intracellular currents: Iinj = [{}] nA'.format(
            ', '.join([f'{I:.2f}' for I in Iinj])))
        return Iinj


class IintraFiber(EStimFiber):

    simkey = 'senn_Iintra'
    A_range = (1e-12, 1e-7)  # A

    def preProcessAmps(self, I):
        return I


class SonicFiber(SennFiber):

    simkey = 'senn_SONIC'
    A_range = (1e0, 6e5)  # Pa

    def __init__(self, *args, a=32e-9, fs=1., **kwargs):
        # Retrieve point neuron object
        pneuron = args[0]

        # Assign attributes
        self.pneuron = pneuron
        self.mechname = self.pneuron.name + 'auto'
        self.a = a            # m
        self.fs = fs          # (-)

        # Initialize connector NBLS objects
        self.connector = SeriesConnector(vref=f'Vm_{self.mechname}', rmin=None)
        self.nbls = NeuronalBilayerSonophore(self.a, self.pneuron)

        self.fref = None
        self.pylkp = None
        super().__init__(*args, **kwargs)

    @classmethod
    def initFromMeta(cls, meta):
        return cls(getPointNeuron(meta['neuron']), *cls.getSennArgs(meta), a=meta['a'], fs=meta['fs'])

    def str_biophysics(self):
        return f'{super().str_biophysics()}, a = {self.a * 1e9:.1f} nm'

    def createSections(self, ids):
        ''' Create morphological sections. '''
        self.nodes = {
            id: SonicNode(self.pneuron, id, cell=self, a=self.a, fs=self.fs, nbls=self.nbls)
            for id in ids}
        self.sections = {id: node.section for id, node in self.nodes.items()}

    def setPyLookup(self, f):
        if self.pylkp is None or f != self.fref:
            self.pylkp = self.nbls.getLookup2D(f, self.fs)
            self.fref = f

    def preProcessAmps(self, A):
        return A

    def setDrives(self, source):
        ''' Set distributed stimulation drives. '''
        self.setFuncTables(source.f)
        amps = source.computeNodesAmps(self)
        self.amps = self.preProcessAmps(amps)
        logger.debug('Acoustic pressures: A = [{}] kPa'.format(
            ', '.join([f'{A * 1e-3:.2f}' for A in self.amps])))
        for i, sec in enumerate(self.sections.values()):
            setattr(sec, 'Adrive_{}'.format(self.mechname), self.amps[i] * 1e-3)

    def meta(self, source, pp):
        meta = super().meta(source, pp)
        meta.update({
            'a': self.a,
            'fs': self.fs
        })
        return meta

    def filecodes(self, source, pp):
        return {
            **self.modelcodes,
            'a': f'{self.a * 1e9:.0f}nm',
            'fs': f'fs{self.fs * 1e2:.0f}%' if self.fs <= 1 else None,
            **source.filecodes(),
            'nature': pp.nature,
            **pp.filecodes
        }

    def titrate(self, source, pp):
        Arange = self.getArange(source)
        A_conv_thr = np.abs(Arange[1] - Arange[0]) / 1e4
        return threshold(
            lambda x: self.titrationFunc(source.updatedX(x), pp),
            Arange, eps_thr=A_conv_thr, rel_eps_thr=1e-2, precheck=True)
