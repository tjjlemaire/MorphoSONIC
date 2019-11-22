# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-22 18:56:41

import abc
import pickle
import numpy as np
import pandas as pd
from inspect import signature

from PySONIC.core import Model, PointNeuron, NeuronalBilayerSonophore
from PySONIC.utils import si_format, pow10_format, logger, plural, binarySearch, pow2Search, filecode, simAndSave
from PySONIC.constants import *
from PySONIC.postpro import detectSpikes, prependDataFrame

from .pyhoc import *
from ..constants import *
from .node import IintraNode, SonicNode
from .connectors import SeriesConnector


class SennFiber(metaclass=abc.ABCMeta):
    ''' Generic interface to the SENN fiber model. '''

    tscale = 'ms'  # relevant temporal scale of the model
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

        # Compute nodal and internodal axial resistance ()
        self.R_node = self.resistance(self.nodeD, self.nodeL)  # Ohm
        self.R_inter = self.resistance(self.interD, self.interL)  # Ohm

        # Assign nodes IDs
        self.ids = [f'node{i}' for i in range(self.nnodes)]

        # Construct model
        self.construct()

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

    def preProcessAmps(self, A):
        ''' Convert stimulus intensities to a model-friendly unit

            :param A: model-sized vector of stimulus intensities
            :return: model-sized vector of converted stimulus intensities
        '''
        return A

    @property
    @abc.abstractmethod
    def setStimAmps(self, amps, config):
        ''' Set distributed stimulus amplitudes. '''
        return NotImplementedError

    def setStimON(self, value):
        return setStimON(self, value)

    def toggleStim(self):
        return toggleStim(self)

    @Model.checkTitrate
    @Model.addMeta
    def simulate(self, psource, A, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param psource: point source object
            :param A: stimulus amplitude (A)
            :param pp: pulsed protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
            :return: output dataframe
        '''
        logger.info(self.desc(self.meta(psource, A, pp)))

        # Set recording vectors
        t = setTimeProbe()
        stim = setStimProbe(self.sections[self.ids[0]], self.mechname)
        probes = {k: self.nodes[k].setProbesDict(v) for k, v in self.sections.items()}

        # Set distributed stimulus amplitudes
        self.setStimAmps(psource.computeNodesAmps(self, A))

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

    def meta(self, psource, A, pp):
        return {**self.modelMeta(), **{
            'psource': psource,
            'A': A,
            'pp': pp
        }}

    def desc(self, meta):
        psource = meta["psource"]
        return f'{self}: simulation @ {repr(psource)}, {psource.strAmp(meta["A"])}, {meta["pp"].pprint()}'

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        nspikes_start = detectSpikes(data[self.ids[0]])[0].size
        nspikes_end = detectSpikes(data[self.ids[-1]])[0].size
        return nspikes_start > 0 and nspikes_end > 0

    def getArange(self, psource):
        return [psource.computeSourceAmp(self, x) for x in self.A_range]

    def titrationFunc(self, args):
        psource, A, *args = args
        data, _ = self.simulate(psource, A, *args)
        return self.isExcited(data)

    def titrate(self, psource, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param psource: point source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
        Amin, Amax = self.getArange(psource)
        A_conv_thr = np.abs(Amax - Amin) / 1e4
        Athr = pow2Search(self.titrationFunc, [psource, pp], 1, Amin, Amax)
        if np.isnan(Athr):
            return Athr
        Arange = (Athr / 2, Athr)
        Athr = binarySearch(
            self.titrationFunc, [psource, pp], 1, Arange, A_conv_thr)
        return Athr

    def getPltVars(self, *args, **kwargs):
        return self.pneuron.getPltVars(*args, **kwargs)

    def getPltScheme(self, *args, **kwargs):
        return self.pneuron.getPltScheme(*args, **kwargs)

    def modelCodes(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nnodes': f'{self.nnodes}node{plural(self.nnodes)}',
            'rs': f'rs{self.rs:.0f}(Ohm.cm)',
            'nodeD': f'nodeD{(self.nodeD * 1e6):.1f}um',
            'nodeL': f'nodeL{(self.nodeL * 1e6):.1f}um',
            'interD': f'interD{(self.interD * 1e6):.1f}um',
            'interL': f'interL{(self.interL * 1e6):.1f}um'
        }

    def filecodes(self, psource, A, pp):
        return {
            **self.modelCodes(),
            **psource.filecodes(A),
            'nature': 'CW' if pp.isCW() else 'PW',
            **pp.filecodes()
        }

    def filecode(self, *args):
        return filecode(self, *args)

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)

    def getSpikesTimings(self, data, zcross=True):
        ''' Return an array containing occurence times of spikes detected on a collection of nodes.

            :param data: simulation output dataframe
            :param zcross: boolean stating whether to consider ascending zero-crossings preceding peaks
                as temporal reference for spike occurence timings
            :return: dictionary of spike occurence times (s) per node.
        '''
        tspikes = {}
        nspikes = None

        for id, df in data.items():

            # Detect spikes on current trace
            ispikes, *_ = detectSpikes(df, key='Vm', mph=SPIKE_MIN_VAMP, mpt=SPIKE_MIN_DT,
                                       mpp=SPIKE_MIN_VPROM)

            # Assert consistency of spikes propagation
            if nspikes is None:
                nspikes = ispikes.size
                if nspikes == 0:
                    logger.warning('no spikes detected')
                    return None
            else:
                assert ispikes.size == nspikes, 'Inconsistent number of spikes in different nodes'

            if zcross:
                # Consider spikes as time of zero-crossing preceding each peak
                t = df['t'].values  # s
                Vm = df['Vm'].values  # mV
                i_zcross = np.where(np.diff(np.sign(Vm)) > 0)[0]  # detect ascending zero-crossings
                slopes = (Vm[i_zcross + 1] - Vm[i_zcross]) / (t[i_zcross + 1] - t[i_zcross])  # slopes (mV/s)
                offsets = Vm[i_zcross] - slopes * t[i_zcross]  # offsets (mV)
                tzcross = -offsets / slopes  # interpolated times (s)
                errmsg = 'Ascending zero crossing #{} (t = {:.2f} ms) not preceding peak #{} (t = {:.2f} ms)'
                for ispike, (tzc, tpeak) in enumerate(zip(tzcross, t[ispikes])):
                    assert tzc < tpeak, errmsg.format(ispike, tzc * 1e3, ispike, tpeak * 1e3)
                tspikes[id] = tzcross
            else:
                tspikes[id] = t[ispikes]
        return pd.DataFrame(tspikes)

    def getConductionVelocity(self, data, ids=None):
        ''' Compute average conduction speed from simulation results.

            :param data: simulation output dataframe
            :return: array of condiction speeds per spike (m/s).
        '''
        # By default, consider all fiber nodes
        if ids is None:
            ids = self.ids.copy()

        # Remove end nodes from calculations if present
        for x in [0, -1]:
            if self.ids[x] in ids:
                ids.remove(self.ids[x])

        # Compute node to node distances
        indexes = [self.ids.index(id) for id in ids]  # relevant node indexes
        xcoords = self.getNodeCoords()[indexes]  # corresponding x-coordinates
        distances = np.diff(xcoords)  # node-to-node distances (m)

        # Compute node-to-node delays
        tspikes = self.getSpikesTimings({id: data[id] for id in ids})  # relevant spikes timing dataframe
        delays = np.abs(np.diff(tspikes.values, axis=1))  # node-to-node delays (s)

        # Compute conduction velocities
        distances = np.tile(distances, (1, delays.shape[0]))
        velocities = distances / delays  # m/s

        # Average over all spikes and all segments
        cv = np.mean(velocities)  # m/s

        return cv

    @staticmethod
    def getSpikeAmp(data, key='Vm'):
        amps = np.array([np.ptp(df[key].values) for df in data.values()])
        return amps.min(), amps.max()


class EStimFiber(SennFiber):

    def __init__(self, *args, **kwargs):
        self.connector = None
        super().__init__(*args, **kwargs)

    def createSections(self, ids):
        ''' Create morphological sections. '''
        self.nodes = {id: IintraNode(self.pneuron, id, cell=self) for id in ids}
        self.sections = {id: node.section for id, node in self.nodes.items()}

    def setStimAmps(self, amps):
        ''' Set distributed stimulation amplitudes.

            :param amps: model-sized vector of stimulus amplitudes
        '''
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

    def titrationFunc(self, args):
        psource, A, *args = args
        if psource.is_cathodal:
            A = -A
        data, _ = self.simulate(psource, A, *args)
        return self.isExcited(data)

    def titrate(self, psource, *args, **kwargs):
        Ithr = super().titrate(psource, *args, **kwargs)
        if psource.is_cathodal:
            Ithr = -Ithr
        return Ithr


class IextraFiber(EStimFiber):

    simkey = 'senn_Iextra'
    A_range = (1e0, 1e4)  # mV

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
    A_range = (1e-11, 1e-7)  # nA


class SonicFiber(SennFiber):

    simkey = 'senn_SONIC'
    A_range = (1e0, 6e5)  # Pa

    def __init__(self, *args, a=32e-9, Fdrive=500e3, fs=1., **kwargs):
        # Retrieve point neuron object
        pneuron = args[0]

        # Assign attributes
        self.pneuron = pneuron
        self.mechname = self.pneuron.name + 'auto'
        self.a = a            # m
        self.Fdrive = Fdrive  # Hz
        self.fs = fs          # (-)

        # Initialize connector NBLS objects
        self.connector = SeriesConnector(vref=f'Vm_{self.mechname}', rmin=None)
        self.nbls = NeuronalBilayerSonophore(self.a, self.pneuron, self.Fdrive)

        super().__init__(*args, **kwargs)

    def str_biophysics(self):
        return f'{super().str_biophysics()}, a = {self.a * 1e9:.1f} nm'

    def createSections(self, ids):
        ''' Create morphological sections. '''
        pylkp = self.nbls.getLookup2D(self.Fdrive, self.fs)
        self.nodes = {
            id: SonicNode(
                self.pneuron, id, cell=self, a=self.a, Fdrive=self.Fdrive, fs=self.fs,
                nbls=self.nbls, pylkp=pylkp)
            for id in ids
        }
        self.sections = {id: node.section for id, node in self.nodes.items()}

    def setStimAmps(self, amps):
        ''' Set US stimulation amplitudes.

            :param amps: model-sized vector of stimulus pressure amplitudes (Pa)
        '''
        self.amps = self.preProcessAmps(amps)
        logger.debug('Acoustic pressures: A = [{}] kPa'.format(
            ', '.join([f'{A * 1e-3:.2f}' for A in self.amps])))
        for i, sec in enumerate(self.sections.values()):
            setattr(sec, 'Adrive_{}'.format(self.mechname), self.amps[i] * 1e-3)

    def meta(self, psource, A, pp):
        meta = super().meta(psource, A, pp)
        meta.update({
            'a': self.a,
            'Fdrive': self.Fdrive,
            'fs': self.fs
        })
        return meta

    def filecodes(self, psource, A, pp):
        return {
            **self.modelCodes(),
            'a': '{:.0f}nm'.format(self.a * 1e9),
            'Fdrive': '{:.0f}kHz'.format(self.Fdrive * 1e-3),
            'fs': 'fs{:.0f}%'.format(self.fs * 1e2) if self.fs < 1 else None,
            **psource.filecodes(A),
            'nature': 'CW' if pp.isCW() else 'PW',
            **pp.filecodes()
        }


def myelinatedFiber(fiber_class, pneuron, fiberD, nnodes, rs=1e2, nodeL=2.5e-6, d_ratio=0.7, inter_ratio=100., **kwargs):
    ''' Create a single-cable myelinated fiber model.

        :param pneuron: point-neuron model object
        :param fiberD: fiber outer diameter (m)
        :param nnodes: number of nodes
        :param rs: axoplasmic resistivity (Ohm.cm)
        :param nodeL: nominal node length (m)
        :param d_ratio: ratio of axon (inner-myelin) and fiber (outer-myelin) diameters
        :param inter_ratio: ratio of internode length to fiber diameter
    '''
    if fiberD <= 0:
        raise ValueError('fiber diameter must be positive')
    if nodeL <= 0.:
        raise ValueError('node length must be positive')
    if d_ratio > 1. or d_ratio < 0.:
        raise ValueError('fiber-axon diameter ratio must be in [0, 1]')
    if inter_ratio <= 0.:
        raise ValueError('fiber diameter - internode length ratio must be positive')

    # Define fiber geometrical parameters
    nodeD = d_ratio * fiberD       # m
    nodeL = nodeL                  # m
    interD = d_ratio * fiberD      # m
    interL = inter_ratio * fiberD  # m

    # Create fiber model instance
    return fiber_class(pneuron, nnodes, rs, nodeD, nodeL, interD, interL, **kwargs)


def unmyelinatedFiber(fiber_class, pneuron, fiberD, rs=1e2, fiberL=1e-2, maxNodeL=50e-6, **kwargs):
    ''' Create a single-cable unmyelinated fiber model.

        :param pneuron: point-neuron model object
        :param fiberD: fiber outer diameter (m)
        :param rs: axoplasmic resistivity (Ohm.cm)
        :param fiberL: fiber length (m)
        :param maxNodeL: maximal node length (m)
    '''
    if fiberD <= 0:
        raise ValueError('fiber diameter must be positive')
    if fiberL <= 0:
        raise ValueError('fiber length must be positive')
    if maxNodeL <= 0. or maxNodeL > fiberL:
        raise ValueError('maximum node length must be in [0, fiberL]')

    # Compute number of nodes (ensuring odd number) and node length from fiber length
    nnodes = int(np.ceil(fiberL / maxNodeL))
    if nnodes % 2 == 0:
        nnodes += 1
    nodeL = fiberL / nnodes

    # Define other fiber geometrical parameters (with zero-length internode)
    nodeD = fiberD   # m
    interD = fiberD  # m
    interL = 0.      # m

    # Create fiber model instance
    fiber = fiber_class(pneuron, nnodes, rs, nodeD, nodeL, interD, interL, **kwargs)
    fiber.A_range = (fiber.A_range[0], 1e7)  # mV
    return fiber
