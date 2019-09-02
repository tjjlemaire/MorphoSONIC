# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-02 15:56:58

import abc
import pickle
import numpy as np
import pandas as pd
from inspect import signature

from PySONIC.core import Model, PointNeuron, NeuronalBilayerSonophore
from PySONIC.neurons import *
from PySONIC.utils import si_format, pow10_format, logger, plural, binarySearch, pow2Search
from PySONIC.constants import *
from PySONIC.postpro import detectSpikes

from .pyhoc import *
from .node import Node, IintraNode, SonicNode
from .connectors import SeriesConnector
from ..constants import *


class SennFiber(metaclass=abc.ABCMeta):

    tscale = 'ms'  # relevant temporal scale of the model

    @property
    @abc.abstractmethod
    def simkey(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    def __init__(self, pneuron, fiberD, nnodes, rs=1e2, nodeL=2.5e-6, d_ratio=0.7):
        ''' Initialize fiber model.

            :param pneuron: point-neuron model object
            :param fiberD: fiber outer diameter (m)
            :param nnodes: number of nodes
            :param rs: axoplasmic resistivity (Ohm.cm)
            :param nodeL: nominal node length (m)
            :param d_ratio: ratio of axon (inner-myelin) and fiber (outer-myelin) diameters
        '''
        if not isinstance(pneuron, PointNeuron):
            raise TypeError(f'{pneuron} is not a valid PointNeuron instance')
        if nnodes % 2 == 0:
            raise ValueError('Number of nodes must be odd')
        if fiberD <= 0:
            raise ValueError('Fiber diameter must be positive')

        # Assign attributes
        self.pneuron = pneuron
        self.rs = rs  # Ohm.cm
        self.nnodes = nnodes
        self.fiberD = fiberD
        self.d_ratio = d_ratio

        # Define fiber geometrical parameters
        self.nodeD = self.d_ratio * self.fiberD  # m
        self.nodeL = 2.5e-6  # m
        self.interD = self.d_ratio * self.fiberD  # m
        self.interL = 100 * self.fiberD  # m

        # Compute nodal and internodal axial resistance ()
        self.R_node = self.resistance(self.nodeD, self.nodeL)  # Ohm
        self.R_inter = self.resistance(self.interD, self.interL)  # Ohm

        # Create node sections with assigned membrane dynamics
        self.ids = [f'node{i}' for i in range(self.nnodes)]
        self.createSections(self.ids)
        self.mechname = self.nodes[self.ids[0]].mechname
        self.setGeometry()  # must be called PRIOR to build_custom_topology()
        self.setResistivity()
        self.setTopology()

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{self.__class__.__name__}({self.strBiophysics()}, {self.strNodes()}, d = {self.fiberD * 1e6:.1f} um)'

    def reset(self):
        ''' clear all sections and re-initialize model. '''
        self.clear()
        self.__init__(self.pneuron, self.fiberD, self.nnodes, rs=self.rs, nodeL=self.nodeL, d_ratio=self.d_ratio)

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

    def strNodes(self):
        return f'{self.nnodes} node{plural(self.nnodes)}'

    def strResistivity(self):
        return f'rs = ${pow10_format(self.rs)}$ Ohm.cm'

    def strBiophysics(self):
        return f'{self.pneuron.name} neuron'

    def strGeom(self):
        ''' Format model geometrical parameters into string. '''
        params = {
            'nodeD': self.nodeD,
            'nodeL': self.nodeL,
            'interD': self.interD,
            'interL': self.interL
        }
        lbls = {key: f'{key} = {si_format(val, 1)}m' for key, val in params.items()}
        return ', '.join(lbls.values())

    def pprint(self):
        ''' Pretty-print naming of the model instance. '''
        return f'{self.pneuron.name} neuron, {self.strNodes()}, {self.strResistivity()}, {self.strGeom()}'

    @property
    @abc.abstractmethod
    def createSections(self, ids):
        ''' Create morphological sections. '''
        return NotImplementedError

    def clear(self):
        del self.sections

    def setGeometry(self):
        ''' Set sections geometry. '''
        logger.debug(f'defining sections geometry: {self.strGeom()}')
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

    def setResistivity(self):
        ''' Set sections axial resistivity, corrected to account for internodes and membrane capacitance
            in the Q-based differentiation scheme. '''

        logger.debug(f'nominal nodal resistivity: rs = {self.rs:.0f} Ohm.cm')
        rho_nodes = np.ones(self.nnodes) * self.rs  # Ohm.cm

        # Adding extra resistivity to account for half-internodal resistance
        # for each connected side of each node
        logger.debug('adding extra-resistivity to account for internodal resistance')
        R_extra = np.array([self.R_inter / 2] + [self.R_inter] * (self.nnodes - 2) + [self.R_inter / 2])  # Ohm
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

    @property
    @abc.abstractmethod
    def setStimAmps(self, amps, config):
        ''' Set distributed stimulus amplitudes. '''
        return NotImplementedError

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        for sec in self.sections.values():
            setattr(sec, f'stimon_{self.mechname}', value)
        return value

    def toggleStim(self):
        ''' Toggle stimulation and set appropriate next toggle event. '''
        # OFF -> ON at pulse onset
        if self.stimon == 0:
            self.stimon = self.setStimON(1)
            self.cvode.event(min(self.tstim, h.t + self.Ton), self.toggleStim)
        # ON -> OFF at pulse offset
        else:
            self.stimon = self.setStimON(0)
            if (h.t + self.Toff) < self.tstim - h.dt:
                self.cvode.event(h.t + self.Toff, self.toggleStim)

        # Re-initialize cvode if active
        if self.cvode.active():
            self.cvode.re_init()
        else:
            h.fcurrent()

    def integrate(self, tstop, tstim, PRF, DC, dt, atol):
        ''' Integrate the model differential variables for a given duration, while updating the
            value of the boolean parameter stimon during ON and OFF periods throughout the numerical
            integration, according to stimulus parameters.

            Integration uses an adaptive time step method by default.

            :param tstop: duration of numerical integration (s)
            :param tstim: stimulus duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step (s). If provided, the fixed time step method is used.
            :param atol: absolute error tolerance (default = 1e-3). If provided, the adaptive
                time step method is used.
        '''
        # Convert input parameters to NEURON units
        tstim *= 1e3
        tstop *= 1e3
        PRF /= 1e3
        if dt is not None:
            dt *= 1e3

        # Update PRF for CW stimuli to optimize integration
        if DC == 1.0:
            PRF = 1 / tstim

        # Set pulsing parameters used in CVODE events
        self.Ton = DC / PRF
        self.Toff = (1 - DC) / PRF
        self.tstim = tstim

        # Set integration parameters
        h.secondorder = 2
        self.cvode = h.CVode()
        if dt is not None:
            h.dt = dt
            self.cvode.active(0)
            logger.debug(f'fixed time step integration (dt = {h.dt} ms)')
        else:
            self.cvode.active(1)
            if atol is not None:
                def_atol = self.cvode.atol()
                self.cvode.atol(atol)
                logger.debug(f'adaptive time step integration (atol = {self.cvode.atol()})')

        # Initialize
        self.stimon = self.setStimON(0)
        h.finitialize(self.pneuron.Qm0() * 1e5)  # nC/cm2
        self.stimon = self.setStimON(1)
        self.cvode.event(self.Ton, self.toggleStim)

        # Integrate
        while h.t < tstop:
            h.fadvance()

        # Set absolute error tolerance back to default value if changed
        if atol is not None:
            self.cvode.atol(def_atol)

        return 0

    @Model.checkTitrate('A')
    @Model.addMeta
    def simulate(self, psource, A, tstim, toffset, PRF, DC, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param psource: point source object
            :param A: stimulus amplitude (A)
            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle
            :param dt: integration time step (s)
        '''
        logger.info(
            '%s: simulation @ %s, %s, t = %ss (%ss offset)%s', self, repr(psource),
            psource.strAmp(A), *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(
                si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

        # Set recording vectors
        t = setTimeProbe()
        stim = setStimProbe(self.sections[self.ids[0]], self.mechname)
        probes = {k: self.nodes[k].setProbesDict(v) for k, v in self.sections.items()}

        # Set distributed stimulus amplitudes and integrate model
        self.setStimAmps(psource.computeNodesAmps(self, A))
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

        # Store output in dataframes§
        data = {}
        for id in self.sections.keys():
            data[id] = pd.DataFrame({
                't': vec_to_array(t) * 1e-3,  # s
                'stimstate': vec_to_array(stim)
            })
            for k, v in probes[id].items():
                data[id][k] = vec_to_array(v)
            data[id].loc[:,'Qm'] *= 1e-5  # C/m2

        # Resample data to regular sampling rate
        data = {id: Node.resample(df, DT_TARGET) for id, df in data.items()}

        # Prepend initial conditions (prior to stimulation)
        data = {id: Node.prepend(df) for id, df in data.items()}

        return data

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        nspikes_start = detectSpikes(data[self.ids[0]])[0].size
        nspikes_end = detectSpikes(data[self.ids[-1]])[0].size
        return nspikes_start > 0 and nspikes_end > 0

    @property
    @abc.abstractmethod
    def getArange(self, psource):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def titrationFunc(self, psource):
        return NotImplementedError

    def titrate(self, psource, tstim, toffset, PRF=100., DC=1.):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given duration, PRF and duty cycle.

            :param psource: point source object
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param method: integration method
            :return: determined threshold amplitude (Pa)
        '''
        Amin, Amax = self.getArange(psource)
        A_conv_thr = np.abs(Amax - Amin) / 1e4
        Athr = pow2Search(self.titrationFunc, [psource, tstim, toffset, PRF, DC], 1, Amin, Amax)
        Arange = (Athr / 2, Athr)
        return binarySearch(
            self.titrationFunc, [psource, tstim, toffset, PRF, DC], 1, Arange, A_conv_thr)

    def meta(self, psource, A, tstim, toffset, PRF, DC):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'fiberD': self.fiberD,
            'nnodes': self.nnodes,
            'rs': self.rs,
            'psource': psource,
            'A': A,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC
        }

    def filecode(self, *args):
        ''' Generate file code given a specific combination of model input parameters. '''
        # If meta dictionary was passed, generate inputs list from it
        if len(args) == 1 and isinstance(args[0], dict):
            meta = args[0]
            meta.pop('tcomp', None)
            sig = signature(self.meta).parameters
            args = [meta[k] for k in sig]

        # Create file code by joining string-encoded inputs with underscores
        codes = self.filecodes(*args).values()
        return '_'.join([x for x in codes if x is not None])

    def simAndSave(self, outdir, *args):
        ''' Simulate the model and save the results in a specific output directory. '''
        out = self.simulate(*args)
        if out is None:
            return None
        data, meta = out
        if None in args:
            args = list(args)
            iNone = next(i for i, arg in enumerate(args) if arg is None)
            sig = signature(self.meta).parameters
            key = list(sig.keys())[iNone]
            args[iNone] = meta[key]
        fpath = '{}/{}.pkl'.format(outdir, self.filecode(*args))
        with open(fpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', fpath)
        return fpath

    def getSpikesTimings(self, data, zcross=True):
        ''' Return an array containing occurence times of spikes detected on a collection of nodes.

            :param data: simulation output dataframe
            :param zcross: boolean stating whether to consider ascending zero-crossings preceding peaks
                as temporal reference for spike occurence timings
            :return: dictionary of spike occurence times (s) per node.
        '''

        tspikes = {}
        nspikes = None
        for id in self.ids:

            # Detect spikes on current trace
            ispikes, *_ = detectSpikes(data[id], key='Vm', mph=SPIKE_MIN_VAMP, mpt=SPIKE_MIN_DT,
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
                t = data[id]['t'].values  # s
                Vm = data[id]['Vm'].values  # mV
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

    def getConductionVelocity(self, data):
        ''' Compute average conduction speed from simulation results.

            :param data: simulation output dataframe
            :return: array of condiction speeds per spike (m/s).
        '''
        d = np.diff(self.getNodeCoords())[0]  # node-to-node distance (m)
        tspikes = self.getSpikesTimings(data)  # spikes timing dataframe
        delays = np.abs(np.diff(tspikes.values, axis=1))  # node-to-node delays (s)
        delays = delays[:, 1:-1]  # remove delays from both extremity segments
        cv = d / delays  # node-to-node conduction velocities (m/s)
        return np.mean(cv)

    @staticmethod
    def getSpikeAmp(data, key='Vm'):
        amps = np.array([np.ptp(df[key].values) for df in data.values()])
        return amps.min(), amps.max()

    def filecodes(self, *args):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nnodes': f'{self.nnodes}node{plural(self.nnodes)}',
            'fiberD': f'{(self.fiberD * 1e6):.2f}um',
        }


class EStimSennFiber(SennFiber):

    def __init__(self, pneuron, fiberD, nnodes, **kwargs):
        mechname = pneuron.name + 'auto'
        # self.connector = SeriesConnector(vref=f'Vm_{mechname}', rmin=None)
        # self.connector = SeriesConnector(vref='v', rmin=None)
        self.connector = None
        super().__init__(pneuron, fiberD, nnodes, **kwargs)

    def createSections(self, ids):
        ''' Create morphological sections. '''
        self.nodes = {id: IintraNode(self.pneuron, id, cell=self) for id in ids}
        self.sections = {id: node.section for id, node in self.nodes.items()}

    def preProcessAmps(self, amps):
        return NotImplementedError

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

    def getPltVars(self, *args, **kwargs):
        return self.pneuron.getPltVars(*args, **kwargs)

    def getPltScheme(self, *args, **kwargs):
        return self.pneuron.getPltScheme(*args, **kwargs)

    def getArange(self, psource):
        return [psource.computeSourceAmp(self, x) for x in self.A_range]

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

    def filecodes(self, *args):
        psource, A, tstim, *args = args
        fiber_codes = super().filecodes(*args)
        psource_codes = psource.filecodes(A)
        pneuron_codes = self.pneuron.filecodes(A, tstim, *args)
        for key in ['simkey', 'neuron', 'Astim', 'tstim']:
            del pneuron_codes[key]
        return {**fiber_codes, **psource_codes, **{'tstim': '{:.2f}ms'.format(tstim * 1e3)},**pneuron_codes}


class VextSennFiber(EStimSennFiber):

    simkey = 'senn_Vext'
    A_range = (1e0, 1e3)  # mV

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


class IinjSennFiber(EStimSennFiber):

    simkey = 'senn_Iinj'
    A_range = (1e-11, 1e-7)  # nA

    def preProcessAmps(self, Iinj):
        ''' Assign array of injeccted injected currents.

            :param Iinj: model-sized vector of intracellular injected currents (nA)
            :return: model-sized vector of intracellular injected currents (nA)
        '''
        return Iinj


class AStimSennFiber(SennFiber):

    simkey = 'senn_US'
    A_range = (1e0, 6e5)  # Pa

    def __init__(self, pneuron, fiberD, nnodes, a=32e-9, Fdrive=500e3, fs=1., **kwargs):
        mechname = pneuron.name + 'auto'
        self.a = a            # m
        self.Fdrive = Fdrive  # Hz
        self.fs = fs          # (-)
        self.connector = SeriesConnector(vref=f'Vm_{mechname}', rmin=None)
        self.nbls = NeuronalBilayerSonophore(self.a, pneuron, self.Fdrive)
        super().__init__(pneuron, fiberD, nnodes, **kwargs)

    def createSections(self, ids):
        ''' Create morphological sections. '''
        self.nodes = {
            id: SonicNode(
                self.pneuron, id, cell=self, a=self.a, Fdrive=self.Fdrive, fs=self.fs, nbls=self.nbls)
            for id in ids
        }
        self.sections = {id: node.section for id, node in self.nodes.items()}

    def getPltVars(self, *args, **kwargs):
        return self.nbls.getPltVars(*args, **kwargs)

    def getPltScheme(self, *args, **kwargs):
        return self.nbls.getPltScheme(*args, **kwargs)

    def preProcessAmps(self, A):
        ''' Assign array of US pressures.

            :param A: model-sized vector of US pressures (Pa)
            :return: model-sized vector of US pressures (Pa)
        '''
        return A

    def setStimAmps(self, amps):
        ''' Set US stimulation amplitudes.

            :param amps: model-sized vector of stimulus pressure amplitudes (Pa)
        '''
        self.amps = self.preProcessAmps(amps)
        logger.debug('Acoustic pressures: A = [{}] kPa'.format(
            ', '.join([f'{A:.2f}' for A in self.amps])))
        for i, sec in enumerate(self.sections.values()):
            setattr(sec, 'Adrive_{}'.format(self.mechname), self.amps[i])

    def getArange(self, psource):
        return [psource.computeSourceAmp(self, x) for x in self.A_range]

    def titrationFunc(self, args):
        psource, A, *args = args
        data, _ = self.simulate(psource, A, *args)
        return self.isExcited(data)

    def meta(self, psource, A, tstim, toffset, PRF, DC):
        meta = super().meta(psource, A, tstim, toffset, PRF, DC)
        meta.update({
            'a': self.a,
            'Fdrive': self.Fdrive,
            'fs': self.fs
        })
        return meta

    def filecodes(self, *args):
        psource, A, tstim, *args = args
        fiber_codes = super().filecodes(*args)
        psource_codes = psource.filecodes(A)
        pneuron_codes = self.pneuron.filecodes(A, tstim, *args)
        for key in ['simkey', 'neuron', 'Astim', 'tstim']:
            del pneuron_codes[key]
        return {**fiber_codes, **psource_codes, **{'tstim': '{:.2f}ms'.format(tstim * 1e3)},**pneuron_codes}
