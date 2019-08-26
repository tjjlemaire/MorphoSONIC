# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-26 10:34:29

import abc
import pickle
import numpy as np
import pandas as pd
from inspect import signature

from PySONIC.core import Model
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

    def __init__(self, pneuron, fiberD, nnodes, rs=1e2):

        if nnodes % 2 == 0:
            raise ValueError('Number of nodes must be odd')

        # Assign attributes
        self.pneuron = pneuron
        self.rs = rs  # Ohm.cm
        self.nnodes = nnodes
        self.fiberD = fiberD

        # Define fiber geometrical parameters
        self.nodeD = 0.7 * fiberD  # m
        self.nodeL = 2.5e-6  # m
        self.interD = fiberD  # m
        self.interL = 100 * fiberD  # m

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
        return f'SennFiber({self.strBiophysics()}, {self.strNodes()}, d = {self.interD * 1e6:.1f} um)'

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
        return 4 * rs * L / (np.pi * d**2) * 1e-2

    def setResistivity(self):
        ''' Set sections axial resistivity, corrected to account for internodes and membrane capacitance
            in the Q-based differentiation scheme. '''

        logger.debug(f'setting nodal resistivity to {self.rs:.0f} Ohm.cm')
        for sec in self.sections.values():
            sec.Ra = self.rs

        logger.debug('adjusting resistivities to account for internodal sections')
        for i, sec in enumerate(self.sections.values()):
            R_extra = 0
            for ind in [0, self.nnodes - 1]:
                if i != ind:
                    R_extra += self.R_inter / 2
            sec.Ra *= (1 + R_extra / self.R_node)

        # In case the axial coupling variable is v (an alias for membrane charge density),
        # multiply resistivity by membrane capacitance to ensure consistency of Q-based
        # differential scheme, where Iax = dV / r = dQ / (r * cm)
        if self.connector is None or self.connector.vref == 'v':
            logger.debug('adjusting resistivities to account for Q-based differential scheme')
        for sec in self.sections.values():
            sec.Ra *= self.pneuron.Cm0 * 1e2

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

            :param psource: point source object with 2d-location relative to the fiber (m)
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
        self.setStimAmps(psource.computeDistributedAmps(A, self))
        self.integrate(tstim + toffset, tstim, PRF, DC, dt, atol)

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

        # Resample data to regular sampling rate
        data = {id: Node.resample(df, DT_EFFECTIVE / 10) for id, df in data.items()}

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

            :param psource: point source object with 2d-location (in m) and
                stimulus amplitude (in modality units)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param method: integration method
            :return: determined threshold amplitude (Pa)
        '''
        Amin, Amax = self.getArange(psource)
        A_conv_thr = np.abs(Amax - Amin) / 1e3
        Athr = pow2Search(self.titrationFunc, [psource, tstim, toffset, PRF, DC], 1, Amin, Amax)
        Arange = (Athr / 2, Athr)
        return binarySearch(
            self.titrationFunc, [psource, tstim, toffset, PRF, DC], 1, Arange, A_conv_thr)

    def meta(self, psource, A, tstim, toffset, PRF, DC):
        meta = {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'fiberD': self.fiberD,
            'nnodes': self.nnodes,
            'rs': self.rs,
            'psource': psource,
            # 'psource_z': psource.z,
            'A': A,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC
        }
        return meta

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


class VextSennFiber(SennFiber):

    simkey = 'senn_ESTIM'
    Ve_range = (1e0, 1e3)  # mV

    def __init__(self, pneuron, fiberD, nnodes, **kwargs):
        mechname = pneuron.name + 'auto'
        self.connector = SeriesConnector(vref=f'Vm_{mechname}', rmin=None)
        # self.connector = SeriesConnector(vref='v', rmin=None)
        self.connector = None
        super().__init__(pneuron, fiberD, nnodes, **kwargs)

    def createSections(self, ids):
        self.nodes = {id: IintraNode(self.pneuron, id, cell=self) for id in ids}
        self.sections = {id: node.section for id, node in self.nodes.items()}

    def setStimAmps(self, Ve):
        ''' Insert extracellular mechanism into node sections and set extracellular potential values.

            :param Vexts: model-sized vector of extracellular potentials (mV)
            or single value (assigned to first node)
            :return: section-specific labels
        '''
        logger.debug('Extracellular potentials: Ve = [{}] mV'.format(
            ', '.join([f'{v:.2f}' for v in Ve])))

        # Compute intracelular currents equivalent to extracelular potential drive
        self.Iinj =  np.diff(Ve, 2) / (self.R_node + self.R_inter) * 1e6  # nA
        self.Iinj = np.pad(self.Iinj, (1, 1), 'constant')  # zero-padding on both extremities

        # Add zero-padding on both extremities
        logger.debug('Equivalent intracellular currents: Iinj = [{}] nA'.format(
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

    def getArange(self, psource):
        return [psource.reverseComputeAmp(self, x) for x in self.Ve_range]

    def titrationFunc(self, args):
        psource, A, *args = args
        if psource.is_cathodal:
            A = -A
        data, _ = self.simulate(psource, A, *args)
        is_excited = self.isExcited(data)
        return is_excited

    def titrate(self, psource, *args, **kwargs):
        Ithr = super().titrate(psource, *args, **kwargs)
        if psource.is_cathodal:
            Ithr = -Ithr
        return Ithr

    def getPltVars(self, *args, **kwargs):
        return self.pneuron.getPltVars(*args, **kwargs)

    def getPltScheme(self, *args, **kwargs):
        return self.pneuron.getPltScheme(*args, **kwargs)

    def filecodes(self, *args):
        psource, A, tstim, *args = args
        codes = {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nnodes': f'{self.nnodes}node{plural(self.nnodes)}',
            'fiberD': f'{(self.fiberD * 1e6):.2f}um',
            'psource': f'ps({(psource.x * 1e3):.1f},{(psource.z * 1e3):.1f})mm',
            'A': f'{(A * 1e3):.2f}mA',
            'tstim': '{:.2f}ms'.format(tstim * 1e3),
        }
        pneuron_codes = self.pneuron.filecodes(A, tstim, *args)
        for key in ['simkey', 'neuron', 'Astim', 'tstim']:
            del pneuron_codes[key]
        codes.update(pneuron_codes)
        return codes



