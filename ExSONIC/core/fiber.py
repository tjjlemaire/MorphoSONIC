# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-14 10:48:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-21 18:01:25

import abc
import numpy as np
from boltons.strutils import cardinalize
from PySONIC.utils import logger, si_format
from PySONIC.threshold import threshold
from PySONIC.postpro import detectSpikes

from .nmodel import SpatiallyExtendedNeuronModel
from ..sources import *
from ..constants import *


class FiberNeuronModel(SpatiallyExtendedNeuronModel):
    ''' Generic interface for fiber (single or double cable) NEURON models. '''

    # Boolean stating whether to use equivalent currents for imposed extracellular voltage fields
    use_equivalent_currents = True

    def __init__(self, fiberD, nnodes=None, fiberL=None, **kwargs):
        ''' Initialization.

            :param fiberD: fiber outer diameter (m)
            :param nnodes: number of nodes
            :param fiberL: length of fiber (m)
        '''
        self.fiberD = fiberD
        self.checkInitArgs(nnodes, fiberL)
        # Compute number of nodes from fiberL if not explicited
        if nnodes is None:
            nnodes = self.getNnodes(fiberL, self.node_to_node_L)
        self.nnodes = nnodes
        super().__init__(**kwargs)

    def checkInitArgs(self, nnodes, fiberL):
        ''' Check that at least one of fiberL or nnodes is provided. '''
        isnone = [x is None for x in [nnodes, fiberL]]
        if all(isnone) or not any(isnone):
            raise ValueError(
                'one (and only one) of "fiberL" or "nnodes" parameters must be provided')

    def getNnodes(self, fiberL, node_to_node_L):
        ''' Compute number of nodes corresponding to a given fiber length. '''
        return int(np.ceil(fiberL / node_to_node_L)) + 1

    def copy(self):
        other = self.__class__(self.fiberD, self.nnodes)
        other.rs = self.rs
        other.pneuron = self.pneuron
        return other

    @property
    def fiberD(self):
        return self._fiberD

    @fiberD.setter
    def fiberD(self, value):
        if value <= 0:
            raise ValueError('fiber diameter must be positive')
        self.set('fiberD', value)

    @property
    def nnodes(self):
        ''' Number of nodes. '''
        return self._nnodes

    @nnodes.setter
    def nnodes(self, value):
        # if value % 2 == 0:
        #     logger.warning(f'even number of nodes ({value})')
        self.set('nnodes', value)

    @property
    def central_ID(self):
        return f'node{self.nnodes // 2}'

    @property
    def ninters(self):
        ''' Number of (abstract) internodal sections. '''
        return self.nnodes - 1

    @property
    def nodeD(self):
        return self._nodeD

    @nodeD.setter
    def nodeD(self, value):
        if value <= 0:
            raise ValueError('node diameter must be positive')
        self.set('nodeD', value)

    @property
    def nodeL(self):
        return self._nodeL

    @nodeL.setter
    def nodeL(self, value):
        if value <= 0:
            raise ValueError('node length must be positive')
        self.set('nodeL', value)

    @property
    def interD(self):
        return self._interD

    @interD.setter
    def interD(self, value):
        if value <= 0:
            raise ValueError('internode diameter must be positive')
        self.set('interD', value)

    @property
    def interL(self):
        return self._interL

    @interL.setter
    def interL(self, value):
        if value < 0:
            raise ValueError('internode length must be positive or null')
        self.set('interL', value)

    @property
    def node_to_node_L(self):
        raise NotImplementedError

    @property
    def rhoa(self):
        ''' Axoplasmic resistivity (Ohm.cm) '''
        return self.rs

    @property
    def R_node(self):
        ''' Node intracellular axial resistance (Ohm). '''
        return self.axialResistance(self.rhoa, self.nodeL, self.nodeD)

    @property
    def R_node_to_node(self):
        raise NotImplementedError

    @property
    def ga_node_to_node(self):
        ''' Node-to-node axial conductance per node unit area (S/cm2). '''
        Ga_node_to_node = 1 / self.R_node_to_node  # S
        Anode = self.nodes[self.central_ID].Am     # cm2
        return Ga_node_to_node / Anode             # S/cm2

    @property
    def nodeIDs(self):
        ''' IDs of the model nodes sections. '''
        return [f'node{i}' for i in range(self.nnodes)]

    @property
    @abc.abstractmethod
    def length(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_myelinated(self):
        raise NotImplementedError

    @property
    def refsection(self):
        return self.nodes[self.nodeIDs[0]]

    @property
    def nonlinear_sections(self):
        return self.nodes

    @property
    def nodelist(self):
        return list(self.nodes.values())

    def str_geometry(self):
        return f'fiberD = {si_format(self.fiberD, 1)}m'

    def str_nodes(self):
        return f'{self.nnodes} {cardinalize("node", self.nnodes)}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.str_geometry()}, {self.str_nodes()})'

    @property
    def meta(self):
        return {
            'simkey': self.simkey,
            'fiberD': self.fiberD,
            'nnodes': self.nnodes
        }

    @staticmethod
    def getMetaArgs(meta):
        return [meta['fiberD'], meta['nnodes']], {}

    @property
    def modelcodes(self):
        return {
            'simkey': self.simkey,
            'fiberD': f'fiberD{(self.fiberD * M_TO_UM):.1f}um',
            'nnodes': self.str_nodes().replace(' ', '')
        }

    def filecodes(self, source, pp, *args):
        codes = super().filecodes(source, pp, *args)
        codes['tstim'] = f'{pp.tstim * 1e3:.2f}ms'
        return codes

    def getXCoords(self):
        return {k: getattr(self, f'get{k.title()}Coords')() for k in self.sections.keys()}

    def getXBounds(self):
        xcoords = self.getXCoords()
        xmin = min(v.min() for v in xcoords.values())
        xmax = max(v.max() for v in xcoords.values())
        return xmin, xmax

    @property
    def z(self):
        return 0.

    def getXZCoords(self):
        return {k: np.vstack((v, np.ones(v.size) * self.z)).T
                for k, v in self.getXCoords().items()}  # m

    @abc.abstractmethod
    def isInternodalDistance(self, d):
        raise NotImplementedError

    @property
    def CV_estimate(self):
        raise NotImplementedError

    @property
    def AP_travel_time_estimate(self):
        ''' Estimated AP travel time (assuming excitation at central node). '''
        return (self.length / 2.) / self.CV_estimate  # s

    def getConductionVelocity(self, data, ids=None, out='median'):
        ''' Compute average conduction speed from simulation results.

            :param data: simulation output dataframe
            :return: conduction speed output (m/s).
        '''
        # By default, consider all fiber nodes
        if ids is None:
            ids = self.nodeIDs.copy()

        # Remove end nodes from calculations if present
        for x in [0, -1]:
            if self.nodeIDs[x] in ids:
                ids.remove(self.nodeIDs[x])

        # Compute spikes timing dataframe (based on nspikes majority voting) and
        # update list of relevant sections accordingly
        tspikes = self.getSpikesTimings({id: data[id] for id in ids})  # (nspikes x nnodes)
        ids = list(tspikes.columns.values)  # (nnodes)

        # Get coordinates of relevant nodes
        indexes = [self.nodeIDs.index(id) for id in ids]  # (nnodes)
        xcoords = self.getNodeCoords()[indexes]  # (nnodes)

        # Compute distances across consecutive nodes only, and associated spiking delays
        # for first spike only
        distances, delays = [], []  # (nnodes - 1)
        for i in range(len(ids) - 1):
            d = xcoords[i] - xcoords[i - 1]
            if self.isInternodalDistance(d):
                distances.append(d)
                dt = np.abs(tspikes.values[0][i] - tspikes.values[0][i - 1])
                delays.append(dt)   # node-to-node delay

        # Compute conduction velocities for each considered node pair
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

    def getEndSpikeTrain(self, data):
        ''' Detect spikes on end node. '''
        ispikes, *_ = detectSpikes(
            data[self.nodeIDs[-1]], key='Vm', mph=SPIKE_MIN_VAMP, mpt=SPIKE_MIN_DT,
            mpp=SPIKE_MIN_VPROM)
        if len(ispikes) == 0:
            return None
        return data.time[ispikes]

    def getEndFiringRate(self, data):
        ''' Compute firing rate from spikes detected on end node. '''
        tspikes = self.getEndSpikeTrain(data)
        if tspikes is None:
            return np.nan
        # return np.mean(1 / np.diff(tspikes))
        return 1 / np.mean(np.diff(tspikes))

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        ids = {
            'proximal': self.nodeIDs[0],
            'distal': self.nodeIDs[-1],
            'central': self.central_ID}
        nspikes = {k: detectSpikes(data[v])[0].size for k, v in ids.items()}
        has_spiked = {k: v > 0 for k, v in nspikes.items()}
        has_spiked['ends'] = has_spiked['proximal'] and has_spiked['distal']
        if not has_spiked['ends'] and has_spiked['central']:
            logger.warning('AP did not reach end nodes')
        return has_spiked['ends']

    def checkForConduction(self, pp):
        ''' Check that a protocol should allow for full AP conduction if excitation is reached. '''
        if pp.toffset < REL_AP_TRAVEL_FACTOR * self.AP_travel_time_estimate:
            AP_travel_str = f'estimated AP travel time: {self.AP_travel_time_estimate * 1e3:.1f} ms'
            raise ValueError(f'offset duration too short for full AP conduction ({AP_travel_str})')

    def titrate(self, source, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param source: source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
        # Check that protocol should allow for full AP conduction, if any
        self.checkForConduction(pp)

        # Run titration procedure
        Arange = self.getArange(source)
        xthr = threshold(
            lambda x: self.titrationFunc(
                self.simulate(source.updatedX(-x if source.is_cathodal else x), pp)[0]),
            Arange,
            x0=self.getStartPoint(Arange),
            eps_thr=self.getAbsConvThr(Arange),
            rel_eps_thr=REL_EPS_THR,
            precheck=source.xvar_precheck)
        if source.is_cathodal:
            xthr = -xthr
        return xthr

    def needsFixedTimeStep(self, source):
        if isinstance(source, (ExtracellularCurrent, GaussianVoltageSource)):
            if not self.use_equivalent_currents:
                return True
        # if self.has_ext_mech:
        #     return True
        return False

    def simulate(self, source, pp):
        return super().simulate(
            source, pp,
            dt=self.fixed_dt if self.needsFixedTimeStep(source) else None)
