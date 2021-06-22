# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-22 11:56:23

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, isWithin

from ..constants import *
from ..core import FiberNeuronModel, addSonicFeatures


class SingleCableFiber(FiberNeuronModel):
    ''' Generic single-cable fiber model. '''

    def copy(self):
        other = super().copy()
        other.innerD_ratio = self.innerD_ratio
        other.interL_ratio = self.interL_ratio
        return other

    @property
    def innerD_ratio(self):
        return self._innerD_ratio

    @innerD_ratio.setter
    def innerD_ratio(self, value):
        value = isWithin('axoplasm - fiber diameter ratio', value, (0., 1.))
        self.set('innerD_ratio', value)

    @property
    def interL_ratio(self):
        return self._interL_ratio

    @interL_ratio.setter
    def interL_ratio(self, value):
        if value < 0:
            raise ValueError('internode length - fiber diameter ratio must be positive or null')
        self.set('interL_ratio', value)

    @property
    def nodeD(self):
        return self.innerD_ratio * self.fiberD

    @property
    def interD(self):
        return self.innerD_ratio * self.fiberD

    @property
    def interL(self):
        return self.interL_ratio * self.fiberD

    @property
    def node_to_node_L(self):
        return self.nodeL + self.interL

    @property
    def R_inter(self):
        ''' Internodal intracellular axial resistance (Ohm). '''
        return self.axialResistance(self.rs, self.interL, self.interD)

    @property
    def R_node_to_node(self):
        ''' Nonde-to-node intracellular axial resistance (Ohm). '''
        return self.R_node + self.R_inter

    def clearSections(self):
        ''' delete all model sections. '''
        self.nodes = None

    def getNodeCoords(self):
        ''' Return vector of node coordinates along axial dimension, centered at zero (um). '''
        xcoords = (self.nodeL + self.interL) * np.arange(self.nnodes) + self.nodeL / 2
        return xcoords - xcoords[int((self.nnodes - 1) / 2)]

    def isInternodalDistance(self, d):
        return np.isclose(d, (self.nodeL + self.interL))

    @property
    def length(self):
        return self.nnodes * self.nodeL + (self.nnodes - 1) * self.interL

    @property
    def sections(self):
        return {'node': self.nodes}

    @property
    def seclist(self):
        return self.nodelist

    def createSections(self):
        self.nodes = {k: self.createSection(
            k, mech=self.mechname, states=self.pneuron.statesNames()) for k in self.nodeIDs}

    def setGeometry(self):
        logger.debug(f'defining sections geometry: {self.str_geometry()}')
        for sec in self.nodelist:
            sec.setGeometry(self.nodeD, self.nodeL)

    def setResistivity(self):
        logger.debug(f'nominal nodal resistivity: rs = {self.rs:.0f} Ohm.cm')
        rho_nodes = np.ones(self.nnodes) * self.rs  # Ohm.cm

        # Adding extra resistivity to account for half-internodal resistance
        # for each connected side of each node
        if self.R_inter > 0:
            logger.debug('adding extra-resistivity to account for internodal resistance')
            rho_extra = self.R_extra * self.rs / self.R_node  # Ohm.cm
            rho_nodes += rho_extra  # Ohm.cm

        # Assigning resistivities to sections
        for sec, rho in zip(self.nodelist, rho_nodes):
            sec.setResistivity(rho)

    @property
    def R_extra(self):
        ''' Vector of extra internodal resistances to add to nodal axial resistances. '''
        return np.ones(self.nnodes) * self.R_inter  # Ohm

    def setTopology(self):
        for i in range(self.nnodes - 1):
            self.connect('node', i, 'node', i + 1)

    def toInjectedCurrents(self, Ve):
        Iinj = np.diff(Ve['node'], 2) / (self.R_node + self.R_inter) * MA_TO_NA  # nA
        return {'node': np.pad(Iinj, (1, 1), 'constant')}  # zero-padding on both extremities


@addSonicFeatures
class SennFiber(SingleCableFiber):
    ''' Spatially Extended Nonlinear Node (SENN) myelinated fiber model from Reilly 1985.

        Reference: Reilly, J.P., Freeman, V.T., and Larkin, W.D. (1985). Sensory effects
        of transient electrical stimulation--evaluation with a neuroelectric model.
        IEEE Trans Biomed Eng 32, 1001–1011.
    '''
    simkey = 'senn'
    is_myelinated = True
    _innerD_ratio = 0.7                  # axoplasm - fiber diameter ratio (McNeal 1976)
    _interL_ratio = 100                  # internode length - fiber diameter ratio (McNeal 1976)
    _nodeL = 2.5e-6                      # node length (m, from McNeal 1976)
    _rs = 110.0                          # axoplasm resistivity (Ohm.cm, from McNeal 1976)
    _pneuron = getPointNeuron('FHnode')  # amphibian node membrane equations

    @property
    def CV_estimate(self):
        ''' Estimated diameter-dependent conduction veclocity
            (from linear fit across 5-20 um range)
        '''
        return 4.2 * self.fiberD * M_TO_UM  # m/s


@addSonicFeatures
class SweeneyFiber(SingleCableFiber):
    ''' SENN fiber variant from Sweeney 1987 with mammalian (rabbit) nodal membrane dynamics.

        Reference: Sweeney, J.D., Mortimer, J.T., and Durand, D. (1987). Modeling of
        mammalian myelinated nerve for functional neuromuscular stimulation. IEEE 9th
        Annual Conference of the Engineering in Medicine and Biology Society 3, 1577–1578.
    '''
    simkey = 'sweeney'
    is_myelinated = True
    _innerD_ratio = 0.6                  # internode length - fiber diameter ratio
    _interL_ratio = 100                  # internode length - fiber diameter ratio (McNeal 1976)
    _nodeL = 1.5e-6                      # node length (m)
    _rs = 54.7                           # axoplasm resistivity (Ohm.cm)
    _pneuron = getPointNeuron('SWnode')  # mammalian node membrane equations


@addSonicFeatures
class UnmyelinatedFiber(SingleCableFiber):
    ''' Single-cable unmyelinated fiber model from Sundt 2015.

        Reference: Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
        root ganglia in an unmyelinated sensory neuron: a modeling study.
        Journal of Neurophysiology (2015)

    '''
    simkey = 'unmyelinated'
    is_myelinated = False
    _innerD_ratio = 1.0                  # no myelin impacting the ratio
    _interL_ratio = 0                    # zero internode length
    _rs = 100.0                          # axoplasm resistivity (Ohm.cm)
    _pneuron = getPointNeuron('SUseg')   # membrane equations

    # Critical values established from convergence study (m)
    abs_NodeL_thr = 22e-6                                          # nodeL: absolute threshold
    lin_NodeL_thr = lambda x: 16.3 * x + 9.1e-6                 # nodeL: fiberD-dep threshold
    fiberL_thr = 3e-3                                              # fiberL: absolute threshold

    @classmethod
    def NodeL_thr(cls, fiberD):
        ''' nodeL: general threshold '''
        return min(cls.lin_NodeL_thr(fiberD), cls.abs_NodeL_thr)

    def __init__(self, fiberD, nnodes=None, fiberL=None, maxNodeL=None, **kwargs):
        ''' Initialization.

            :param fiberD: fiber outer diameter (m)
            :param fiberL: fiber length (m)
            :param maxNodeL: maximal node length (m)
        '''
        self.checkInitArgs(nnodes, fiberL)
        # Compute maxNodeL if not provided
        if maxNodeL is None:
            maxNodeL = self.NodeL_thr(fiberD)
        # Compute fiberL (from nnodes) if not provided
        if fiberL is None:
            fiberL = nnodes * maxNodeL
        # Print warning if fiberL smaller than convergence threshold
        if fiberL <= self.fiberL_thr:
            logger.warning(
                f'convergence not guaranteed for fiberL < {self.fiberL_thr * M_TO_MM:.1f} mm')
        # Check that maxNodeL is smaller than fiberL
        maxNodeL = isWithin('maximum node length', maxNodeL, (0., fiberL))
        # Compute number of nodes if not explicited
        if nnodes is None:
            nnodes = self.getNnodes(fiberL, maxNodeL)
        # Compute precise nodeL as fiberL / nnodes ratio
        self.nodeL = fiberL / nnodes
        # Initialize with correct number of nodes
        super().__init__(fiberD, nnodes=nnodes, **kwargs)

    @property
    def CV_estimate(self):
        ''' Estimated diameter-dependent conduction veclocity
            (from linear fit across 0.2-1.4 um range)
        '''
        return 0.3 * self.fiberD * M_TO_UM + 0.2  # m/s


# @addSonicFeatures
# class HHFiber(SingleCableFiber):
#     ''' Single-cable unmyelinated fiber model from Hodgkin-Huxley 1952. '''
#     simkey = 'hh'
#     is_myelinated = False
#     _innerD_ratio = 1.0                  # no myelin impacting the ratio
#     _interL_ratio = 0                    # zero internode length
#     _rs = 100.0                          # axoplasm resistivity (Ohm.cm)
#     _pneuron = getPointNeuron('HHseg')   # membrane equations

#     # Critical values established from convergence study (m)
#     abs_NodeL_thr = 22e-6                              # absolute max. node length
#     lindep_NodeL_thr = lambda _, x: 16.3 * x + 9.1e-6  # fiber diameter dependent max. node length
#     fiberL_thr = 3e-3                                  # minimal fiber length

#     def __init__(self, fiberD, nnodes=None, fiberL=5e-3, maxNodeL=None, **kwargs):
#         ''' Initialization.

#             :param fiberD: fiber outer diameter (m)
#             :param fiberL: fiber length (m)
#             :param maxNodeL: maximal node length (m)
#         '''
#         if nnodes is None and fiberL is None:
#             raise ValueError('at least one of "fiberL" or "nnodes" parameters must be provided')
#         if maxNodeL is None:
#             maxNodeL = min(self.lindep_NodeL_thr(fiberD), self.abs_NodeL_thr)
#         if fiberL is None:
#             fiberL = nnodes * maxNodeL
#         if fiberL <= self.fiberL_thr:
#             logger.warning(f'fiber length must be at least {self.fiberL_thr * M_TO_MM:.1f} mm')
#         maxNodeL = isWithin('maximum node length', maxNodeL, (0., fiberL))

#         # Compute number of nodes (ensuring odd number) and node length from fiber length
#         nnodes = int(np.ceil(fiberL / maxNodeL))
#         if nnodes % 2 == 0:
#             nnodes += 1
#         self.nodeL = fiberL / nnodes

#         # Initialize with correct number of nodes
#         super().__init__(fiberD, nnodes, **kwargs)
