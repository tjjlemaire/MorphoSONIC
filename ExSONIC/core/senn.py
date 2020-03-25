# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 15:18:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-25 12:49:59

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import si_format, logger, plural
from PySONIC.threshold import threshold

from ..utils import getNmodlDir, load_mechanisms, array_print_options
from .nmodel import FiberNeuronModel
from .pyhoc import IClamp, ExtField
from .node import IintraNode, SonicNode
from .connectors import SerialConnectionScheme


class SennFiber(FiberNeuronModel):
    ''' Generic single-cable, Spatially Extended Nonlinear Node (SENN) fiber model. '''

    def __init__(self, pneuron, nnodes, rs, nodeD, nodeL, interD, interL):
        ''' Constructor.

            :param pneuron: point-neuron model object
            :param nnodes: number of nodes
            :param rs: axoplasmic resistivity (Ohm.cm)
            :param nodeD: node diameter (m)
            :param nodeL: node length (m)
            :param interD: internode diameter (m)
            :param interL: internode length (m)
        '''
        # Assign attributes
        self.pneuron = pneuron
        self.rs = rs          # Ohm.cm
        self.nnodes = nnodes
        self.nodeD = nodeD    # m
        self.nodeL = nodeL    # m
        self.interD = interD  # m
        self.interL = interL  # m

        # Compute resistances
        self.R_node = self.axialResistance(self.rs, self.nodeL, self.nodeD)  # Ohm
        self.R_inter = self.axialResistance(self.rs, self.interL, self.interD)  # Ohm

        # Load mechanisms and set appropriate membrane mechanism
        load_mechanisms(getNmodlDir(), self.modfile)

        # Construct model
        self.construct()

    @property
    def nodeL(self):
        return self._nodeL

    @nodeL.setter
    def nodeL(self, value):
        if value <= 0:
            raise ValueError('node length must be positive')
        self._nodeL = value

    @property
    def refsection(self):
        return self.sections['node'][self.nodeIDs[0]]

    @staticmethod
    def getSennArgs(meta):
        return [meta[x] for x in ['nnodes', 'rs', 'nodeD', 'nodeL', 'interD', 'interL']]

    @property
    def meta(self):
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

    @classmethod
    def initFromMeta(cls, meta):
        return cls(getPointNeuron(meta['neuron']), *cls.getSennArgs(meta))

    def construct(self):
        ''' Create and connect node sections with assigned membrane dynamics. '''
        self.createSections()
        self.setGeometry()  # must be called PRIOR to build_custom_topology()
        self.setResistivity()
        self.setTopology()

    def clear(self):
        ''' delete all model sections. '''
        del self.sections

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

    def getNodeCoords(self):
        ''' Return vector of node coordinates along axial dimension, centered at zero (um). '''
        xcoords = (self.nodeL + self.interL) * np.arange(self.nnodes) + self.nodeL / 2
        return xcoords - xcoords[int((self.nnodes - 1) / 2)]

    def isNormalDistance(self, d):
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
        ''' Create morphological sections. '''
        self.nodes = {k: self.createSection(
            k, mech=self.mechname, states=self.pneuron.statesNames()) for k in self.nodeIDs}

    def setGeometry(self):
        ''' Set sections geometry. '''
        logger.debug(f'defining sections geometry: {self.str_geometry()}')
        for sec in self.nodelist:
            sec.setGeometry(self.nodeD, self.nodeL)

    def setResistivity(self):
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

        # Assigning resistivities to sections
        for sec, rho in zip(self.nodelist, rho_nodes):
            sec.setResistivity(rho)

    def setTopology(self):
        ''' Connect the sections in series. '''
        for sec1, sec2 in zip(self.nodelist[:-1], self.nodelist[1:]):
            sec2.connect(sec1)

    @property
    def modelcodes(self):
        return {
            **self.corecodes,
            'nnodes': f'{self.nnodes}node{plural(self.nnodes)}',
            'rs': f'rs{self.rs:.0f}ohm.cm',
            'nodeD': f'nodeD{(self.nodeD * 1e6):.1f}um',
            'nodeL': f'nodeL{(self.nodeL * 1e6):.1f}um',
            'interD': f'interD{(self.interD * 1e6):.1f}um',
            'interL': f'interL{(self.interL * 1e6):.1f}um'
        }

    @property
    def fiberD(self):
        if self.interL == 0.:  # unmyelinated case -> fiberD = nodeD
            return self.nodeD
        else:  # myelinated case: fiberD = interL / 100. (default inter_ratio)
            return self.interL / 100.

    @property
    def is_myelinated(self):
        return self.interL > 0.


class EStimFiber(SennFiber):

    def __init__(self, *args, **kwargs):
        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Set invariant function tables
        self.setFuncTables()

    def setPyLookup(self, *args, **kwargs):
        return IintraNode.setPyLookup(self, *args, **kwargs)

    def titrate(self, source, pp):
        Ithr = threshold(
            lambda x: self.titrationFunc(source.updatedX(-x if source.is_cathodal else x), pp),
            self.getArange(source), rel_eps_thr=1e-2, precheck=False)
        if source.is_cathodal:
            Ithr = -Ithr
        return Ithr


class IintraFiber(EStimFiber):

    simkey = 'senn_Iintra'
    A_range = (1e-12, 1e-7)  # A

    def setDrives(self, source):
        ''' Set distributed stimulation drives. '''
        Iinj = source.computeDistributedAmps(self)['node']
        with np.printoptions(**array_print_options):
            logger.debug(f'Intracellular currents: Iinj = {Iinj} nA')
        self.drives = [IClamp(sec(0.5), I) for sec, I in zip(self.nodelist, Iinj)]


class IextraFiber(EStimFiber):

    simkey = 'senn_Iextra'
    A_range = (1e0, 1e5)  # mV
    use_equivalent_currents = True

    def toInjectedCurrents(self, Ve):
        ''' Convert extracellular potential array into equivalent injected currents.

            :param Ve: model-sized vector of extracellular potentials (mV)
            :return: model-sized vector of intracellular injected currents (nA)
        '''
        Iinj = np.diff(Ve, 2) / (self.R_node + self.R_inter) * self.mA_to_nA  # nA
        return np.pad(Iinj, (1, 1), 'constant')  # zero-padding on both extremities

    def setDrives(self, source):
        ''' Set distributed stimulation drives. '''
        Ve = source.computeDistributedAmps(self)['node']
        with np.printoptions(**array_print_options):
            logger.debug(f'Extracellular potentials: Ve = {Ve} mV')
        if self.use_equivalent_currents:
            # Variant 1: inject equivalent intracellular currents
            self.drives = [IClamp(sec(0.5), I) for sec, I in zip(
                self.nodelist, self.toInjectedCurrents(Ve))]
        else:
            # Variant 2: insert extracellular mechanisms for a more realistic depiction
            # of the extracellular field
            self.drives = [ExtField(sec, ve) for sec, ve in zip(self.nodelist, Ve)]

    def simulate(self, source, pp):
        dt = None if self.use_equivalent_currents else 1e-5
        return super().simulate(source, pp, dt=dt)


class SonicFiber(SennFiber):

    simkey = 'senn_SONIC'
    A_range = (1e0, 6e5)  # Pa

    def __init__(self, *args, a=32e-9, fs=1., **kwargs):
        # Assign attributes
        self.pneuron = args[0]
        self.a = a            # m
        self.fs = fs          # (-)

        # Initialize connection scheme and NBLS object
        self.connection_scheme = SerialConnectionScheme(vref=f'Vm', rmin=None)
        logger.debug(f'Assigning custom connection scheme: {self.connection_scheme}')
        self.nbls = NeuronalBilayerSonophore(self.a, self.pneuron)

        # Initalize reference frequency and python lookup to None
        self.fref = None
        self.pylkp = None

        # Initialize parent class
        super().__init__(*args, **kwargs)

    @classmethod
    def initFromMeta(cls, meta):
        return cls(getPointNeuron(meta['neuron']),
                   *cls.getSennArgs(meta), a=meta['a'], fs=meta['fs'])

    def str_biophysics(self):
        return f'{super().str_biophysics()}, a = {self.a * 1e9:.1f} nm'

    def setPyLookup(self, *args, **kwargs):
        return SonicNode.setPyLookup(self, *args, **kwargs)

    def setDrives(self, source):
        ''' Set distributed stimulation drives. '''
        self.setFuncTables(source.f)
        amps = source.computeDistributedAmps(self)
        with np.printoptions(**array_print_options):
            logger.debug(f'Acoustic pressures: A = {amps * 1e-3} kPa')
        for A, sec in zip(amps, self.nodelist):
            sec.setMechValue('Adrive', A * 1e-3)

    @property
    def meta(self):
        return {
            **super().meta,
            'a': self.a,
            'fs': self.fs
        }

    @property
    def corecodes(self):
        return {
            **super().corecodes,
            'a': f'{self.a * 1e9:.0f}nm',
            'fs': f'fs{self.fs * 1e2:.0f}%' if self.fs <= 1 else None
        }

    def titrate(self, source, pp):
        self.setFuncTables(source.f)  # pre-loading lookups to have a defined Arange
        Arange = self.getArange(source)
        A_conv_thr = np.abs(Arange[1] - Arange[0]) / 1e4
        return threshold(
            lambda x: self.titrationFunc(source.updatedX(x), pp),
            Arange, eps_thr=A_conv_thr, rel_eps_thr=1e-2, precheck=True)
