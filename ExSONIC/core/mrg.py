# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-27 23:08:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-28 17:02:23

import numpy as np

from PySONIC.core import Lookup


def getMRGparameters(fiberD, interp_method='quadratic'):
    # Refs from McIntyre 2002
    refs = {'fiberD': np.array([5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]) * 1e-6}
    tables = {
        'nodeD': np.array([1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]) * 1e-6,
        'interD': np.array([3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]) * 1e-6,
        'interL': np.array([500., 750., 1000., 1150., 1250., 1350., 1400., 1450., 1500.]) * 1e-6,
        'flutL': np.array([35., 38., 40., 46., 50., 54., 56., 58., 60.]) * 1e-6,
        'nlayers': np.array([80, 100, 110, 120, 130, 135, 140, 145, 150]),
        # 'G': np.array([0.605, 0.661, 0.690, 0.700, 0.719, 0.739, 0.767, 0.791]),
    }
    mrg_lkp = Lookup(refs, tables, interp_method=interp_method)
    return list(mrg_lkp.project('fiberD', fiberD).tables.values())

# out = getMRGparameters(6.0e-6)



class MRGFiber(NeuronModel):
    ''' Generic double-cable, myelinated fiber model.

        The model integrates an axon model developed by McIntyre 2002.
        This extends the McIntyre model to allow any diameter to be used
    '''

    tscale = 'ms'  # relevant temporal scale of the model

    # Constant geometrical parameters
    nodeL = 1.
    mysaL = 3.
    mysa_space = 0.002
    flut_space = 0.004
    stin_space = 0.004

    def __init__(self, pneuron, nnodes, rs, nodeD, interD, interL, flutL, nlayers):
        ''' Constructor.

            :param pneuron: point-neuron model object
            :param nnodes: number of nodes
            :param rs: axoplasmic resistivity (Ohm.cm)
            :param nodeD: node diameter (m)
            :param nodeL: node length (m)
            :param interD: internode diameter (m)
            :param interL: internode length (m)
        '''
        # stinL = (interL - (nodeL + 2 * (mysaL + flutL))) / 6

        self._g = fit_g(self.fiberD)
        self._axonD = fit_axonD(self.fiberD)
        self._nodeD = fit_nodeD(self.fiberD)
        self._paraD1 = fit_paraD1(self.fiberD)
        self._paraD2 = fit_paraD2(self.fiberD)
        self._deltax = fit_deltax(self.fiberD)
        self._paraLength2 = fit_paraLength2(self.fiberD)
        self._nl = fit_nl(self.fiberD)

        load_mechanisms(getNmodlDir())

        self._init_parameters(diameter)
        self._create_sections()
        self._build_topology()
        self._define_biophysics()

    @property
    def nnodes(self):
        return self._nnodes

    @nnodes.setter
    def nnodes(self, value):
        if value % 2 == 0:
            raise ValueError('number of nodes must be odd')
        self._nnodes = value

    @property
    def ninters(self):
        return self.nnodes - 1

    @property
    def nMYSA(self):
        return 2 * self.ninters

    @property
    def nFLUT(self):
        return 2 * self.ninters

    @property
    def nSTIN(self):
        return 6 * self.ninters

    @property
    def ntot(self):
        return self.nnodes + self.nMYSA + self.nFLUT + self.nSTIN

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


    def _init_parameters(self, diameter):
        """ Initialize all cell parameters. """

        # morphological parameters
        self.fiberD = diameter
        self._paraLength1 = 3
        self._nodeLength = 1.0
        self._spaceP1 = 0.002
        self._spaceP2 = 0.004
        self._spaceI = 0.004

        # electrical parameters
        self._rhoa = 0.7e6  # Ohm-um
        self._mycm = 0.1  # uF/cm2/lamella membrane
        self._mygm = 0.001  # S/cm2/lamella membrane

        # fit the parameters with polynomials to allow any diameter
        interpolationDegree = 3
        experimentalDiameters = [5.7, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]
        experimentalG = [0.605, 0.661, 0.690, 0.700, 0.719, 0.739, 0.767, 0.791]
        experimentalAxonD = [3.4, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]
        experimentalNodeD = [1.9, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]
        experimentalParaD1 = [1.9, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]
        experimentalParaD2 = [3.4, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]
        experimentalDeltaX = [500, 1000, 1150, 1250, 1350, 1400, 1450, 1500]
        experimentalParaLength2 = [35, 40, 46, 50, 54, 56, 58, 60]
        experimentalNl = [80, 110, 120, 130, 135, 140, 145, 150]
        fit_g = np.poly1d(np.polyfit(experimentalDiameters, experimentalG, interpolationDegree))
        fit_axonD = np.poly1d(np.polyfit(experimentalDiameters,
                                         experimentalAxonD, interpolationDegree))
        fit_nodeD = np.poly1d(np.polyfit(experimentalDiameters,
                                         experimentalNodeD, interpolationDegree))
        fit_paraD1 = np.poly1d(np.polyfit(experimentalDiameters,
                                          experimentalParaD1, interpolationDegree))
        fit_paraD2 = np.poly1d(np.polyfit(experimentalDiameters,
                                          experimentalParaD2, interpolationDegree))
        fit_deltax = np.poly1d(np.polyfit(experimentalDiameters,
                                          experimentalDeltaX, interpolationDegree))
        fit_paraLength2 = np.poly1d(np.polyfit(experimentalDiameters,
                                               experimentalParaLength2, interpolationDegree))
        fit_nl = np.poly1d(np.polyfit(experimentalDiameters, experimentalNl, interpolationDegree))

        # interpolate
        self._g = fit_g(self.fiberD)
        self._axonD = fit_axonD(self.fiberD)
        self._nodeD = fit_nodeD(self.fiberD)
        self._paraD1 = fit_paraD1(self.fiberD)
        self._paraD2 = fit_paraD2(self.fiberD)
        self._deltax = fit_deltax(self.fiberD)
        self._paraLength2 = fit_paraLength2(self.fiberD)
        self._nl = fit_nl(self.fiberD)

        self._Rpn0 = (self._rhoa * .01) / \
            (np.pi * ((((self._nodeD / 2) + self._spaceP1)**2) - ((self._nodeD / 2)**2)))
        self._Rpn1 = (self._rhoa * .01) / \
            (np.pi * ((((self._paraD1 / 2) + self._spaceP1)**2) - ((self._paraD1 / 2)**2)))
        self._Rpn2 = (self._rhoa * .01) / \
            (np.pi * ((((self._paraD2 / 2) + self._spaceP2)**2) - ((self._paraD2 / 2)**2)))
        self._Rpx = (self._rhoa * .01) / \
            (np.pi * ((((self._axonD / 2) + self._spaceI)**2) - ((self._axonD / 2)**2)))
        self._interLength = (self._deltax - self._nodeLength -
                             (2 * self._paraLength1) - (2 * self._paraLength2)) / 6


        self.nodeToNodeDistance = self._nodeLength + 2 * \
            self._paraLength1 + 2 * self._paraLength2 + 6 * self._interLength
        self.totalFiberLength = self._nodeLength * self._axonNodes + self._paraNodes1 * \
            self._paraLength1 + self._paraNodes2 * self._paraLength2 + self._interLength * self._axonInter

    def _create_sections(self):
        """ Create the sections of the cell. """

        # NOTE: cell=self is required to tell NEURON of this object.
        self.node = [h.Section(name='node', cell=self) for x in range(self._axonNodes)]
        self.mysa = [h.Section(name='mysa', cell=self) for x in range(self._paraNodes1)]
        self.flut = [h.Section(name='flut', cell=self) for x in range(self._paraNodes2)]
        self.stin = [h.Section(name='stin', cell=self) for x in range(self._axonInter)]
        self.segments = []
        for i, node in enumerate(self.node):
            self.segments.append([node, self._nodeLength / 2 + i * (self._nodeLength + 2 *
                                                                    self._paraLength1 + 2 * self._paraLength2 + 6 * self._interLength), 'node'])
        for i, mysa in enumerate(self.mysa):
            if i % 2 == 0:
                self.segments.append([mysa, self._nodeLength + self._paraLength1 / 2 + (round(i / 2)) * (
                    self._nodeLength + 2 * self._paraLength1 + 2 * self._paraLength2 + 6 * self._interLength), 'paranode'])
            else:
                self.segments.append([mysa, self._nodeLength + self._paraLength1 + 2 * self._paraLength2 + 6 * self._interLength + self._paraLength1 / 2 + (
                    round(i / 2)) * (self._nodeLength + 2 * self._paraLength1 + 2 * self._paraLength2 + 6 * self._interLength), 'paranode'])
        for i, flut in enumerate(self.flut):
            if i % 2 == 0:
                self.segments.append([flut, self._nodeLength + self._paraLength2 + self._paraLength2 / 2 + (round(i / 2)) * (
                    self._nodeLength + 2 * self._paraLength1 + 2 * self._paraLength2 + 6 * self._interLength), 'paranode'])
            else:
                self.segments.append([flut, self._nodeLength + self._paraLength1 + self._paraLength2 + 6 * self._interLength + self._paraLength2 / 2 + (
                    round(i / 2)) * (self._nodeLength + 2 * self._paraLength1 + 2 * self._paraLength2 + 6 * self._interLength), 'paranode'])
        for i, stin in enumerate(self.stin):
            self.segments.append([stin, self._nodeLength + self._paraLength1 + self._paraLength2 + i % 6 * self._interLength + self._interLength / 2 + (
                round(i / 6)) * (self._nodeLength + 2 * self._paraLength1 + 2 * self._paraLength2 + self._axonInter * self._interLength), 'paranode'])

    def _define_biophysics(self):
        """ Assign the membrane properties across the cell. """
        for node in self.node:
            node.nseg = 1
            node.diam = self._nodeD
            node.L = self._nodeLength
            node.Ra = self._rhoa / 10000
            node.cm = 2
            node.insert('axnode')
            node.insert('extracellular')
            node.xraxial[0] = self._Rpn0
            node.xg[0] = 1e10
            node.xc[0] = 0

        for mysa in self.mysa:
            mysa.nseg = 1
            mysa.diam = self.fiberD
            mysa.L = self._paraLength1
            mysa.Ra = self._rhoa * (1 / (self._paraD1 / self.fiberD)**2) / 10000
            mysa.cm = 2 * self._paraD1 / self.fiberD
            mysa.insert('pas')
            mysa.g_pas = 0.001 * self._paraD1 / self.fiberD
            mysa.e_pas = -80
            mysa.insert('extracellular')
            mysa.xraxial[0] = self._Rpn1
            mysa.xg[0] = self._mygm / (self._nl * 2)
            mysa.xc[0] = self._mycm / (self._nl * 2)

        for flut in self.flut:
            flut.nseg = 1
            flut.diam = self.fiberD
            flut.L = self._paraLength2
            flut.Ra = self._rhoa * (1 / (self._paraD2 / self.fiberD)**2) / 10000
            flut.cm = 2 * self._paraD2 / self.fiberD
            flut.insert('pas')
            flut.g_pas = 0.0001 * self._paraD2 / self.fiberD
            flut.e_pas = -80
            flut.insert('extracellular')
            flut.xraxial[0] = self._Rpn2
            flut.xg[0] = self._mygm / (self._nl * 2)
            flut.xc[0] = self._mycm / (self._nl * 2)

        for stin in self.stin:
            stin.nseg = 1
            stin.diam = self.fiberD
            stin.L = self._interLength
            stin.Ra = self._rhoa * (1 / (self._axonD / self.fiberD)**2) / 10000
            stin.cm = 2 * self._axonD / self.fiberD
            stin.insert('pas')
            stin.g_pas = 0.0001 * self._axonD / self.fiberD
            stin.e_pas = -80
            stin.insert('extracellular')
            stin.xraxial[0] = self._Rpx
            stin.xg[0] = self._mygm / (self._nl * 2)
            stin.xc[0] = self._mycm / (self._nl * 2)

    def _build_topology(self):
        """ connect the sections together """
        # childSection.connect(parentSection, [parentX], [childEnd])
        for i in range(self._axonNodes - 1):
            self.node[i].connect(self.mysa[2 * i], 0, 1)
            self.mysa[2 * i].connect(self.flut[2 * i], 0, 1)
            self.flut[2 * i].connect(self.stin[6 * i], 0, 1)
            self.stin[6 * i].connect(self.stin[6 * i + 1], 0, 1)
            self.stin[6 * i + 1].connect(self.stin[6 * i + 2], 0, 1)
            self.stin[6 * i + 2].connect(self.stin[6 * i + 3], 0, 1)
            self.stin[6 * i + 3].connect(self.stin[6 * i + 4], 0, 1)
            self.stin[6 * i + 4].connect(self.stin[6 * i + 5], 0, 1)
            self.stin[6 * i + 5].connect(self.flut[2 * i + 1], 0, 1)
            self.flut[2 * i + 1].connect(self.mysa[2 * i + 1], 0, 1)
            self.mysa[2 * i + 1].connect(self.node[i + 1], 0, 1)

    """
    Redefinition of inherited methods
    """

    def connect_to_target(self, target, nodeN=-1):
        nc = h.NetCon(self.node[nodeN](1)._ref_v, target, sec=self.node[nodeN])
        nc.threshold = -30
        nc.delay = 0
        return nc

    def is_artificial(self):
        """ Return a flag to check whether the cell is an integrate-and-fire or artificial cell. """
        return 0
