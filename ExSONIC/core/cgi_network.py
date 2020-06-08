# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-06-07 14:42:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-06-08 18:44:39

import numpy as np
from neuron import h, hclass

from PySONIC.utils import logger

from .pyhoc import Matrix
from ..utils import seriesGeq
from ..constants import *


class SquareMatrix(Matrix):
    ''' Interface to a square matrix object. '''

    def __new__(cls, n):
        ''' Instanciation. '''
        return super(SquareMatrix, cls).__new__(cls, n, n)

    def emptyClone(self):
        ''' Return empty matrix of identical shape. '''
        return SquareMatrix(self.nrow())

    def addLink(self, i, j, w):
        ''' Add a bi-directional link between two nodes with a specific weight.

            :param i: first node index
            :param j: second node index
            :param w: link weight
        '''
        self.addVal(i, i, w)
        self.addVal(i, j, -w)
        self.addVal(j, j, w)
        self.addVal(j, i, -w)


class DiagonalMatrix(SquareMatrix):
    ''' Interface to a diagonal matrix. '''

    def __new__(cls, x):
        ''' Instanciation. '''
        return super(DiagonalMatrix, cls).__new__(cls, x.size)

    def __init__(self, x):
        ''' Initialization.

            :param x: vector used to fill the diagonal.
        '''
        self.setdiag(0, h.Vector(x))


class ConductanceMatrix(SquareMatrix):
    ''' Interface to an axial conductance matrix. '''

    def __new__(cls, Gvec, **_):
        ''' Instanciation. '''
        return super(ConductanceMatrix, cls).__new__(cls, Gvec.size)

    def __init__(self, Gvec, links=None):
        ''' Initialization.

            :param Gvec: vector of reference conductances for each element (S)
            :param links: list of paired indexes inicating links across nodes.
        '''
        self.Gvec = Gvec
        if links is not None:
            self.addLinks(links)

    @property
    def Gvec(self):
        return self._Gvec

    @Gvec.setter
    def Gvec(self, value):
        assert value.size == self.nrow(), 'conductance vector does not match number of rows'
        self._Gvec = value

    def Gij(self, i, j):
        ''' Half conductance in series. '''
        return 2 * seriesGeq(self.Gvec[i], self.Gvec[j])

    def addLink(self, i, j):
        ''' Add a link between two nodes.

            :param i: first node index
            :param j: second node index
        '''
        super().addLink(i, j, self.Gij(i, j))

    def removeLink(self, i, j):
        ''' Remove a link between two nodes.

            :param i: first node index
            :param j: second node index
        '''
        super().addLink(i, j, -self.Gij(i, j))

    def addLinks(self, links):
        ''' Add cross-nodes links to the matrix.

            :param links: list of paired indexes inicating links across nodes.
        '''
        for i, j in links:
            self.addLink(i, j)
        self.checkNullRows()

    def checkNullRows(self):
        ''' Check that all rows sum up to zero (or close). '''
        for i in range(self.nrow()):
            rsum = self.getrow(i).sum()
            assert np.isclose(rsum, .0, atol=1e-12), f'non-zero sum on line {i}: {rsum}'


class NormalizedConductanceMatrix(ConductanceMatrix):
    ''' Interface to an normalized axial conductance matrix. '''

    def __new__(cls, Gvec, *args, **kwargs):
        ''' Instanciation. '''
        return super(NormalizedConductanceMatrix, cls).__new__(cls, Gvec)

    def __init__(self, Gvec, xnorm=None, **kwargs):
        ''' Initialization.

            :param xnorm: vector specifying the normalization factor for each row
        '''
        self.xnorm = np.ones(Gvec.size) if xnorm is None else xnorm
        super().__init__(Gvec, **kwargs)

    def setNorm(self, value):
        ''' Set a new row-normalization vector. '''
        assert value.size == self.nrow(), 'normalizing vector does not match number of rows'
        for i in range(self.nrow()):
            self.setrow(i, self.getrow(i) * self.xnorm[i] / value[i])
        self.xnorm = value

    def setVal(self, i, j, x):
        ''' Set matrix element divided by the corresponding row normalizer. '''
        super().setVal(i, j, x / self.xnorm[i])

    def getVal(self, i, j):
        ''' Get matrix element multiplied by the corresponding row normalizer. '''
        return super().getVal(i, j) * self.xnorm[i]

    def mulByRow(self, x):
        ''' Return new matrix with rows multiplied by vector values. '''
        assert x.size == self.nrow(), f'Input vector must be of size {self.nrow()}'
        mout = self.emptyClone()
        for i in range(self.nrow()):
            mout.setrow(i, self.getrow(i) * x[i])
        return mout

    def scaled(self):
        ''' Return a "scaled" version of the matrix where each row is multiplied
            by its corresponding normalizer.
        '''
        return self.mulByRow(self.xnorm)


class PointerVector(hclass(h.Vector)):
    ''' Interface to a pointing vector, i.e. a vector that can "point" toward
        subsets of other vectors.
    '''

    def __init__(self, *args, **kwargs):
        ''' Initialization. '''
        self.refs = []
        super().__init__(*args, **kwargs)

    # def __setitem__(self, i, x):
    #     ''' Item setter, adding the difference between the old an new value to all
    #         reference vectors with their respective offsets.
    #     '''
    #     xdiff = x - self.get(i)
    #     print(i, x, xdiff)
    #     self.set(i, x)
    #     for v, offset in self.refs:
    #         v.set(i + offset, v.get(i + offset) + xdiff)

    def setVal(self, i, x):
        ''' Item setter, adding the difference between the old an new value to all
            reference vectors with their respective offsets.
        '''
        xdiff = x - self.get(i)
        # print(i, x, xdiff)
        self.set(i, x)
        for v, offset in self.refs:
            v.set(i + offset, v.get(i + offset) + xdiff)

    def addVal(self, i, x):
        self.setVal(i, self.get(i) + x)

    def addTo(self, v, offset, fac=1):
        ''' Add the vector to a destination vector with a specific offset and factor. '''
        for i, x in enumerate(self):
            v.set(i, v.get(i) + x * fac)

    def addRef(self, v, offset):
        ''' Add a reference vector to "point" towards. '''
        assert self.size() + offset <= v.size(), 'exceeds reference dimensions'
        self.addTo(v, offset)
        self.refs.append((v, offset))

    def removeRef(self, iref):
        ''' Remove a reference vector from the list. '''
        self.addTo(*self.refs[iref], fac=-1)
        del self.refs[iref]


def pointerMatrix(MatrixBase):
    ''' Interface to a pointing matrix, i.e. a matrix that can "point" toward
        subsets of other matrices.
    '''

    class PointerMatrix(MatrixBase):

        def __init__(self, *args, **kwargs):
            ''' Initialization. '''
            self.refs = []
            super().__init__(*args, **kwargs)

        def addTo(self, mout, i, j, fac=1):
            ''' Add the current matrix to a destination matrix, starting at a specific
                row and column index.
            '''
            for k in range(self.nrow()):
                for l in range(self.ncol()):
                    mout.addVal(k + i, l + j, self.getval(k, l) * fac)

        def addRef(self, m, row_offset, col_offset):
            ''' Add a reference matrix to "point" towards. '''
            assert self.nrow() + row_offset <= m.nrow(), 'exceeds reference dimensions'
            assert self.ncol() + col_offset <= m.ncol(), 'exceeds reference dimensions'
            self.addTo(m, row_offset, col_offset)
            self.refs.append((m, row_offset, col_offset))

        def removeRef(self, iref):
            self.addTo(*self.refs[iref], fac=-1)
            del self.refs[iref]

        def setVal(self, i, j, x):
            xold = self.getval(i, j)
            super().setVal(i, j, x)
            xdiff = self.getval(i, j) - xold
            for m, row_offset, col_offset in self.refs:
                m.addVal(i + row_offset, j + col_offset, xdiff)

    PointerMatrix.__name__ = f'Pointer{MatrixBase.__name__}'

    return PointerMatrix


class HybridNetwork:
    ''' Interface used to build a hybrid voltage network amongst a list of sections.

        Consider a neuron model consisting of a list of sections connected in series.

        We define the following terms:

        - vi: intracelular voltage
        - vm: transmembrane voltage
        - vx: extracellular voltage
        - ex: imposed voltage outside of the surrounding extracellular membrane
        - is: stimulating current
        - cm: membrane capacitance
        - i(vm): transmembrane ionic current
        - ga: intracellular axial conductance between nodes
        - cx: capacitance of surrounding extracellular membrane (e.g. myelin)
        - gx: transverse conductance of surrounding extracellular membrane (e.g. myelin)
        - gp: extracellular axial conductance between nodes (e.g. periaxonal space)
        - j: index indicating the connection to a neighboring node

        Governing equations for internal and external nodes are, respectively:

        (1) cm * dvm/dt + i(vm) = is + ga_j * (vi_j - vi)
        (2) cx * dvx/dt + gx * (vx - ex) = cm * dvm/dt + i(vm) + gp_j * (vx_j - vx)

        Putting all voltage dependencies on the left-hand sides, and developing, we find:

        (1) cm * dvm/dt + ga_j * (vi - vi_j) = is - i(vm)
        (2) cx * dvx/dt - cm * dvm/dt + gx * vx + gp_j * (vx - vx_j) = i(vm) + gx * ex

        Re-expressing vi as (vm + vx), we find the matrix equation rows for the two nodes:

        (1) cm * dvm/dt + ga_j * (vm - vm_j) + ga_j * (vx - vx_j) = is - i(vm)
        (2) cx * dvx/dt - cm * dvm/dt + gx * vx + gp_j * (vx - vx_j) = i(vm) + gx * ex

        Special attention must be brought on the fact that we use NEURONS's v variable as an
        alias for the section's membrane charge density Qm. Therefore, the system must be
        adapted in consequence. To this effect, we introduce a change of variable:

        Qm = cm * vm; dQm/dt = cm * dvm/dt  (cm considered time-invariant)

        Recasting the system according to this change of variable thus yields:

        (1) dQm/dt + ga_j * (Qm/cm - Qm_j/cm_j) + ga_j * (vx - vx_j) = is - i(vm)
        (2) cx * dvx/dt - dQm/dt + gx * vx + gp_j * (vx - vx_j) = i(vm) + gx * ex

        Finally, we replace dQm/dt + i(vm) in (2) by equivalents intracellular axial and
        stimulation currents, in order to remove the need to access net membrane current.
        After re-arranging all linear voltage terms on the left-hand side, we have:

        (1) dQm/dt + ga_j * (Qm/cm - Qm_j/cm_j) + ga_j * (vx - vx_j) = is - i(vm)
        (2) cx * dvx/dt + ga_j * (Qm/cm - Qm_j/cm_j) + (ga_j + gp_j) * (vx - vx_j) + gx * vx
            = is + gx * ex

        Developing to isolate Qm elements, we have:

        (1) dQm/dt + ga_j/cm * Qm - ga_j/cm_j * Qm_j + ga_j * (vx - vx_j) = is - i(vm)
        (2) cx * dvx/dt + ga_j/cm * Qm - ga_j/cm_j * Qm_j + (ga_j + gp_j) * (vx - vx_j) + gx * vx
            = is + gx * ex

        Among these equation rows, 2 terms are automatically handled by NEURON, namely:
        - LHS-1: dQm/dt
        - RHS-1: is - i(vm)

        Hence, 8 terms remain to be handled as part of the additional network:
        - LHS-1: ga_j/cm * Qm - ga_j/cm_j * Qm_j
        - LHS-1: ga_j * vx - ga_j * vx_j
        - LHS-2: cx * dvx/dt
        - LHS-2: ga_j/cm * Qm - ga_j/cm_j * Qm_j
        - LHS-2: ga_j * vx - ga_j * vx_j
        - LHS-2: gp_j * (vx - vx_j)
        - LHS-2: gx * vx
        - RHS-2: is + gx * ex

        Note that axial conductance terms (ga and gp) must be normalized by the
        appropriate section membrane area in order to obtain a consistent system.

        How do we do this?

        The above equations describe a partial differential system of the form:

            C * dy/dt + G * y = I

        with C a capacitance matrix, G a conductance matrix, y a hybrid vector
        of transmembrane charge density and external voltage, and I a current vector.

        Thankfully, NEURON provides a so-called "LinearMechanism" class that allows
        to define such a system, where the first n elements of the y vector
        can be mapped to the "v" variable of a specific list of sections. This allows
        us to add a linear network atop of NEURON's native network, in which we define
        the additional terms.

        Concretely, a model of n sections connected in series can be represented by a "CGI"
        system of size 2*n, where the first n items correspond to membrane charge density
        nodes, and the following n items correspond to external voltage nodes.

        The corresponding linear mechanism added atop of NEURON's native linear network
        would then consist of the following capacitance and conductance matrices C and G,
        and current vector I:

                -------------------------------
                |              |              |
                |       0      |       0      |
                |              |              |
        C  =    -------------------------------
                |              |              |
                |        0     |      Cx      |
                |              |              |
                -------------------------------

                -------------------------------
                |              |              |
                |     Ga/cm    |      Ga      |
                |              |              |
        G =     -------------------------------
                |              |              |
                |     Ga/cm    | Ga + Gp + Gx |
                |              |              |
                -------------------------------

                -------------------------------
        I =     |        0     |  gx*ex + is  |
                -------------------------------

        Where internal terms are defined as:
        - Ga: n-by-n matrix of intracellular axial conductance
        - Ga/cm: Ga matrix normalized by capacitance at each time step
        - Gp: n-by-n matrix of extracellular axial conductance
        - Gx: n-by-n sparse matrix of transverse extracellular conductance
        - Cx: n-by-n sparse matrix of transverse extracellular capacitance

        Note also that the gx conductance is connected in series with another transverse
        conductance of value 1e9, to mimick NEURON's default 2nd extracellular layer.
    '''

    def __init__(self, seclist, connections, has_ext_layer, is_dynamic_cm=False):
        ''' Initialization.

            :param seclist: list of sections.
            :param connections: list of index pairs (tuples) indicating connections across sections
            :param has_ext_layer: boolean indicating whether to implement an extracellular layer
            :param is_dynamic_cm: boolean indicating whether membrane capacitance is time-varying
        '''
        # Assign attributes
        self.seclist = seclist
        self.connections = connections
        self.has_ext_layer = has_ext_layer
        self.is_dynamic_cm = is_dynamic_cm
        self.setGlobalComponents()
        self.setBaseLayer()
        if self.has_ext_layer:
            self.setExtracellularLayer()
        self.startLM()

    def __repr__(self):
        cm = {False: 'static', True: 'dynamic'}[self.is_dynamic_cm]
        return f'{self.__class__.__name__}({self.nsec} sections, {self.nlayers} layers, {cm} cm)'

    @property
    def seclist(self):
        return self._seclist

    @seclist.setter
    def seclist(self, value):
        self._seclist = value

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, value):
        self._connections = value

    @property
    def has_ext_layer(self):
        return self._has_ext_layer

    @has_ext_layer.setter
    def has_ext_layer(self, value):
        self._has_ext_layer = value

    @property
    def nsec(self):
        ''' Number of sections in the network. '''
        return len(self.seclist)

    @property
    def nlayers(self):
        ''' Number of layers in the network. '''
        return 1 if not self.has_ext_layer else 2

    @property
    def size(self):
        ''' Overall size if the network (i.e. number of nodes). '''
        return self.nsec * self.nlayers

    def getVector(self, k):
        ''' Get a vector of values of a given parameter for each section in the list.

            :param k: parameter name
            :return: 1D numpy array with parameter values
        '''
        return np.array([getattr(sec, k) for sec in self.seclist])

    def setGlobalComponents(self):
        ''' Set the network's global components used by NEURON's linear Mechanism. '''
        self.C = SquareMatrix(self.size)  # capacitance matrix (mF/cm2)
        self.G = SquareMatrix(self.size)  # conductance matrix (S/cm2)
        self.y = h.Vector(self.size)      # charge density / extracellular voltage vector (mV)
        self.I = h.Vector(self.size)      # current vector (mA/cm2)

    def setBaseLayer(self):
        ''' Set components used to define the network's base layer. '''
        # Get required vectors
        self.Am = self.getVector('Am')   # membrane area (cm2)
        self.cm = self.getVector('Cm0')  # resting membrane capacitance (uF/cm2)
        self.ga = self.getVector('Ga')   # intracellular axial conductance (S)
        # Define Gacm matrix and point it towards G top-left
        self.Gacm = pointerMatrix(NormalizedConductanceMatrix)(
            self.ga, links=self.connections, xnorm=(self.cm * self.Am))
        self.Gacm.addRef(self.G, 0, 0)

    def setExtracellularLayer(self):
        ''' Set components used to define the network's extracellular layer. '''
        # Get required vectors
        self.cx = self.getVector('cx')  # uF/cm2
        self.gx = self.getVector('gx')  # S/cm2
        self.gp = self.getVector('Gp')  # S

        # Define additional matrices and point them towards G
        self.Cx = pointerMatrix(DiagonalMatrix)(self.cx * UF_CM2_TO_MF_CM2)
        self.Ga = pointerMatrix(NormalizedConductanceMatrix)(
            self.ga, links=self.connections, xnorm=self.Am)
        self.Gp = pointerMatrix(NormalizedConductanceMatrix)(
            self.gp, links=self.connections, xnorm=self.Am)
        self.Gx = pointerMatrix(DiagonalMatrix)(self.gx)

        # Add references to global matrices
        self.Cx.addRef(self.C, self.nsec, self.nsec)  # Bottom-right: Cx * dvx/dt
        self.Gacm.addRef(self.G, self.nsec, 0)        # Bottom-left: Ga/cm * Qm
        self.Ga.addRef(self.G, 0, self.nsec)          # Top-right: Ga * vx
        self.Gp.addRef(self.G, self.nsec, self.nsec)  # Bottom-right: Gp * vx
        self.Ga.addRef(self.G, self.nsec, self.nsec)  # Bottom-right: Ga * vx
        self.Gx.addRef(self.G, self.nsec, self.nsec)  # Bottom-right: Gx * vx

        # Define pointing vectors toward extracellular voltage and currents
        self.istim = PointerVector(self.nsec)
        self.istim.addRef(self.I, self.nsec)
        self.iex = PointerVector(self.nsec)
        self.iex.addRef(self.I, self.nsec)
        self.vx = PointerVector(self.nsec)
        self.vx.addRef(self.y, self.nsec)

    def index(self, sec):
        return self.seclist.index(sec)

    def getVextRef(self, sec):
        ''' Get reference to a section's extracellular voltage variable. '''
        if not self.has_ext_layer:
            raise ValueError('Network does not have an extracellular layer')
        return self.y._ref_x[self.index(sec) + self.nsec]

    @property
    def relx(self):
        return h.Vector([0.5] * self.nsec)

    @property
    def sl(self):
        return h.SectionList(self.seclist)

    def startLM(self):
        ''' Feed network into a LinearMechanism object. '''
        # Set initial conditions vector
        self.y0 = h.Vector(self.size)
        # Define linear mechanism arguments
        lm_args = [self.C, self.G, self.y, self.y0, self.I, self.sl, self.relx]
        # Add update callback only if dynamic cm or extracellular layer
        if self.is_dynamic_cm:
            lm_args = [self.updateCmTerms] + lm_args
        # Create LinearMechanism object
        self.lm = h.LinearMechanism(*lm_args)

    def clear(self):
        ''' Delete the network's LinearMechanism object. '''
        del self.lm

    def updateCmTerms(self):
        ''' Update capacitance-dependent network components. '''
        # Update membrane capacitance vector
        for i, sec in enumerate(self.seclist):
            self.cm[i] = sec.getCm(x=0.5)
        # Modify Gacm matrix accordingly
        self.Gacm.setNorm(self.cm * self.Am)

    def setIstim(self, sec, I):
        ''' Set the stimulation current of a given section to a new value. '''
        self.istim.setVal(self.index(sec), I / MA_TO_NA / sec.Am)

    def setEx(self, sec, old_ex, new_ex):
        ''' Set the imposed extracellular potential of a given section to a new value
            (propagating the discontinuity into the vx vector).
        '''
        i = self.index(sec)
        self.vx.addVal(i, new_ex - old_ex)
        self.iex.setVal(i, self.gx[i] * new_ex)

    def log(self):
        ''' Print network components. '''
        if self.size > 40:
            logger.warning('exceeding max number of sections to print network correctly')
            return
        fmt = '%-8g' if self.size <= 22 else '%-4g'
        with np.printoptions(**array_print_options):
            logger.info(f'cm: {self.cm} uF/cm2')
            if self.has_ext_layer:
                logger.info(f'cx: {self.cx} uF/cm2')
            logger.info('C (mF/cm2):')
            self.C.printf(fmt)
            logger.info(f'ga: {self.ga}')
            logger.info('Ga:')
            self.Ga.printf()
            if self.has_ext_layer:
                logger.info(f'gp: {self.gp}')
                logger.info('Gp:')
                self.Gp.printf()
                logger.info(f'gx: {self.gx}')
            logger.info(f'G:')
            self.G.printf(fmt)
