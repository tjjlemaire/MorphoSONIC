# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-19 14:42:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-20 15:16:52

from .pyhoc import *

class NeuronModel(metaclass=abc.ABCMeta):

    @property
    def modfile(self):
        return f'{self.pneuron.name}.mod'

    @property
    def mechname(self):
        return f'{self.pneuron.name}auto'

    def createSection(self, id):
        ''' Create a model section with a given id. '''
        if hasattr(self, 'connection_scheme'):
            return IaxSection(self.connection_scheme,
                self.mechname, self.pneuron.statesNames(), name=id, cell=self)
        else:
            return Section(
                self.mechname, self.pneuron.statesNames(), name=id, cell=self)

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        h.finitialize(self.pneuron.Qm0 * 1e5)  # nC/cm2

    def setTimeProbe(self):
        ''' Set time probe. '''
        return Probe(h._ref_t)

    def integrate(self, pp, dt, atol):
        ''' Integrate a model differential variables for a given duration, while updating the
            value of the boolean parameter stimon during ON and OFF periods throughout the numerical
            integration, according to stimulus parameters.

            Integration uses an adaptive time step method by default.

            :param pp: pulsed protocol object
            :param dt: integration time step (s). If provided, the fixed time step method is used.
            :param atol: absolute error tolerance (default = 1e-3). If provided, the adaptive
                time step method is used.
        '''
        tstim, toffset, PRF, DC = pp.tstim, pp.toffset, pp.PRF, pp.DC
        tstop = tstim + toffset

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
        self.initToSteadyState()
        self.stimon = self.setStimON(1)
        self.cvode.event(self.Ton, self.toggleStim)

        # Integrate
        while h.t < tstop:
            h.fadvance()

        # Set absolute error tolerance back to default value if changed
        if atol is not None:
            self.cvode.atol(def_atol)

        return 0

    def toggleStim(self):
        ''' Toggle stimulus state (ON -> OFF or OFF -> ON) and set appropriate next toggle event. '''
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

    def setModLookup(self, *args, **kwargs):
        ''' Get the appropriate model 2D lookup and translate it to Hoc. '''
        # Set Lookup
        self.setPyLookup(*args, **kwargs)

        # Convert lookups independent variables to hoc vectors
        self.Aref = h.Vector(self.pylkp.refs['A'] * 1e-3)  # kPa
        self.Qref = h.Vector(self.pylkp.refs['Q'] * 1e5)   # nC/cm2

        # Convert lookup tables to hoc matrices
        # !!! hoc lookup dictionary must be a member of the class,
        # otherwise the assignment below does not work properly !!!
        self.lkp = {'V': Matrix(self.pylkp['V'])}  # mV
        for ratex in self.pneuron.alphax_list.union(self.pneuron.betax_list):
            self.lkp[ratex] = Matrix(self.pylkp[ratex] * 1e-3)  # ms-1
        for taux in self.pneuron.taux_list:
            self.lkp[taux] = Matrix(self.pylkp[taux] * 1e3)  # ms
        for xinf in self.pneuron.xinf_list:
            self.lkp[xinf] = Matrix(self.pylkp[xinf])  # (-)

    def setFuncTables(self, *args, **kwargs):
        ''' Set neuron-specific interpolation tables along the charge dimension,
            and link them to FUNCTION_TABLEs in the MOD file of the corresponding
            membrane mechanism.
        '''
        logger.debug(f'loading {self.mechname} membrane dynamics lookup tables')

        # Set Lookup
        self.setModLookup(*args, **kwargs)

        # Assign hoc matrices to 2D interpolation tables in membrane mechanism
        for k, v in self.lkp.items():
            self.setFuncTable(self.mechname, k, v, self.Aref, self.Qref)

    @staticmethod
    def setFuncTable(mechname, fname, matrix, xref, yref):
        ''' Set the content of a 2-dimensional FUNCTION TABLE of a density mechanism.

            :param mechname: name of density mechanism
            :param fname: name of the FUNCTION_TABLE reference in the mechanism
            :param matrix: HOC Matrix object with values to be linearly interpolated
            :param xref: HOC Vector object with reference values for interpolation in the 1st dimension
            :param yref: HOC Vector object with reference values for interpolation in the 2nd dimension
            :return: the updated HOC object
        '''
        # Check conformity of inputs
        dims_not_matching = 'reference vector size ({}) does not match matrix {} dimension ({})'
        nx, ny = matrix.nrow(), matrix.ncol()
        assert xref.size() == nx, dims_not_matching.format(xref.size(), '1st', nx)
        assert yref.size() == ny, dims_not_matching.format(yref.size(), '2nd', nx)

        # Get the HOC function that fills in a specific FUNCTION_TABLE in a mechanism
        fillTable = getattr(h, f'table_{fname}_{mechname}')

        # Call function and return
        return fillTable(matrix._ref_x[0][0], nx, xref._ref_x[0], ny, yref._ref_x[0])