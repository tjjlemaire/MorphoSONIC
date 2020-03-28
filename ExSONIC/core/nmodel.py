# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-19 14:42:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-03-27 23:37:37

import abc
from neuron import h
import numpy as np
import pandas as pd
from scipy import stats

from PySONIC.core import Model, PointNeuron
from PySONIC.postpro import detectSpikes, prependDataFrame
from PySONIC.utils import logger, si_format, plural, filecode, simAndSave, isIterable
from PySONIC.constants import *

from .pyhoc import *


class NeuronModel(metaclass=abc.ABCMeta):

    int_methods = {
        0: 'backward Euler method',
        1: 'Crank-Nicholson method',
        2: 'Crank-Nicholson method with fixed currents at mid-steps',
        3: 'CVODE multi order variable time step method',
        4: 'DASPK (Differential Algebraic Solver with Preconditioned Krylov) method'
    }

    def setCelsius(self, celsius=None):
        if celsius is None:
            try:
                celsius = self.pneuron.celsius
            except AttributeError:
                raise ValueError('celsius value not provided and not found in PointNeuron class')
        h.celsius = celsius

    @property
    def pneuron(self):
        return self._pneuron

    @pneuron.setter
    def pneuron(self, value):
        if not isinstance(value, PointNeuron):
            raise TypeError(f'{value} is not a valid PointNeuron instance')
        self._pneuron = value

    @property
    def modfile(self):
        return f'{self.pneuron.name}.mod'

    @property
    def mechname(self):
        return f'{self.pneuron.name}auto'

    @staticmethod
    def axialSectionArea(d_out, d_in=0.):
        ''' Compute the cross-section area of a axial cylinder section expanding between an
            inner diameter (presumably zero) and an outer diameter.

            :param d_out: outer diameter (m)
            :param d_in: inner diameter (m)
            :return: cross-sectional area (m2)
        '''
        return np.pi * ((d_out)**2 - d_in**2) / 4.

    @classmethod
    def axialResistancePerUnitLength(cls, rho, *args, **kwargs):
        ''' Compute the axial resistance per unit length of a cylindrical section.

            :param rho: axial resistivity (Ohm.cm)
            :return: resistance per unit length (Ohm/cm)
        '''
        return rho / cls.axialSectionArea(*args, **kwargs) * 1e-4  # Ohm/cm

    @classmethod
    def axialResistance(cls, rho, L, *args, **kwargs):
        ''' Compute the axial resistance of a cylindrical section.

            :param rho: axial resistivity (Ohm.cm)
            :param L: cylinder length (m)
            :return: resistance (Ohm)
        '''
        return cls.axialResistancePerUnitLength(rho, *args, **kwargs) * L * 1e2  # Ohm

    def createSection(self, id, mech=None, states=None, Cm0=None):
        ''' Create a model section with a given id. '''
        if Cm0 is None:
            Cm0 = self.pneuron.Cm0 * 1e2  # uF/cm2
        args = []
        if hasattr(self, 'connection_scheme'):
            section_class = CustomConnectMechQSection
            args.append(self.connection_scheme)
        else:
            section_class = MechQSection
        return section_class(
            *args, mechname=mech, states=states, name=id, cell=self, Cm0=Cm0)

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        h.finitialize(self.pneuron.Qm0 * 1e5)  # nC/cm2

    def setTimeProbe(self):
        ''' Set time probe. '''
        return Probe(h._ref_t)

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        self.section.setStimON(value)
        return value

    def getIntegrationMethod(self):
        ''' Get the method used by NEURON for the numerical integration of the system. '''
        method_type_code = self.cvode.current_method() % 1000 // 100
        method_type_str = self.int_methods[method_type_code]
        if self.cvode.active():
            return f'{method_type_str} (atol = {self.cvode.atol()})'
        else:
            return f'{method_type_str} (fixed dt = {h.dt} ms)'

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
        self.cvode = h.CVode()
        if dt is not None:
            h.secondorder = 0  # using backward Euler method if fixed time step
            h.dt = dt
            self.cvode.active(0)
        else:
            self.cvode.active(1)
            if atol is not None:
                def_atol = self.cvode.atol()
                self.cvode.atol(atol)

        # Initialize
        self.stimon = self.setStimON(0)
        self.initToSteadyState()
        self.stimon = self.setStimON(1)
        self.cvode.event(self.Ton, self.toggleStim)

        # Integrate
        logger.debug(f'integrating system using {self.getIntegrationMethod()}')
        while h.t < tstop:
            h.fadvance()

        # Set absolute error tolerance back to default value if changed
        if atol is not None:
            self.cvode.atol(def_atol)

        return 0

    def toggleStim(self):
        ''' Toggle stim state (ON -> OFF or OFF -> ON) and set appropriate next toggle event. '''
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

    @abc.abstractmethod
    def setPyLookup(self, *args, **kwargs):
        raise NotImplementedError

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
            :param xref: HOC Vector object with reference values for interpolation in 1st dimension
            :param yref: HOC Vector object with reference values for interpolation in 2nd dimension
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

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    def simulate(self, drive, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param drive: drive object
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
            :return: output dataframe
        '''
        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.section.setStimProbe()
        probes = self.section.setProbesDict()

        # Set drive and integrate model
        self.setDrive(drive)
        self.integrate(pp, dt, atol)

        # Store output in dataframe
        data = pd.DataFrame({
            't': t.to_array() * 1e-3,  # s
            'stimstate': stim.to_array()
        })
        for k, v in probes.items():
            data[k] = v.to_array()
        data.loc[:, 'Qm'] *= 1e-5  # C/m2

        # Prepend initial conditions (prior to stimulation)
        data = prependDataFrame(data)

        return data


class FiberNeuronModel(NeuronModel):

    tscale = 'ms'  # relevant temporal scale of the model
    mA_to_nA = 1e6  # conversion factor

    @property
    @abc.abstractmethod
    def simkey(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, value):
        if value <= 0:
            raise ValueError('longitudinal resistivity must be positive')
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
            raise ValueError('internode length must be positive or null')
        self._interL = value

    @property
    def nodeIDs(self):
        return [f'node{i}' for i in range(self.nnodes)]

    @property
    @abc.abstractmethod
    def refsection(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def length(self):
        raise NotImplementedError

    @property
    def sectypes(self):
        return list(self.sections.keys())

    def getXCoords(self):
        return {k: getattr(self, f'get{k.title()}Coords')() for k in self.sections.keys()}

    @property
    def z(self):
        return 0.

    def getXZCoords(self):
        return {k: np.vstack((v, np.ones(v.size) * self.z)).T
                for k, v in self.getXCoords().items()}  # m

    @property
    def drives(self):
        if not hasattr(self, '_drives'):
            self._drives = []
        return self._drives

    @drives.setter
    def drives(self, value):
        if not isIterable(value):
            raise ValueError('drives must be an iterable')
        for item in value:
            if not hasattr(item, 'toggle'):
                raise ValueError(f'drive {item} has no toggle method')
        self._drives = value

    def setStimON(self, value):
        for sec in self.seclist:
            sec.setStimON(value)
        for drive in self.drives:
            drive.toggle(value)
        return value

    def setOtherProbes(self):
        return {}

    def getSectionsDetails(self):
        ''' Get details about the model's sections. '''
        d = {}
        for secdict in self.sections.values():
            sec = secdict[list(secdict.keys())[0]]
            dd = sec.getDetails()
            if len(d) == 0:
                d = {'nsec': [], **{k: [] for k in dd.keys()}}
            d['nsec'].append(len(secdict))
            for k, v in dd.items():
                d[k].append(v)
        return pd.DataFrame(d, index=self.sectypes)

    def logSectionsDetails(self):
        return f'sections details:\n{self.getSectionsDetails().to_markdown()}'

    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    def simulate(self, source, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param source: source object
            :param pp: pulsed protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method (default = 1e-3)
            :return: output dataframe
        '''
        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.refsection.setStimProbe()
        node_probes = {k: v.setProbesDict() for k, v in self.nodes.items()}
        other_probes = self.setOtherProbes()

        # Set distributed drives
        self.setDrives(source)

        # Integrate model
        self.integrate(pp, dt, atol)

        # Store output in dataframes
        data = {}
        for id in self.nodes.keys():
            data[id] = pd.DataFrame({
                't': t.to_array() * 1e-3,  # s
                'stimstate': stim.to_array()
            })
            for k, v in node_probes[id].items():
                data[id][k] = v.to_array()
            data[id].loc[:, 'Qm'] *= 1e-5  # C/m2

        for sectype, secdict in other_probes.items():
            for k, v in secdict.items():
                data[k] = pd.DataFrame({
                    't': t.to_array() * 1e-3,  # s
                    'stimstate': stim.to_array(),
                    'Vm': v['Vm'].to_array()})

        # Prepend initial conditions (prior to stimulation)
        data = {id: prependDataFrame(df) for id, df in data.items()}

        return data

    def getSpikesTimings(self, data, zcross=True, spikematch='majority'):
        ''' Return an array containing occurence times of spikes detected on a collection of nodes.

            :param data: simulation output dataframe
            :param zcross: boolean stating whether to use ascending zero-crossings preceding peaks
                as temporal reference for spike occurence timings
            :return: dictionary of spike occurence times (s) per node.
        '''
        tspikes = {}
        nspikes = np.zeros(len(data.items()))
        errmsg = 'Ascending zero crossing #{} (t = {:.2f} ms) not prior to peak #{} (t = {:.2f} ms)'

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
                    i_zcross = np.where(np.diff(np.sign(Vm)) > 0)[0]  # ascending zero-crossings
                    # If mismatch, remove irrelevant zero-crossings by taking only the ones
                    # preceding each detected peak
                    if i_zcross.size > ispikes.size:
                        i_zcross = np.array([i_zcross[(i_zcross - i1) < 0].max() for i1 in ispikes])
                    # Compute slopes (mV/s)
                    slopes = (Vm[i_zcross + 1] - Vm[i_zcross]) / (t[i_zcross + 1] - t[i_zcross])
                    # Interpolate times of zero crossings
                    tzcross = t[i_zcross] - (Vm[i_zcross] / slopes)
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

    @abc.abstractmethod
    def isNormalDistance(self, d):
        raise NotImplementedError

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
            if self.isNormalDistance(d):
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

    def getSpikeAmp(self, data, ids=None, key='Vm', out='range'):
        # By default, consider all fiber nodes
        if ids is None:
            ids = self.nodeIDs.copy()
        amps = np.array([np.ptp(data[id][key].values) for id in ids])
        if out == 'range':
            return amps.min(), amps.max()
        elif out == 'median':
            return np.median(amps)
        elif out == 'mean':
            return np.mean(amps)
        else:
            raise AttributeError(f'invalid out option: {out}')

    def desc(self, meta):
        return f'{self}: simulation @ {meta["source"]}, {meta["pp"].desc}'

    @property
    def nodelist(self):
        return list(self.nodes.values())

    @property
    @abc.abstractmethod
    def seclist(self):
        raise NotImplementedError

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        nspikes_start = detectSpikes(data[self.nodeIDs[0]])[0].size
        nspikes_end = detectSpikes(data[self.nodeIDs[-1]])[0].size
        return nspikes_start > 0 and nspikes_end > 0

    def filecodes(self, source, pp, _):
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

    def str_biophysics(self):
        return f'{self.pneuron.name} neuron'

    def str_nodes(self):
        return f'{self.nnodes} node{plural(self.nnodes)}'

    def str_resistivity(self):
        return f'rs = {si_format(self.rs)}Ohm.cm'

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return '{}({})'.format(self.__class__.__name__, ', '.join([
            self.str_biophysics(),
            self.str_nodes(),
            self.str_resistivity(),
            self.str_geometry()
        ]))

    @property
    def corecodes(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name
        }

    @property
    def quickcode(self):
        return '_'.join([
            *self.corecodes.values(),
            f'fiberD{self.fiberD * 1e6:.2f}um'
        ])

    def getPltVars(self, *args, **kwargs):
        return self.pneuron.getPltVars(*args, **kwargs)

    @property
    def pltScheme(self):
        return self.pneuron.pltScheme

    @abc.abstractmethod
    def setDrives(self, source):
        ''' Set distributed stimulus amplitudes. '''
        raise NotImplementedError

    @abc.abstractmethod
    def titrate(self, source, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param source: source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
        raise NotImplementedError

    def getArange(self, source):
        return [source.computeSourceAmp(self, x) for x in self.A_range]

    def getAstart(self, source):
        return source.computeSourceAmp(self, self.A_start)

    def titrationFunc(self, *args):
        data, _ = self.simulate(*args)
        return self.isExcited(data)

    def reset(self):
        ''' delete and re-construct all model sections. '''
        self.clear()
        self.construct()
