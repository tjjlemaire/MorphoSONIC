# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-19 14:42:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-30 17:14:49

import abc
from neuron import h
import numpy as np
import pandas as pd
from scipy import stats
from boltons.strutils import cardinalize

from PySONIC.core import Model, PointNeuron
from PySONIC.postpro import detectSpikes, prependDataFrame
from PySONIC.utils import logger, si_format, filecode, simAndSave, isIterable
from PySONIC.constants import *
from PySONIC.threshold import threshold, titrate, Thresholder

from .pyhoc import *
from .sources import *
from ..utils import array_print_options
from ..constants import *


class NeuronModel(metaclass=abc.ABCMeta):
    ''' Generic interface for NEURON models. '''

    tscale = 'ms'                 # relevant temporal scale of the model
    section_class = MechQSection  # default type of NEURON section
    is_constructed = False

    # integration methods
    int_methods = {
        0: 'backward Euler method',
        1: 'Crank-Nicholson method',
        2: 'Crank-Nicholson method with fixed currents at mid-steps',
        3: 'CVODE multi order variable time step method',
        4: 'DASPK (Differential Algebraic Solver with Preconditioned Krylov) method'
    }

    def __init__(self, construct=True):
        ''' Initialization. '''
        logger.debug(f'Creating {self} model')
        load_mechanisms(getNmodlDir(), self.modfile)
        if construct:
            self.construct()

    def set(self, attrkey, value):
        ''' Set attribute if not existing or different, and reset model if already constructed. '''
        realkey = f'_{attrkey}'
        if not hasattr(self, realkey) or value != getattr(self, realkey):
            setattr(self, realkey, value)
            if self.is_constructed:
                logger.debug(f'resetting model with {attrkey} = {value}')
                self.reset()

    @property
    @abc.abstractmethod
    def simkey(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self):
        raise NotImplementedError

    @property
    def modelcodes(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name
        }

    @property
    def modelcode(self):
        return '_'.join(self.modelcodes.values())

    @property
    def pneuron(self):
        return self._pneuron

    @pneuron.setter
    def pneuron(self, value):
        if not isinstance(value, PointNeuron):
            raise TypeError(f'{value} is not a valid PointNeuron instance')
        self.set('pneuron', value)

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
        return rho / cls.axialSectionArea(*args, **kwargs) / M_TO_CM**2  # Ohm/cm

    @classmethod
    def axialResistance(cls, rho, L, *args, **kwargs):
        ''' Compute the axial resistance of a cylindrical section.

            :param rho: axial resistivity (Ohm.cm)
            :param L: cylinder length (m)
            :return: resistance (Ohm)
        '''
        return cls.axialResistancePerUnitLength(rho, *args, **kwargs) * L * M_TO_CM  # Ohm

    def setCelsius(self, celsius=None):
        if celsius is None:
            try:
                celsius = self.pneuron.celsius
            except AttributeError:
                raise ValueError('celsius value not provided and not found in PointNeuron class')
        h.celsius = celsius

    @property
    @abc.abstractmethod
    def seclist(self):
        raise NotImplementedError

    @property
    def nsections(self):
        return len(self.seclist)

    def construct(self):
        ''' Create, specify and connect morphological model sections. '''
        self.createSections()
        self.setGeometry()
        self.setResistivity()
        self.setBiophysics()
        self.setExtracellular()
        self.setTopology()
        self.is_constructed = True

    @abc.abstractmethod
    def createSections(self):
        ''' Create morphological sections. '''
        raise NotImplementedError

    def setGeometry(self):
        ''' Set sections geometry. '''
        pass

    def setResistivity(self):
        ''' Set sections axial resistivity. '''
        pass

    def setBiophysics(self):
        ''' Set the membrane biophysics of all model sections. '''
        for sec in self.seclist:
            if isinstance(sec, MechSection) and sec.mechname is not None:
                sec.insert(sec.mechname)
        if issubclass(self.section_class, QSection):
            self.setFuncTables()

    def setTopology(self):
        ''' Connect morphological sections. '''
        pass

    def setExtracellular(self):
        ''' Set the sections' extracellular mechanisms. '''
        pass

    @abc.abstractmethod
    def clear(self):
        ''' Clear all model sections and drive objects. '''
        raise NotImplementedError

    def reset(self):
        ''' Delete and re-construct all model sections. '''
        self.clear()
        self.construct()

    def createSection(self, id, *args, mech=None, states=None, Cm0=None):
        ''' Create a model section with a given id. '''
        if Cm0 is None:
            Cm0 = self.pneuron.Cm0 * F_M2_TO_UF_CM2  # uF/cm2
        args = [x for x in args if x is not None]
        return self.section_class(*args, mechname=mech, states=states, name=id, cell=self, Cm0=Cm0)

    def getIntegrationMethod(self):
        ''' Get the method used by NEURON for the numerical integration of the system. '''
        method_type_code = self.cvode.current_method() % 1000 // 100
        method_type_str = self.int_methods[method_type_code]
        if self.cvode.active():
            return f'{method_type_str} (atol = {self.cvode.atol()})'
        else:
            return f'{method_type_str} (fixed dt = {h.dt} ms)'

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        h.t = 0
        self.setStimValue(0)
        if issubclass(self.section_class, QSection):
            x0 = self.pneuron.Qm0 * C_M2_TO_NC_CM2  # nC/cm2
        else:
            x0 = self.pneuron.Vm0  # mV
        h.finitialize(x0)

    def setTimeProbe(self):
        ''' Set time probe. '''
        return Probe(h._ref_t)

    def setStimValue(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        # Set "stimon" attribute in all model sections
        for sec in self.seclist:
            sec.setStimON(value)
        # Set multiplying factor of all model drives
        for drive in self.drives:
            drive.set(value)

        # For all transitions except the one at time zero
        if h.t > 0:
            # If adaptive solver: re-initialize the integrator
            if self.cvode.active():
                self.cvode.re_init()
            # Otherwise, re-align currents with current states and potential
            else:
                h.fcurrent()

    def createStimSetter(self, value):
        return lambda: self.setStimValue(value)

    @staticmethod
    def fixStimVec(stim, dt):
        ''' Quick fix for stimulus vector discrepancy for fixed time step simulations. '''
        if dt is None:
            return stim
        else:
            return np.hstack((stim[1:], stim[-1]))

    def integrate(self, pp, dt, atol):
        ''' Integrate a model differential variables for a given duration, while updating the
            value of the boolean parameter stimon during ON and OFF periods throughout the numerical
            integration, according to stimulus parameters.

            Integration uses an adaptive time step method by default.

            :param pp: pulsed protocol object
            :param dt: integration time step (s). If provided, the fixed time step method is used.
            :param atol: absolute error tolerance. If provided, the adaptive
                time step method is used.
        '''
        # Set integration parameters
        self.cvode = h.CVode()
        if dt is not None:
            h.secondorder = 0  # using backward Euler method if fixed time step
            h.dt = dt * S_TO_MS
            self.cvode.active(0)
        else:
            self.cvode.active(1)
            if atol is not None:
                def_atol = self.cvode.atol()
                self.cvode.atol(atol)

        # Initialize
        self.initToSteadyState()

        # Set events
        for tevent, new_stim_value in pp.stimEvents():
            if tevent == 0:
                self.setStimValue(new_stim_value)
            else:
                self.cvode.event((tevent - TRANSITION_DT) * S_TO_MS)
                self.cvode.event(tevent * S_TO_MS, self.createStimSetter(new_stim_value))

        # Integrate
        logger.debug(f'integrating system using {self.getIntegrationMethod()}')
        tstop = pp.tstop * S_TO_MS
        while h.t < tstop:
            h.fadvance()

        # Set absolute error tolerance back to default value if changed
        if atol is not None:
            self.cvode.atol(def_atol)

        return 0

    def setPyLookup(self):
        ''' Set the appropriate model 2D lookup. '''
        if not hasattr(self, 'pylkp') or self.pylkp is None:
            self.pylkp = self.pneuron.getLookup()
            self.pylkp.refs['A'] = np.array([0.])
            for k, v in self.pylkp.items():
                self.pylkp[k] = np.array([v])

    def setModLookup(self, *args, **kwargs):
        ''' Get the appropriate model 2D lookup and translate it to Hoc. '''
        # Set Lookup
        self.setPyLookup(*args, **kwargs)

        # Convert lookups independent variables to hoc vectors
        self.Aref = h.Vector(self.pylkp.refs['A'] * PA_TO_KPA)
        self.Qref = h.Vector(self.pylkp.refs['Q'] * C_M2_TO_NC_CM2)

        # Convert lookup tables to hoc matrices
        # !!! hoc lookup dictionary must be a member of the class,
        # otherwise the assignment below does not work properly !!!
        self.lkp = {'V': Matrix(self.pylkp['V'])}  # mV
        for ratex in self.pneuron.alphax_list.union(self.pneuron.betax_list):
            self.lkp[ratex] = Matrix(self.pylkp[ratex] / S_TO_MS)
        for taux in self.pneuron.taux_list:
            self.lkp[taux] = Matrix(self.pylkp[taux] * S_TO_MS)
        for xinf in self.pneuron.xinf_list:
            self.lkp[xinf] = Matrix(self.pylkp[xinf])

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

        # Call function
        fillTable(matrix._ref_x[0][0], nx, xref._ref_x[0], ny, yref._ref_x[0])

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

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    def simulate(self, drive, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param drive: drive object
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
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
        t = t.to_array() / S_TO_MS  # s
        stim = self.fixStimVec(stim.to_array(), dt)
        data = pd.DataFrame({
            't': t,  # s
            'stimstate': stim
        })
        for k, v in probes.items():
            data[k] = v.to_array()
        data.loc[:, 'Qm'] /= C_M2_TO_NC_CM2  # C/m2

        # Prepend initial conditions (prior to stimulation)
        data = prependDataFrame(data)

        return data

    @property
    def titrationFunc(self):
        return self.pneuron.titrationFunc

    def titrate(self, *args, **kwargs):
        return titrate(self, *args, **kwargs)

    def getPltVars(self, *args, **kwargs):
        return self.pneuron.getPltVars(*args, **kwargs)

    @property
    def pltScheme(self):
        return self.pneuron.pltScheme

    @property
    @abc.abstractmethod
    def filecodes(self, *args):
        raise NotImplementedError

    def filecode(self, *args):
        return filecode(self, *args)

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)


class SpatiallyExtendedNeuronModel(NeuronModel):
    ''' Generic interface for spatially-extended NEURON models. '''

    # Boolean stating whether to use equivalent currents for imposed extracellular voltage fields
    use_equivalent_currents = False

    @abc.abstractstaticmethod
    def getMetaArgs(meta):
        raise NotImplementedError

    @classmethod
    def initFromMeta(cls, meta, construct=False):
        args, kwargs = cls.getMetaArgs(meta)
        return cls(*args, **kwargs, construct=construct)

    @staticmethod
    def inputs():
        return {
            'section': {
                'desc': 'section',
                'label': 'section',
                'unit': ''
            }
        }

    def filecodes(self, source, pp, *_):
        return {
            **self.modelcodes,
            **source.filecodes,
            'nature': pp.nature,
            **pp.filecodes
        }

    @property
    def rmin(self):
        ''' Lower bound for axial resistance * membrane area (Ohm/cm2). '''
        return None

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, value):
        if value <= 0:
            raise ValueError('longitudinal resistivity must be positive')
        self.set('rs', value)

    def str_resistivity(self):
        return f'rs = {si_format(self.rs)}Ohm.cm'

    @property
    @abc.abstractmethod
    def refsection(self):
        ''' Model reference section (used mainly to monitor stimon parameter). '''
        raise NotImplementedError

    @property
    def sectypes(self):
        return list(self.sections.keys())

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

    @property
    @abc.abstractmethod
    def nonlinear_sections(self):
        ''' Sections that contain nonlinear dynamics. '''
        raise NotImplementedError

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
            if not hasattr(item, 'set'):
                raise ValueError(f'drive {item} has no "set" method')
        self._drives = value

    def setOtherProbes(self):
        return {}

    def desc(self, meta):
        return f'{self}: simulation @ {meta["source"]}, {meta["pp"].desc}'

    def connect(self, k1, i1, k2, i2):
        ''' Connect two sections referenced by their type and index.

            :param k1: type of parent section
            :param i1: index of parent section in subtype dictionary
            :param k2: type of child section
            :param i2: index of child section in subtype dictionary
        '''
        self.sections[k2][f'{k2}{i2:d}'].connect(self.sections[k1][f'{k1}{i1:d}'])

    def setIClamps(self, Iinj_dict):
        ''' Set distributed intracellular current clamps. '''
        logger.debug(f'Intracellular currents:')
        with np.printoptions(**array_print_options):
            for k, Iinj in Iinj_dict.items():
                logger.debug(f'{k}: Iinj = {Iinj} nA')
        iclamps = []
        for k, Iinj in Iinj_dict.items():
            iclamps += [IClamp(sec, I) for sec, I in zip(self.sections[k].values(), Iinj)]
        return iclamps

    def toInjectedCurrents(self, Ve):
        ''' Convert extracellular potential array into equivalent injected currents.

            :param Ve: model-sized vector of extracellular potentials (mV)
            :return: model-sized vector of intracellular injected currents (nA)
        '''
        raise NotImplementedError

    def setVext(self, Ve_dict):
        ''' Set distributed extracellular voltages. '''
        logger.debug(f'Extracellular potentials:')
        with np.printoptions(**array_print_options):
            for k, Ve in Ve_dict.items():
                logger.debug(f'{k}: Ve = {Ve} mV')
        if self.use_equivalent_currents:
            # Variant 1: inject equivalent intracellular currents
            return self.setIClamps(self.toInjectedCurrents(Ve_dict))
        else:
            # Variant 2: insert extracellular mechanisms for a more realistic depiction
            # of the extracellular field
            emechs = []
            for k, Ve in Ve_dict.items():
                emechs += [ExtField(sec, v) for sec, v in zip(self.sections[k].values(), Ve)]
            return emechs

    @property
    def drive_funcs(self):
        return {
            IntracellularCurrent: self.setIClamps,
            ExtracellularCurrent: self.setVext,
            GaussianVoltageSource: self.setVext
        }

    def setDrives(self, source):
        ''' Set distributed stimulus amplitudes. '''
        self.drives = []
        amps_dict = source.computeDistributedAmps(self)
        match = False
        for source_class, drive_func in self.drive_funcs.items():
            if isinstance(source, source_class):
                self.drives = drive_func(amps_dict)
                match = True
        if not match:
            raise ValueError(f'Unknown source type: {source}')

    @property
    def Aranges(self):
        return {
            IntracellularCurrent: IINJ_RANGE,
            ExtracellularCurrent: VEXT_RANGE
        }

    def getArange(self, source):
        ''' Get the stimulus amplitude range allowed at the fiber level. '''
        for source_class, Arange in self.Aranges.items():
            if isinstance(source, source_class):
                return [source.computeSourceAmp(self, x) for x in Arange]
        raise ValueError(f'Unknown source type: {source}')

    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    def simulate(self, source, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param source: source object
            :param pp: pulsed protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
            :return: output dataframe
        '''
        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.refsection.setStimProbe()
        full_probes = {k: v.setProbesDict() for k, v in self.nonlinear_sections.items()}
        other_probes = self.setOtherProbes()

        # Set distributed drives
        self.setDrives(source)

        # Integrate model
        self.integrate(pp, dt, atol)

        # Store output in dataframes
        t = t.to_array() / S_TO_MS  # s
        stim = self.fixStimVec(stim.to_array(), dt)
        data = {}

        # Full states data for sections with nonlinear mechanisms
        for id, probes in full_probes.items():
            data[id] = pd.DataFrame({
                't': t,
                'stimstate': stim,
                **{k: v.to_array() for k, v in probes.items()}
            })
            data[id].loc[:, 'Qm'] /= C_M2_TO_NC_CM2  # C/m2

        # Voltage data only for other sections
        for sectype, probes_dict in other_probes.items():
            for k, v in probes_dict.items():
                data[k] = pd.DataFrame({
                    't': t,
                    'stimstate': stim,
                    'Vm': v['Vm'].to_array()})

        # Prepend initial conditions (prior to stimulation)
        data = {id: prependDataFrame(df) for id, df in data.items()}

        return data

    def getSpikesTimings(self, data, zcross=True, spikematch='majority'):
        ''' Return an array containing occurence times of spikes detected on a collection of sections.

            :param data: simulation output dataframe
            :param zcross: boolean stating whether to use ascending zero-crossings preceding peaks
                as temporal reference for spike occurence timings
            :return: dictionary of spike occurence times (s) per section.
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
                        assert tzc < tpeak, errmsg.format(
                            ispike, tzc * S_TO_MS, ispike, tpeak * S_TO_MS)
                    tspikes[id] = tzcross
                else:
                    tspikes[id] = t[ispikes]

        if spikematch == 'strict':
            # Assert consistency of spikes propagation
            assert np.all(nspikes == nspikes[0]), 'Inconsistent spike number across sections'
            if nspikes[0] == 0:
                logger.warning('no spikes detected')
                return None
        else:
            # Use majority voting
            nfrequent = np.int(stats.mode(nspikes).mode)
            tspikes = {k: v for k, v in tspikes.items() if len(v) == nfrequent}

        return pd.DataFrame(tspikes)

    def getSpikeAmp(self, data, ids=None, key='Vm', out='range'):
        # By default, consider all sections with nonlinear dynamics
        if ids is None:
            ids = list(self.nonlinear_sections.keys())
        amps = np.array([np.ptp(data[id][key].values) for id in ids])
        if out == 'range':
            return amps.min(), amps.max()
        elif out == 'median':
            return np.median(amps)
        elif out == 'mean':
            return np.mean(amps)
        else:
            raise AttributeError(f'invalid out option: {out}')

    def titrationFunc(self, data):
        return self.isExcited(data)

    def getStartPoint(self, Arange):
        scale = 'lin' if Arange[0] == 0 else 'log'
        return Thresholder.getStartPoint(Arange, x=REL_START_POINT, scale=scale)

    def getAbsConvThr(self, Arange):
        return np.abs(Arange[1] - Arange[0]) / 1e4

    def titrate(self, source, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param source: source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
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


class FiberNeuronModel(SpatiallyExtendedNeuronModel):
    ''' Generic interface for fiber (single or double cable) NEURON models. '''

    # Boolean stating whether to use equivalent currents for imposed extracellular voltage fields
    use_equivalent_currents = True

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
        if value % 2 == 0:
            raise ValueError('number of nodes must be odd')
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
    def rhoa(self):
        ''' Axoplasmic resistivity (Ohm.cm) '''
        return self.rs

    @property
    def R_node(self):
        ''' Node intracellular axial resistance (Ohm). '''
        return self.axialResistance(self.rhoa, self.nodeL, self.nodeD)

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

    @property
    def z(self):
        return 0.

    def getXZCoords(self):
        return {k: np.vstack((v, np.ones(v.size) * self.z)).T
                for k, v in self.getXCoords().items()}  # m

    @abc.abstractmethod
    def isInternodalDistance(self, d):
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

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        nspikes_start = detectSpikes(data[self.nodeIDs[0]])[0].size
        nspikes_end = detectSpikes(data[self.nodeIDs[-1]])[0].size
        return nspikes_start > 0 and nspikes_end > 0

    def titrate(self, source, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param source: source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
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
