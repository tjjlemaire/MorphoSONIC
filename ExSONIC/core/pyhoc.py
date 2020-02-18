# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-17 11:59:12
# @Author: Theo Lemaire
# @Date:   2018-08-21 19:48:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-11 10:20:22


''' Utilities to manipulate HOC objects. '''

import os
import platform
import numpy as np
from neuron import h, hclass

from PySONIC.utils import logger

# Aliases for NMODL-protected variable names
NEURON_aliases = {'O': 'O1', 'C': 'C1'}

# global list of paths already loaded by load_mechanisms
nrn_dll_loaded = []


def load_mechanisms(path, modfile=None):
    ''' Rewrite of NEURON's native load_mechanisms method to ensure Windows and Linux compatibility.

        :param path: full path to directory containing the MOD files of the mechanisms to load.
        :param modfile (optional): name of specific mechanism to check for untracked changes
        in source file.
    '''
    # If mechanisms of input path are already loaded, return silently
    global nrn_dll_loaded
    if path in nrn_dll_loaded:
        return

    # in case NEURON is assuming a different architecture to Python,
    # we try multiple possibilities
    libname = 'libnrnmech.so'
    libsubdir = '.libs'
    arch_list = [platform.machine(), 'i686', 'x86_64', 'powerpc', 'umac']

    # windows loads nrnmech.dll
    if h.unix_mac_pc() == 3:
        libname = 'nrnmech.dll'
        libsubdir = ''
        arch_list = ['']

    # check for library file existence with every possible architecture
    lib_path = None
    for arch in arch_list:
        candidate_lib_path = os.path.join(path, arch, libsubdir, libname)
        if os.path.exists(candidate_lib_path):
            lib_path = candidate_lib_path

    # if library file does not seem to exist, raise error
    if lib_path is None:
        raise RuntimeError(f'Compiled library file not found for mechanisms in "{path}"')

    # If mechanism name is provided, check for uncompiled changes in source file
    if modfile is not None:
        mod_path = os.path.join(path, modfile)
        if not os.path.isfile(mod_path):
            raise RuntimeError(f'"{modfile}" not found in "{path}"')
        if os.path.getmtime(mod_path) > os.path.getmtime(lib_path):
            raise UserWarning(f'"{modfile}" more recent than compiled library')

    # Load library file and add directory to list of loaded libraries
    h.nrn_load_dll(lib_path)
    nrn_dll_loaded.append(path)


def isAlreadyLoaded(dll_file):
    if dll_file in nrn_dll_loaded:
        return True
    else:
        nrn_dll_loaded.append(dll_file)
        return False


def alias(state):
    ''' Return NEURON state alias. '''
    return NEURON_aliases[state] if state in NEURON_aliases.keys() else state


def array_to_matrix(arr):
    ''' Convert 2D numpy array to Hoc Matrix.

        :param arr: 2D numpy array
        :return: HOC Matrix object
    '''
    nx, ny = arr.shape
    matrix = h.Matrix(nx, ny)
    for i in range(nx):
        matrix.setrow(i, h.Vector(arr[i, :]))
    return matrix


def vec_to_array(vec):
    ''' Convert Hoc vector to numpy array.

        :param vec: HOC Vector object
        :return: 1D numpy array
    '''
    return np.array(vec.to_python())


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
    fillTable = getattr(h, 'table_{}_{}'.format(fname, mechname))

    # Call function and return
    return fillTable(matrix._ref_x[0][0], nx, xref._ref_x[0], ny, yref._ref_x[0])


class Vector(hclass(h.Vector)):
    ''' Neuron vector with an extra method to and return itself as a numpy array. '''

    def to_array(self):
        return np.array(self.to_python())


def probe(variable):
    ''' Recording vector for a particular variable.

        :param variable: NEURON variable to record.
        :return: probe object recording the variable
    '''
    p = Vector()
    p.record(variable)
    return p


def setTimeProbe():
    ''' Set time probe. '''
    return probe(h._ref_t)


def setStimProbe(section, mechname):
    ''' Set recording vector for stimulation state.

        :param section: section to record from
        :param mechname: variable parent mechanism
        :return: stimulation state probe
    '''
    return probe(getattr(section(0.5), mechname)._ref_stimon)


def setRangeProbe(section, var, loc=0.5):
    ''' Set recording vector for a range variable in a specific section location.

        :param section: section to record from
        :param var: range variable to record
        :return: list of recording vectors
    '''
    return probe(getattr(section(loc), '_ref_{}'.format(var)))


class IClamp(hclass(h.IClamp)):
    ''' IClamp object that allows setting parameters on creation. '''

    def __init__(self, segment, amplitude):
        super().__init__(segment)
        self.delay = 0  # we want to exert control over amp starting at 0 ms
        self.dur = 1e9  # dur must be long enough to span all our changes
        self.amp = 0.  # initially, we set the amplitude to zero
        self.xamp = amplitude

    def toggle(self, value):
        self.amp = value * self.xamp


def setRangesProbes(sections, var, locs=None):
    ''' Set recording vectors for a range variable in different sections.

        :param sections: sections to record from
        :param var: range variable to record
        :return: list of recording vectors
    '''
    if locs is None:
        locs = [0.5] * len(sections)
    return map(setRangeProbe, sections, [var] * len(sections), locs)


def insertVext(sec, xr=1e20, xg=1e10, xc=0.):
    ''' Insert extracellular mechanism into section and set appropriate parameters.

        :param sec: section object
        :param xr: axial resistance of extracellular layer (Mohms/cm)
        :param xg: transverse conductance of extracellular layer (S/cm^2)
        :param xc: transverse capacitance of extracellular layer (uF/cm^2)
    '''
    sec.insert('extracellular')
    sec.xraxial[0] = xr
    sec.xg[0] = xg
    sec.xc[0] = xc


def integrate(model, pp, dt, atol):
    ''' Integrate a model differential variables for a given duration, while updating the
        value of the boolean parameter stimon during ON and OFF periods throughout the numerical
        integration, according to stimulus parameters.

        Integration uses an adaptive time step method by default.

        :param model: model instance
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
    model.Ton = DC / PRF
    model.Toff = (1 - DC) / PRF
    model.tstim = tstim

    # Set integration parameters
    h.secondorder = 2
    model.cvode = h.CVode()
    if dt is not None:
        h.dt = dt
        model.cvode.active(0)
        logger.debug(f'fixed time step integration (dt = {h.dt} ms)')
    else:
        model.cvode.active(1)
        if atol is not None:
            def_atol = model.cvode.atol()
            model.cvode.atol(atol)
            logger.debug(f'adaptive time step integration (atol = {model.cvode.atol()})')

    # Initialize
    model.stimon = model.setStimON(0)
    model.initToSteadyState()
    model.stimon = model.setStimON(1)
    model.cvode.event(model.Ton, model.toggleStim)

    # Integrate
    while h.t < tstop:
        h.fadvance()

    # Set absolute error tolerance back to default value if changed
    if atol is not None:
        model.cvode.atol(def_atol)

    return 0


def toggleStim(model):
    ''' Toggle stimulus state (ON -> OFF or OFF -> ON) and set appropriate next toggle event. '''
    # OFF -> ON at pulse onset
    if model.stimon == 0:
        model.stimon = model.setStimON(1)
        model.cvode.event(min(model.tstim, h.t + model.Ton), model.toggleStim)
    # ON -> OFF at pulse offset
    else:
        model.stimon = model.setStimON(0)
        if (h.t + model.Toff) < model.tstim - h.dt:
            model.cvode.event(h.t + model.Toff, model.toggleStim)

    # Re-initialize cvode if active
    if model.cvode.active():
        model.cvode.re_init()
    else:
        h.fcurrent()


def setStimON(model, value):
    ''' Set stimulation ON or OFF.

        :param value: new stimulation state (0 = OFF, 1 = ON)
        :return: new stimulation state
    '''
    for sec in model.sections.values():
        setattr(sec, f'stimon_{model.mechname}', value)
    return value


def setModLookup(model, *args, **kwargs):
    ''' Get the appropriate model 2D lookup and translate it to Hoc. '''
    # Set Lookup
    model.setPyLookup(*args, **kwargs)

    # Convert lookups independent variables to hoc vectors
    model.Aref = h.Vector(model.pylkp.refs['A'] * 1e-3)  # kPa
    model.Qref = h.Vector(model.pylkp.refs['Q'] * 1e5)   # nC/cm2

    # Convert lookup tables to hoc matrices
    # !!! hoc lookup dictionary must be a member of the class,
    # otherwise the assignment below does not work properly !!!
    model.lkp = {'V': array_to_matrix(model.pylkp['V'])}  # mV
    for ratex in model.pneuron.alphax_list.union(model.pneuron.betax_list):
        model.lkp[ratex] = array_to_matrix(model.pylkp[ratex] * 1e-3)  # ms-1
    for taux in model.pneuron.taux_list:
        model.lkp[taux] = array_to_matrix(model.pylkp[taux] * 1e3)  # ms
    for xinf in model.pneuron.xinf_list:
        model.lkp[xinf] = array_to_matrix(model.pylkp[xinf])  # (-)


def setFuncTables(model, *args, **kwargs):
    ''' Set neuron-specific interpolation tables along the charge dimension,
        and link them to FUNCTION_TABLEs in the MOD file of the corresponding
        membrane mechanism.
    '''
    if model.cell == model:
        logger.debug('loading %s membrane dynamics lookup tables', model.mechname)

    # Set Lookup
    model.setModLookup(*args, **kwargs)

    # Assign hoc matrices to 2D interpolation tables in membrane mechanism
    for k, v in model.lkp.items():
        setFuncTable(model.mechname, k, v, model.Aref, model.Qref)
