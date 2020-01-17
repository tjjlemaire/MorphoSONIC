# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-16 19:24:42
# @Author: Theo Lemaire
# @Date:   2018-08-21 19:48:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-11 10:20:22


''' Utilities to manipulate HOC objects. '''

import os
import platform
import numpy as np
from neuron import h

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


def setTimeProbe():
    ''' Set recording vector for time.

        :return: time recording vector
    '''
    t = h.Vector()
    t.record(h._ref_t)
    return t


def setStimProbe(section, mechname):
    ''' Set recording vector for stimulation state.

        :param section: section to record from
        :param mechname: variable parent mechanism
        :return: stimulation state recording vector
    '''
    states = h.Vector()
    states.record(getattr(section(0.5), mechname)._ref_stimon)
    return states


def setRangeProbe(section, var, loc=0.5):
    ''' Set recording vector for a range variable in a specific section location.

        :param section: section to record from
        :param var: range variable to record
        :return: list of recording vectors
    '''
    probe = h.Vector()
    probe.record(getattr(section(loc), '_ref_{}'.format(var)))
    return probe


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