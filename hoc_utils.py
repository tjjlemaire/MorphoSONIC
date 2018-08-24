# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-21 19:48:04
# @Last Modified by:   Theo
# @Last Modified time: 2018-08-24 22:26:13


''' Utilities to manipulate HOC objects. '''

import numpy as np
from neuron import h

# Aliases for NMODL-protected variable names
NEURON_aliases = {'O': 'O1', 'C': 'C1'}

# List of neuron DLLs already loaded in memory
nrn_dll_loaded = []


def isAlreadyLoaded(dll_file):
    if dll_file in nrn_dll_loaded:
        return True
    else:
        nrn_dll_loaded.append(dll_file)
        return False


def alias(state):
    ''' Return NEURON state alias. '''
    return NEURON_aliases[state] if state in NEURON_aliases.keys() else state


def array2Matrix(arr):
    ''' Convert 2D numpy array to Hoc Matrix.

        :param arr: 2D numpy array
        :return: HOC Matrix object
    '''

    nx, ny = arr.shape
    matrix = h.Matrix(nx, ny)
    for i in range(nx):
        matrix.setrow(i, h.Vector(arr[i, :]))
    return matrix


def Vec2array(vec):
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


def attachIClamp(sec, dur, amp, delay=0, loc=0.5):
    ''' Attach a current Clamp to a section.

        :param sec: section to attach the current clamp.
        :param dur: duration of the stimulus (ms).
        :param amp: magnitude of the current (mA/m2).
        :param delay: onset of the injected current (ms)
        :param loc: location on the section where the stimulus is placed
        :return: IClamp object (must be returned to caller space to be effective)
    '''
    pulse = h.IClamp(sec(loc))
    pulse.delay = delay
    pulse.dur = dur
    pulse.amp = amp * sec(0.5).area() * 1e-6  # nA

    return pulse


def attachEStim(sec, Astim, tstim, PRF, DC, loc=0.5):
    ''' Attach a series of current clamps to a section to simulate a pulsed electrical stimulus.

        :param sec: section to attach current clamps.
        :param Astim: injected current density (mA/m2).
        :param tstim: duration of the stimulus (ms)
        :param PRF: pulse repetition frequency (kHz)
        :param DC: stimulus duty cycle
        :param loc: location on the section where the stimulus is placed
        :return: list of iclamp objects
    '''
    # Update PRF for CW stimuli to optimize integration
    if DC == 1.0:
        PRF = 1 / tstim

    # Compute pulses timing
    Tpulse = 1 / PRF
    Ton = DC * Tpulse
    npulses = int(np.round(tstim / Tpulse))

    return [attachIClamp(sec, Ton, Astim, delay=i * Tpulse, loc=loc) for i in range(npulses)]


def setTimeProbe():
    ''' Set recording vector for time.

        :return: time recording vector
    '''
    t = h.Vector()
    t.record(h._ref_t)
    return t


def setStimProbe(cell):
    ''' Set recording vector for stimulation state.

        :param cell: cell to record from
        :return: stimulation state recording vector
    '''
    states = h.Vector()
    states.record(getattr(cell.sections[0](0.5), cell.mechname)._ref_stimon)
    return states


def setRangeProbes(sections, var):
    ''' Set recording vectors for a range variable in different sections.

        :param sections: sections to record from
        :param var: range variable to record
        :return: list of recording vectors
    '''
    probes = []
    for sec in sections:
        probes.append(h.Vector())
        probes[-1].record(getattr(sec(0.5), '_ref_{}'.format(var)))
    return probes
