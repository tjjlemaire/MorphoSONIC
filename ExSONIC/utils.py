# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 14:38:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-15 12:19:13

import os
import pickle
import platform
import numpy as np
from scipy.optimize import curve_fit
from neuron import h

from PySONIC.utils import logger

# global list of paths already loaded by load_mechanisms
nrn_dll_loaded = []

array_print_options = {
    'precision': 2,
    'suppress': True,
    'linewidth': 150,
    'sign': '+'
}


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


def loadData(fpath, frequency=1):
    ''' Load dataframe and metadata dictionary from pickle file. '''
    logger.info(f'Loading data from "{os.path.basename(fpath)}"')
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        data = frame['data']
        data = {k: df.iloc[::frequency] for k, df in data.items()}
        meta = frame['meta']
        return data, meta


def getNmodlDir():
    ''' Return path to directory containing MOD files and compiled mechanisms files. '''
    selfdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(selfdir, 'nmodl')


def chronaxie(durations, thrs):
    ''' Return chronaxie, i.e. stimulus duration for which threshold amplitude
        is twice the rheobase. '''
    if np.all(thrs[np.logical_not(np.isnan(thrs))] < 0.):
        thrs = -thrs

    Ich = 2 * np.nanmin(thrs)   # rheobase current
    return np.interp(Ich, thrs[::-1], durations[::-1], left=np.nan, right=np.nan)  # s


def rheobase(thrs):
    ''' Return rheobase. '''
    return np.nanmin(np.abs(thrs))


def WeissSD(t, tau_e, I0):
    ''' Weiss' classical formulation of the strength-duration relationship,
        taken from Reilly, 2011.

        :param t: pulse duration (s)
        :param tau_e: S/D time constant
        :param I0: rheobase current (A)
        :return: threshold current for the input pulse duration (A)

        Reference: Reilly, J.P. (2011). Electrostimulation: theory, applications,
        and computational model (Boston: Artech House).
    '''
    return I0 * (1 + tau_e / t)


def LapiqueSD(t, tau_e, I0):
    ''' Lapique's exponential formulation of the strength-duration relationship,
        taken from Reilly, 2011.

        :param t: pulse duration (s)
        :param tau_e: S/D time constant
        :param I0: rheobase current (A)
        :return: threshold current for the input pulse duration (A)

        Reference: Reilly, J.P. (2011). Electrostimulation: theory, applications,
        and computational model (Boston: Artech House).
    '''
    return I0 / (1 - np.exp(-t / tau_e))


def fitTauSD(durations, currents, method='Weiss'):
    candidate_fits = {'Weiss': WeissSD, 'Lapique': LapiqueSD}
    try:
        method = candidate_fits[method]
    except KeyError:
        raise ValueError(f'"method" must one of ({", ".join([list(candidate_fits.keys())])})')
    I0 = currents.min()
    tau_e = curve_fit(
        lambda t, tau: method(t, tau, I0),
        durations, currents, p0=chronaxie(durations, currents))[0][0]
    return tau_e, method(durations, tau_e, I0)


def seriesGeq(*G):
    ''' Return the equivalent conductance for n conductances in parallel.

        :param G: list of conductances
        :return: equivalent series condictance.
    '''
    if 0. in G:
        return 0.
    return 1 / sum(map(lambda x: 1 / x, G))
