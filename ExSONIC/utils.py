# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 14:38:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-27 12:15:56

import os
import pickle
import numpy as np
from scipy.optimize import curve_fit
from neuron import h

from PySONIC.constants import *
from PySONIC.utils import si_format, logger


def loadData(fpath, frequency=1):
    ''' Load dataframe and metadata dictionary from pickle file. '''
    logger.info('Loading data from "%s"', os.path.basename(fpath))
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


def chronaxie(durations, Ithrs):
    ''' Return chronaxie, i.e. stimulus duration for which threshold current is twice the rheobase. '''
    if np.all(Ithrs[np.logical_not(np.isnan(Ithrs))] < 0.):
        Ithrs = -Ithrs
    Irh = 2 * np.nanmin(Ithrs)   # rheobase current
    return np.interp(Irh, Ithrs[::-1], durations[::-1])  # s


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


def extractIndexesFromLabels(labels):
    ''' Extract a list of indexes as integers from a list of labels containing indexes. '''
    prefix = os.path.commonprefix(labels)
    if len(prefix) == 0:
        return None
    labels_wo_prefix = [s.split(prefix)[1] for s in labels]
    try:
        indexed_labels = [int(s) for s in labels_wo_prefix]
        return prefix, indexed_labels
    except ValueError as err:
        return None