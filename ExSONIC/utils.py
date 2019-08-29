# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 14:38:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-29 11:46:26

import os
import pickle
import numpy as np
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
    if np.all(Ithrs < 0.):
        Ithrs = -Ithrs
    Irh = 2 * Ithrs.min()  # rheobase current
    return np.interp(Irh, Ithrs[::-1], durations[::-1])  # s