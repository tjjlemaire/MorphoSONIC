# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 14:38:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-26 13:51:52

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


