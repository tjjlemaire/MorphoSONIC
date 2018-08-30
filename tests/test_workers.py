# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-08-30 13:51:29
# @Last Modified by:   Theo
# @Last Modified time: 2018-08-30 14:15:39

import logging

from PySONIC.utils import logger
from PySONIC.solvers import checkBatchLog
from PySONIC.plt import plotBatch
from PySONIC.neurons import *
from ExSONIC._0D import AStimWorker, EStimWorker

# Set logging level
logger.setLevel(logging.DEBUG)

# Batch directory
batch_dir = 'C:/Users/Theo/Documents/test/'

# Model parameters
neuron = CorticalRS()
a = 32e-9  # sonophore diameter (m)

# Stimulation parameters
Fdrive = 500e3  # Hz
Adrive = 100e3  # kPa
Astim = 30.0  # mA/m2
tstim = 150e-3  # s
toffset = 100e-3  # s
PRF = 100.  # Hz
DC = 1.0


# ------------------------- E-STIM ----------------------------------

# Get logfile
log_filepath, _ = checkBatchLog(batch_dir, 'E-STIM')

# Run simulation
worker = EStimWorker(1, batch_dir, log_filepath, neuron, Astim, tstim, toffset, PRF, DC)
print(worker)
outfilepath = worker.__call__()

# Plot profiles
yvars = {
    'v_m': ['Vm'],
    'i_{Na}\ kin.': ['m', 'h', 'm3h'],
    'i_K\ kin.': ['n'],
    'i_M\ kin.': ['p']
}
plotBatch(batch_dir, [outfilepath], title=True, vars_dict=yvars)


# ------------------------- A-STIM ----------------------------------

# Get logfile
log_filepath, _ = checkBatchLog(batch_dir, 'A-STIM')

# Run simulation
worker = AStimWorker(1, batch_dir, log_filepath, neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC)
print(worker)
outfilepath = worker.__call__()

# Plot profiles
yvars = {
    'Q_m': ['Qm'],
    'i_{Na}\ kin.': ['m', 'h', 'm3h'],
    'i_K\ kin.': ['n'],
    'i_M\ kin.': ['p']
}
plotBatch(batch_dir, [outfilepath], title=True, vars_dict=yvars)
