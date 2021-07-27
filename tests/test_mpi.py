# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-16 11:36:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:46:58

import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.core import Batch, getPulseTrainProtocol
from PySONIC.utils import logger
from MorphoSONIC.models import SennFiber
from MorphoSONIC.sources import GaussianAcousticSource
from MorphoSONIC.plt import SectionCompTimeSeries

''' Small script showcasing multiprocessing possibilities for Python+NEURON simulations. '''

logger.setLevel(logging.INFO)

# Fiber model
a = 32e-9  # m
fs = 1.
fiberD = 20e-6  # m
nnodes = 21
fiber = SennFiber(fiberD, nnodes=nnodes, a=a, fs=fs)

# Acoustic source
Fdrive = 500e3  # Hz
amps = np.linspace(50e3, 400e3, 8)  # Pa
source = GaussianAcousticSource(0, fiber.length / 10., Fdrive)

# Pulsing protocol
tpulse = 100e-6  # s
PRF = 10  # Hz
npulses = 10
pp = getPulseTrainProtocol(tpulse, PRF, npulses)

if __name__ == '__main__':
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='Use multiprocessing')
    args = parser.parse_args()
    mpi = args.mpi

    # Initialize and run simulation batch
    batch = Batch(lambda x: fiber.simulate(source.updatedX(x), pp), [[x] for x in amps])
    output = batch.run(loglevel=logger.getEffectiveLevel(), mpi=mpi)

    # Plot results
    for data, meta in output:
        SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

    plt.show()
