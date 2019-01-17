# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-17 19:20:03

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from ExSONIC.utils import sennGeometry
from ExSONIC._1D import SeriesConnector, compareEStim, runPlotAStim


def main():

    # Define argument parser
    ap = ArgumentParser()
    ap.add_argument('-t', '--testset', type=str, default='all', help='Test set')
    ap.add_argument('-n', '--neuron', type=str, default='FH', help='Neuron type')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('--noplot', default=False, action='store_true',
                    help='Do not show figures')

    # Parse arguments
    args = ap.parse_args()

    testset = args.testset
    if testset == 'all':
        testset = ['Iintra', 'Iextra', 'US']
    else:
        testset = [testset]

    # Model parameters
    neuron = getNeuronsDict()[args.neuron]()
    nnodes = 21
    rs = 1e2  # Ohm.cm
    fiberD = 10.0  # um
    nodeD, nodeL, interD, interL = sennGeometry(fiberD)

    # Iintra
    if 'Iintra' in testset:
        tstim = 1e-4  # s
        toffset = 3e-3  # s
        PRF = 100.  # Hz
        DC = 1.0
        connector = SeriesConnector(vref='v', rmin=None)
        compareEStim(neuron, rs, connector, nodeD, nodeL, interD, interL,
                     None, tstim, toffset, PRF, DC, nnodes=nnodes,
                     cmode='seq', verbose=args.verbose)

    # Iextra
    if 'Iextra' in testset:
        z0 = interL
        tstim = 1e-4  # s
        toffset = 10e-3  # s
        PRF = 100.  # Hz
        DC = 1.0
        dt = 1e-6  # s
        Iinj = None
        connector = SeriesConnector(vref='v', rmin=None)
        compareEStim(neuron, rs, connector, nodeD, nodeL, interD, interL,
                     Iinj, tstim, toffset, PRF, DC, nnodes=nnodes,
                     cmode='seq', verbose=args.verbose, z0=z0, dt=dt)

    # US
    if 'US' in testset:
        a = 32.  # nm
        Fdrive = 500.  # kHz
        amps = [50.] + [0.] * (nnodes - 1)  # kPa
        runPlotAStim(neuron, a, Fdrive, rs, SeriesConnector(vref='Vmeff_{}'.format(neuron.name)),
                     nodeD, nodeL, interD, interL, amps, tstim, toffset, PRF, DC, nnodes=nnodes,
                     verbose=args.verbose)

    if not args.noplot:
        plt.show()
    sys.exit(0)


if __name__ == '__main__':
    main()
