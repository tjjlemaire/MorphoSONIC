# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-22 18:53:41

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from ExSONIC.utils import radialGeometry
from ExSONIC._1D import SeriesConnector, compareEStim, runPlotAStim


def main():

    # Define argument parser
    ap = ArgumentParser()
    ap.add_argument('-t', '--testset', type=str, default='all', help='Test set')
    ap.add_argument('-n', '--neuron', type=str, default='RS', help='Neuron type')
    ap.add_argument('--fs', type=float, default=50, help='Membrane sonophore coverage')
    ap.add_argument('--deff', type=float, default=1e-1, help='Submembrane depth')
    ap.add_argument('-A', '--Adrive', type=float, default=None, help='US pressure amplitude')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('--noplot', default=False, action='store_true',
                    help='Do not show figures')

    # Parse arguments
    args = ap.parse_args()

    # Model parameters
    neuron = getNeuronsDict()[args.neuron]()
    rs = 1e2  # Ohm.cm
    config = 'first'

    # Set zero-length internodes
    interD = 1.  # um
    interL = 0.  # um

    # Parse test set
    alltests = ['Iintra', 'US']
    testset = alltests if args.testset == 'all' else [args.testset]
    for test in testset:
        assert test in alltests, '{} is not a valid test'.format(test)

    # # Iintra
    # if 'Iintra' in testset:

    #     # Standard intracellular cathodic stimulus
    #     Astim = 10.0
    #     tstim = 100e-3  # s
    #     toffset = 50e-3  # s
    #     PRF = 100.  # Hz
    #     DC = 1.0

    #     connector = SeriesConnector(vref='v', rmin=None)
    #     compareEStim(neuron, rs, connector, nodeD, nodeL, interD, interL,
    #                  Astim, tstim, toffset, PRF, DC, nnodes=nnodes,
    #                  cmode='seq', verbose=args.verbose, config=config)

    # US
    if 'US' in testset:

        # Standard US stimulus
        a = 32.  # nm
        Fdrive = 500.  # kHz
        tstim = 100e-3  # s
        toffset = 50e-3  # s
        PRF = 100.  # Hz
        Adrive = args.Adrive  # kPa
        DC = 1.

        nodeD, nodeL = radialGeometry(args.deff, a * 1e-3, fc=args.fs / 100.)

        connector = SeriesConnector(vref='Vmeff_{}'.format(neuron.name))
        runPlotAStim(neuron, a, Fdrive, rs, connector, nodeD, nodeL, interD, interL,
                     Adrive, tstim, toffset, PRF, DC, cmode='qual', verbose=args.verbose,
                     config=config)

    if not args.noplot:
        plt.show()
    sys.exit(0)


if __name__ == '__main__':
    main()
