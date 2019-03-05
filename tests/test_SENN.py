# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-04 19:49:51

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
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('--noplot', default=False, action='store_true',
                    help='Do not show figures')

    # Parse arguments
    args = ap.parse_args()

    # Model parameters
    neuron = getNeuronsDict()['FH']()
    nnodes = 15
    rs = 1e2  # Ohm.cm
    fiberD = 20.0  # um
    nodeD, nodeL, interD, interL = sennGeometry(fiberD)
    config = 'central'

    # Parse test set
    alltests = ['Iintra', 'US']
    testset = alltests if args.testset == 'all' else [args.testset]
    for test in testset:
        assert test in alltests, '{} is not a valid test'.format(test)

    # stimulation parameters
    tstim = 1e-4  # s
    toffset = 3e-3  # s
    PRF = 100.  # Hz
    DC = 1.0

    # Iintra
    if 'Iintra' in testset:

        # Reproduction of figure 2 (with intracellular current injection) from:
        # Reilly, J.P., Freeman, V.T., and Larkin, W.D. (1985). Sensory effects of transient
        # electrical stimulation--evaluation with a neuroelectric model.
        # IEEE Trans Biomed Eng 32, 1001–1011.

        connector = SeriesConnector(vref='v', rmin=None)
        compareEStim(neuron, rs, connector, nodeD, nodeL, interD, interL,
                     None, tstim, toffset, PRF, DC, nnodes=nnodes,
                     cmode='seq', verbose=args.verbose, config=config)

    # US
    if 'US' in testset:

        # Application of SENN model to US stimuli
        a = 32.  # nm
        Fdrive = 500.  # kHz
        Adrive = 100.  # kPa

        tstim = 10e-3  # s
        toffset = 30e-3  # s

        connector = SeriesConnector(vref='Vmeff_{}'.format(neuron.name), rmin=None)
        runPlotAStim(neuron, a, Fdrive, rs, connector, nodeD, nodeL, interD, interL,
                     Adrive, tstim, toffset, PRF, DC, nnodes=nnodes,
                     cmode='seq', verbose=args.verbose, config=config)

    if not args.noplot:
        plt.show()
    sys.exit(0)


if __name__ == '__main__':
    main()
