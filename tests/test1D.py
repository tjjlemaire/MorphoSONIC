# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-15 23:37:03

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from ExSONIC._1D import SeriesConnector, compareEStim, runPlotAStim


def runTests(testset, neuron, verbose, hide):

    # Model parameters
    neuron = getNeuronsDict()[neuron]()
    a = 32e-9  # sonophore diameter (m)
    nnodes = 2
    Ra = 1e2  # default order of magnitude found in litterature (Ohm.cm)
    d = 1e-6  # order of magnitude of axon node diameter (m)
    L = 1e-5  # between length order of magnitude of axon node (1 um) and internode (100 um - 1 mm)

    # Stimulation parameters
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    PRF = 100.  # Hz
    DC = 1.0

    # E-STIM
    if 'ESTIM' in testset:
        Astim = 30.0  # mA/m2
        compareEStim(neuron, Ra, SeriesConnector(vref='v'), d, L, Astim, tstim, toffset, PRF, DC,
                     nnodes=nnodes)

    # A-STIM
    if 'ASTIM' in testset:
        Fdrive = 500e3  # Hz
        amps = [50e3, 50e3]  # Pa
        covs = [1.0, .5]
        runPlotAStim(neuron, a, Fdrive, Ra, SeriesConnector(vref='Vmeff_{}'.format(neuron.name)),
                     d, L, amps, tstim, toffset, PRF, DC, nnodes=nnodes, covs=covs)

    if not hide:
        plt.show()


def main():

    # Define argument parser
    ap = ArgumentParser()
    ap.add_argument('-t', '--testset', type=str, default='all', help='Test set')
    ap.add_argument('-n', '--neuron', type=str, default='RS', help='Neuron type')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('--noplot', default=False, action='store_true',
                    help='Do not show figures')

    # Parse arguments
    args = ap.parse_args()

    testset = args.testset
    if testset == 'all':
        testset = ['ESTIM', 'ASTIM']
    else:
        testset = [testset]

    # Run tests
    runTests(testset, args.neuron, args.verbose, args.noplot)
    sys.exit(0)


if __name__ == '__main__':
    main()
