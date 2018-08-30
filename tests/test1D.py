# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 12:25:55

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import *
from ExSONIC._1D import SeriesConnector, compareEStim, runPlotAStim


def runTests(hide):

    # Model parameters
    neuron = CorticalRS()
    a = 32e-9  # sonophore diameter (m)
    nnodes = 3
    Ra = 1e2  # default order of magnitude found in litterature (Ohm.cm)
    d = 1e-6  # order of magnitude of axon node diameter (m)
    L = 1e-5  # between length order of magnitude of axon node (1 um) and internode (100 um - 1 mm)

    # Stimulation parameters
    Fdrive = 500e3  # Hz
    Adrive = 50e3  # kPa
    Astim = 30.0  # mA/m2
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    PRF = 100.  # Hz
    DC = 1.0

    compareEStim(neuron, nnodes, d, L, Ra, SeriesConnector(vref='v'), Astim, tstim, toffset, PRF, DC)

    runPlotAStim(neuron, nnodes, d, L, Ra, SeriesConnector(vref='Vmeff_{}'.format(neuron.name)),
                 a, Fdrive, Adrive, tstim, toffset, PRF, DC, cmode='seq')

    if not hide:
        plt.show()


def main():

    # Define argument parser
    ap = ArgumentParser()
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-n', '--noplot', default=False, action='store_true',
                    help='Do not show figures')

    # Parse arguments
    args = ap.parse_args()
    if args.verbose:
        pass
    else:
        pass

    # Run tests
    runTests(args.noplot)
    sys.exit(0)


if __name__ == '__main__':
    main()
