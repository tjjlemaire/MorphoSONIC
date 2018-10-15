# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 11:26:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-15 21:48:38

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from ExSONIC._0D import runPlotEStim, runPlotAStim, compareAStim



def runTests(neuron, verbose, hide):

    # Model parameters
    neuron = getNeuronsDict()[neuron]()
    a = 32e-9  # sonophore diameter (m)
    fs = 0.7   # membrane sonophore coverage fraction (-)

    # Stimulation parameters
    Fdrive = 500e3  # Hz
    Adrive = 50e3  # Pa
    Astim = 30.0  # mA/m2
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    PRF = 100.  # Hz
    DC = 1.0

    runPlotEStim(neuron, Astim, tstim, toffset, PRF, DC, verbose=verbose)
    runPlotAStim(neuron, a, fs, Fdrive, Adrive, tstim, toffset, PRF, DC, verbose=verbose)
    compareAStim(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC, verbose=verbose)

    if not hide:
        plt.show()


def main():

    # Define argument parser
    ap = ArgumentParser()
    ap.add_argument('-n', '--neuron', type=str, default='RS', help='Neuron type')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('--noplot', default=False, action='store_true',
                    help='Do not show figures')

    # Parse arguments
    args = ap.parse_args()

    # Run tests
    runTests(args.neuron, args.verbose, args.noplot)
    sys.exit(0)


if __name__ == '__main__':
    main()
