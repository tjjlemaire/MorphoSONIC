# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 11:26:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-05 12:21:43

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import *
from ExSONIC._0D import runPlotEStim, compareAStim



def runTests(hide):

    # Model parameters
    neuron = CorticalRS()
    a = 32e-9  # sonophore diameter (m)

    # Stimulation parameters
    Fdrive = 500e3  # Hz
    Adrive = 50e3  # Pa
    Astim = 30.0  # mA/m2
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    PRF = 100.  # Hz
    DC = 1.0

    runPlotEStim(neuron, Astim, tstim, toffset, PRF, DC)
    compareAStim(neuron, a, Fdrive, Adrive, tstim, toffset, PRF, DC)

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
