# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 11:26:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-10 17:47:52

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from ExSONIC._0D import compare


def main():

    # Define argument parser
    ap = ArgumentParser()
    ap.add_argument('-n', '--neuron', type=str, default='RS', help='Neuron type')
    ap.add_argument('-t', '--testset', type=str, default='all', help='Test set')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('--noplot', default=False, action='store_true',
                    help='Do not show figures')

    # Parse arguments
    args = ap.parse_args()

    # Model parameters
    neuron = getNeuronsDict()[args.neuron]()
    verbose = args.verbose
    a = 32.  # sonophore diameter (nm)

    # Stimulation parameters
    Fdrive = 500.  # kHz
    Adrive = 50.  # kPa
    Astim = 1e4  # mA/m2
    tstim = 0.12e-3  # s
    toffset = 10e-3  # s
    PRF = 100.  # Hz
    DC = 1.

    tests = args.testset
    if tests == 'all':
        tests = ['ASTIM', 'ESTIM']
    else:
        tests = [tests]

    if 'ESTIM' in tests:
        compare(neuron, Astim, tstim, toffset, PRF, DC, verbose=verbose)
    if 'ASTIM' in tests:
        compare(neuron, Adrive, tstim, toffset, PRF, DC, a=a, Fdrive=Fdrive, verbose=verbose)


    if not args.noplot:
        plt.show()
    sys.exit(0)


if __name__ == '__main__':
    main()
