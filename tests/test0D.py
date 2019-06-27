# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-27 11:50:15
# @Author: Theo Lemaire
# @Date:   2018-08-30 11:26:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-01-18 11:43:33

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
    a = 32.  # sonophore diameter (nm)

    # Stimulation parameters
    tstim = 0.12e-3  # s
    toffset = 10e-3  # s

    # Parse test set
    alltests = ['Iintra', 'US']
    testset = alltests if args.testset == 'all' else [args.testset]
    for test in testset:
        assert test in alltests, '{} is not a valid test'.format(test)

    # Execute tests
    if 'Iintra' in testset:
        if neuron.name == 'FH':
            Iinj = 1e4  # mA/m2
            tstim = 0.12e-3  # s
            toffset = 3e-3  # s
        else:
            Iinj = 20.0  # mA/m2
            tstim = 100e-3  # s
            toffset = 50e-3  # s
        compare(neuron, Iinj, tstim, toffset, verbose=args.verbose)
    if 'US' in testset:
        Fdrive = 500.  # kHz
        if neuron.name == 'FH':
            Adrive = 500.  # kPa
        else:
            Adrive = 100.  # kPa
        tstim = 100e-3  # s
        toffset = 50e-3  # s
        compare(neuron, Adrive, tstim, toffset, DC=0.5, a=a, Fdrive=Fdrive, verbose=args.verbose)

    if not args.noplot:
        plt.show()
    sys.exit(0)


if __name__ == '__main__':
    main()
