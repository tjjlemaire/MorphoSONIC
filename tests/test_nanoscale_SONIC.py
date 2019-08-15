# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-15 17:43:01
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-04 19:29:58

import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.parsers import TestParser
from PySONIC.neurons import getPointNeuron
# from ExSONIC.utils import radialGeometry
from ExSONIC._1D import SeriesConnector, runPlotAStim


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
    neuron = getPointNeuron(args.neuron)
    rs = 1e2  # Ohm.cm
    config = 'first'

    # Set zero-length internodes
    interD = 1.  # um
    interL = 0.  # um

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
