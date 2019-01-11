# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-30 10:51:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-12-27 18:15:38

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from ExSONIC._1D import SeriesConnector, runPlotAStim


def main():

    # Define argument parser
    ap = ArgumentParser()

    # Neuron type and sonophore radius
    ap.add_argument('--neuron', type=str, default='RS', help='Neuron type')
    ap.add_argument('-a', '--sonophore_radius', type=float, default=32., help='Sonophore radius (nm)')

    # ASTIM parameters
    ap.add_argument('-f', '--Fdrive', type=float, default=500., help='US frequency (kHz)')
    ap.add_argument('-A', '--Adrive', type=float, default=50., help='max US pressure amplitude (kPa)')
    ap.add_argument('--tstim', type=float, default=100., help='Stimulus duration (ms)')
    ap.add_argument('--toffset', type=float, default=100., help='Stimulus offset (ms)')
    ap.add_argument('--PRF', type=float, default=100., help='Pulse repetition frequency (Hz)')
    ap.add_argument('--DC', type=float, default=100., help='Stimulus dutry cycle (%)')

    # Number of nodes and axial resistivity
    ap.add_argument('--rs', type=float, default=100, help='Cytoplasmic resistivity (Ohm.cm)')
    ap.add_argument('-n', '--nnodes', type=int, default=2, help='Number of nodes')

    # Custom geometry
    ap.add_argument('--nodeD', type=float, nargs='+', default=[1.], help='Node diameters (um)')
    ap.add_argument('--nodeL', type=float, nargs='+', default=[1.], help='Node lengths (um)')
    ap.add_argument('--interD', type=float, nargs='+', default=[1.], help='Internode diameters (um)')
    ap.add_argument('--interL', type=float, nargs='+', default=[0.], help='Internode lengths (um)')

    # SENN geometry
    ap.add_argument('--senn', default=False, action='store_true', help='SENN geometry')
    ap.add_argument('-D', '--fiberD', type=float, default=20., help='Fiber diameter (um)')

    # Amplitude distribution
    ap.add_argument('--ampdist', type=str, default='firstonly',
                    help='Acoustic amplitude distribution')

    # Parse arguments
    args = ap.parse_args()
    neuron = getNeuronsDict()[args.neuron]()
    a = args.sonophore_radius  # nm
    Fdrive = args.Fdrive  # kHz
    Adrive = args.Adrive  # kPa
    tstim = args.tstim * 1e-3  # s
    toffset = args.toffset * 1e-3  # s
    PRF = args.PRF  # Hz
    DC = args.DC * 1e-2  # (-)
    rs = args.rs  # Ohm.cm
    nnodes = args.nnodes
    if args.senn:
        print('Using SENN geometry')
        nodeD = 0.7 * args.fiberD  # um
        nodeL = 2.5  # um
        interD = args.fiberD  # um
        interL = 100 * args.fiberD  # um
    else:
        print('Using custom geometry')
        nodeD = np.array(args.nodeD) if len(args.nodeD) > 1 else args.nodeD[0]  # um
        nodeL = np.array(args.nodeL) if len(args.nodeL) > 1 else args.nodeL[0]  # um
        interD = np.array(args.interD) if len(args.interD) > 1 else args.interD[0]  # um
        interL = np.array(args.interL) if len(args.interL) > 1 else args.interL[0]  # um
    amps = {
        'firstonly': np.hstack((np.array([1]), np.zeros(nnodes - 1))),
        'homogeneous': np.ones(nnodes),
        'gaussian': np.exp(-np.linspace(0, 1, nnodes) / (1e-1 * nnodes**2))
    }[args.ampdist] * Adrive  # kPa
    print('amplitude distribution: {} kPa'.format(amps))

    # Run and plot
    connector = SeriesConnector(vref='Vmeff_{}'.format(neuron.name))
    runPlotAStim(neuron, a, Fdrive, rs, connector, nodeD, nodeL, interD, interL, amps, tstim, toffset,
                 PRF, DC, nnodes=nnodes)
    plt.show()


if __name__ == '__main__':
    main()
