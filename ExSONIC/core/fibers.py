# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-27 18:03:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-28 17:47:21

''' Constructor functions for different types of fibers. '''

import os
import numpy as np
import pandas as pd
import csv

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from PySONIC.postpro import boundDataFrame
from PySONIC.core import PulsedProtocol

from .senn import IintraFiber, IextraFiber, SonicFiber
from .sources import *
from ..batches import FiberConvergenceBatch, StrengthDurationBatch


def myelinatedFiber(fiber_class, pneuron, fiberD, nnodes, rs, nodeL, d_ratio, inter_ratio=100., **kwargs):
    ''' Create a single-cable myelinated fiber model.

        :param fiber_class: class of fiber to be instanced
        :param pneuron: point-neuron model object
        :param fiberD: fiber outer diameter (m)
        :param nnodes: number of nodes
        :param rs: axoplasmic resistivity (Ohm.cm)
        :param nodeL: nominal node length (m)
        :param d_ratio: ratio of axon (inner-myelin) and fiber (outer-myelin) diameters
        :param inter_ratio: ratio of internode length to fiber diameter (default = 100)
        :param kwargs: other keyword arguments
        :return: myelinated fiber object
    '''
    if fiberD <= 0:
        raise ValueError('fiber diameter must be positive')
    if nodeL <= 0.:
        raise ValueError('node length must be positive')
    if d_ratio > 1. or d_ratio < 0.:
        raise ValueError('fiber-axon diameter ratio must be in [0, 1]')
    if inter_ratio <= 0.:
        raise ValueError('fiber diameter - internode length ratio must be positive')

    # Define fiber geometrical parameters
    nodeD = d_ratio * fiberD       # m
    nodeL = nodeL                  # m
    interD = d_ratio * fiberD      # m
    interL = inter_ratio * fiberD  # m

    # Create fiber model instance
    return fiber_class(pneuron, nnodes, rs, nodeD, nodeL, interD, interL, **kwargs)


def unmyelinatedFiber(fiber_class, pneuron, fiberD, rs, fiberL=5e-3, maxNodeL=None, **kwargs):
    ''' Create a single-cable unmyelinated fiber model.

        :param fiber_class: class of fiber to be instanced
        :param pneuron: point-neuron model object
        :param fiberD: fiber outer diameter (m)
        :param rs: axoplasmic resistivity (Ohm.cm)
        :param fiberL: fiber length (m)
        :param maxNodeL: maximal node length (m)
        :param kwargs: other keyword arguments
        :return: unmyelinated fiber object
    '''
    if fiberD <= 0:
        raise ValueError('fiber diameter must be positive')
    if fiberL <= 0:
        raise ValueError('fiber length must be positive')
    if fiberL <= 3e-3:
        logger.warning('fiber length is below the convergence threshold')
    if maxNodeL is None:
        nodelength_lin = fiberD * 16.3 + 9.1e-6  #um
        maxNodeL = min(nodelength_lin, 22e-6)
    if maxNodeL <= 0. or maxNodeL > fiberL:
        raise ValueError('maximum node length must be in [0, fiberL]')

    # Compute number of nodes (ensuring odd number) and node length from fiber length
    nnodes = int(np.ceil(fiberL / maxNodeL))
    if nnodes % 2 == 0:
        nnodes += 1
    nodeL = fiberL / nnodes

    # Define other fiber geometrical parameters (with zero-length internode)
    nodeD = fiberD   # m
    interD = fiberD  # m
    interL = 0.      # m

    # Create fiber model instance
    fiber = fiber_class(pneuron, nnodes, rs, nodeD, nodeL, interD, interL, **kwargs)
    return fiber


def myelinatedFiberReilly(fiber_class, fiberD=20e-6, **kwargs):
    '''  Create typical myelinated fiber model, using parameters from Reilly 1985.

        :param fiber_class: class of fiber to be instanced
        :param fiberD: fiber diameter (m)
        :param kwargs: other keyword arguments
        :return: fiber myelinated object

        Reference: Reilly, J.P., Freeman, V.T., and Larkin, W.D. (1985). Sensory effects
        of transient electrical stimulation--evaluation with a neuroelectric model.
        IEEE Trans Biomed Eng 32, 1001–1011.
    '''
    pneuron = getPointNeuron('FH')  # Frog myelinated node membrane equations
    nnodes = 21                     # number of nodes
    rs = 110.0                      # axoplasm resistivity (Ohm.cm, from McNeal 1976)
    nodeL = 2.5e-6                  # node length (m, from McNeal 1976)
    d_ratio = 0.7                   # axon / fiber diameter ratio (from McNeal 1976)
    return myelinatedFiber(fiber_class, pneuron, fiberD, nnodes, rs, nodeL, d_ratio, **kwargs)

def myelinatedFiberSweeney(fiber_class, fiberD=10e-6, **kwargs):
    '''  Create typical myelinated fiber model, using parameters from Sweeney 1987.

        :param fiber_class: class of fiber to be instanced
        :param fiberD: fiber diameter (m)
        :param kwargs: other keyword arguments
        :return: fiber myelinated object

        Reference: Sweeney, J.D., Mortimer, J.T., and Durand, D. (1987). Modeling of
        mammalian myelinated nerve for functional neuromuscular stimulation. IEEE 9th
        Annual Conference of the Engineering in Medicine and Biology Society 3, 1577–1578.
    '''
    pneuron = getPointNeuron('SW')  # mammalian fiber membrane equations
    nnodes = 19                     # number of nodes
    rs = 54.7                       # axoplasm resistivity (Ohm.cm)
    nodeL = 1.5e-6                  # node length (m)
    d_ratio = 0.6                   # axon / fiber diameter ratio
    return myelinatedFiber(fiber_class, pneuron, fiberD, nnodes, rs, nodeL, d_ratio, **kwargs)


def unmyelinatedFiberSundt(fiber_class, fiberD=0.8e-6, fiberL = 5e-3, **kwargs):
    ''' Create typical unmyelinated fiber model, using parameters from Sundt 2015.

        :param fiber_class: class of fiber to be instanced
        :param fiberD: fiber diameter (m)
        :param kwargs: other keyword arguments
        :return: fiber myelinated object

        Reference: Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
        root ganglia in an unmyelinated sensory neuron: a modeling study.
        Journal of Neurophysiology (2015)
    '''
    pneuron = getPointNeuron('sundt')  # DRG peripheral axon membrane equations
    rs = 100.                          # axoplasm resistivity, from Sundt 2015 (Ohm.cm)
    return unmyelinatedFiber(fiber_class, pneuron, fiberD, rs, fiberL, **kwargs)


# Fiber factory functions
fiber_factories = {
    'reilly': myelinatedFiberReilly,
    'sweeney': myelinatedFiberSweeney,
    'sundt': unmyelinatedFiberSundt
}

def getFiberFactory(fiberType):
    ''' Get fiber factory. '''
    try:
        return fiber_factories[fiberType]
    except KeyError as err:
        raise ValueError(f'Unknown fiber type: "{fiberType}"')


def strengthDuration(fiberType, fiberClass, fiberD, tstim_range, toffset=20e-3, outdir='.',
                     zdistance=1e-3, Fdrive=500e3, a=32e-9, fs=1., r=2e-3, sigma=1e-3):

    # Default conversion function
    convert_func = lambda x: x

    # Get fiber factory
    fiber_factory = getFiberFactory(fiberType)

    logger.info(f'creating model with fiberD = {fiberD * 1e6:.2f} um ...')

    if fiberClass == 'intracellular_electrical_stim':
        fiber = fiber_factory(IintraFiber, fiberD=fiberD)
        source = IntracellularCurrent(fiber.nnodes // 2)
        out_key = 'Ithr (A)'

    elif fiberClass == 'extracellular_electrical_stim':
        fiber = fiber_factory(IextraFiber, fiberD=fiberD)
        source = ExtracellularCurrent((0, zdistance), mode='cathode')
        out_key = 'Ithr (A)'

    elif fiberClass == 'acoustic_single_node':
        fiber = fiber_factory(SonicFiber, fiberD=fiberD, a=a, fs=fs)
        source = NodeAcousticSource(fiber.nnodes//2, Fdrive)
        out_key = 'Athr (Pa)'

    elif fiberClass == 'acoustic_gaussian':
        fiber = fiber_factory(SonicFiber, fiberD=fiberD, a=a, fs=fs)
        source = GaussianAcousticSource(0., sigma, Fdrive)
        out_key = 'Athr (Pa)'

    elif fiberClass == 'acoustic_planar_transducer':
        fiber = fiber_factory(SonicFiber, fiberD=fiberD, a=a, fs=fs)
        source = PlanarDiskTransducerSource((0, 0, zdistance), Fdrive, r=r)
        convert_func = lambda x: x * source.relNormalAxisAmp(0.)  # Pa
        out_key = 'Athr (Pa)'

    else:
        raise ValueError(f'Unknown fiber class: {fiberClass}')

    # Create SD batch
    sd_batch = StrengthDurationBatch(
            out_key, source, fiber, tstim_range, toffset, root=outdir, convert_func=convert_func)

    # Run batch
    df = sd_batch.run()

    # Clear fiber model
    fiber.clear()

    # Return batch output
    return df


def currentDistance(fiberType, fiberD, tstim, n_cur, cur_min, cur_max, n_z, z_min, z_max, outdir='.'):

    # Get fiber
    fiber_factory = getFiberFactory(fiberType)
    fiber = fiber_factory(IextraFiber, fiberD=fiberD)

    psource = ExtracellularCurrent((0, z_min), mode='cathode')
    toffset = 20e-3
    pp = PulsedProtocol(tstim, toffset)

    # Get filecode
    filecodes = fiber.filecodes(psource, 1, pp)
    for k in ['nnodes', 'nodeD', 'rs', 'nodeL', 'interD', 'interL', 'I', 'A', 'inode', 'nature', 'toffset', 'PRF', 'DC']:
        if k in filecodes:
            del filecodes[k]
    filecodes['fiberD'] = f'fiberD{(fiberD * 1e6):.2f}um'
    filecodes['cur'] = f'cur{(n_cur):.0f}_{(cur_min*1e3):.2f}mA-{(cur_max*1e3):.2f}mA'
    filecodes['z'] = f'z{(n_z):.0f}_{(z_min*1e3):.2f}mm-{(z_max*1e3):.2f}mm'
    fcode = '_'.join(filecodes.values())

    # Output file
    fname = f'{fcode}_strengthduration_results.txt'
    fpath = os.path.join(outdir, fname)

    # Computation of the current distance matrix if the file does not exist
    if not os.path.isfile(fpath):
        currents = np.linspace(cur_min, cur_max, n_cur)
        zdistances = np.linspace(z_min, z_max, n_z)
        ExcitationMatrix = [[0 for i in range(n_cur)] for j in range(n_z)]
        for i, I in enumerate(currents):
            for j, z in enumerate(zdistances):
                if I > 0:
                    psource = ExtracellularCurrent((0, z), I, mode='anode')
                elif I < 0:
                    psource = ExtracellularCurrent((0, z), I, mode='cathode')
                data, meta= fiber.simulate(psource, pp)
                ExcitationMatrix[j][i] = fiber.isExcited(data)
        # Save results
        np.savetxt(fpath, ExcitationMatrix)

    # Load results
    ExcitationMatrix = np.loadtxt(fpath)

    return ExcitationMatrix

