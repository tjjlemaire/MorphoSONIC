# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-27 18:03:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-05 17:22:27

''' Constructor functions for different types of fibers. '''

import os
import numpy as np

from PySONIC.utils import logger
from PySONIC.core import PulsedProtocol

from .sources import *
from ..batches import StrengthDurationBatch
from .senn import SennFiber, SweeneyFiber, UnmyelinatedFiber


# --------------------- FACTORY FUNCTIONS ---------------------


def myelinatedFiberReilly(fiberD=20e-6, **kwargs):
    ''' Create typical myelinated fiber model, using parameters from Reilly 1985. '''
    return SennFiber(fiberD, 21, **kwargs)


def myelinatedFiberSweeney(fiberD=10e-6, **kwargs):
    ''' Create typical myelinated fiber model, using parameters from Sweeney 1987. '''
    return SweeneyFiber(fiberD, 19, **kwargs)


def unmyelinatedFiberSundt(fiberD=0.8e-6, fiberL=5e-3, **kwargs):
    ''' Create typical unmyelinated fiber model, using parameters from Sundt 2015. '''
    return UnmyelinatedFiber(fiberD, fiberL, **kwargs)


# Fiber classes
fiber_classes = {
    'reilly': SennFiber,
    'sweeney': SweeneyFiber,
    'sundt': UnmyelinatedFiber
}


def getFiberClass(key):
    ''' Get fiber class. '''
    try:
        return fiber_classes[key]
    except KeyError:
        raise ValueError(f'Unknown fiber type: "{key}"')


def strengthDuration(fiber_type, stim_type, fiberD, tstim_range, toffset=20e-3, outdir='.',
                     zdistance='focus', Fdrive=500e3, a=32e-9, fs=1., r=2e-3, sigma=1e-3):

    # Default conversion function
    convert_func = lambda x: x

    # Instanciate fiber model
    logger.info(f'creating model with fiberD = {fiberD * M_TO_UM:.2f} um ...')
    fiber = getFiberClass(fiber_type)(fiberD, 21)

    # Adapt out key and fiber model depending on stimulus modality
    if 'electrical' in stim_type:
        out_key = 'Ithr (A)'
    elif 'acoustic' in stim_type:
        out_key = 'Athr (Pa)'
        fiber.a = a
        fiber.fs = fs

    # Create appropriate source
    if stim_type == 'intracellular_electrical_stim':
        source = IntracellularCurrent(fiber.central_ID)
    elif stim_type == 'extracellular_electrical_stim':
        source = ExtracellularCurrent((0, zdistance), mode='cathode')
    elif stim_type == 'acoustic_single_node':
        source = SectionAcousticSource(fiber.central_ID, Fdrive)
    elif stim_type == 'acoustic_gaussian':
        source = GaussianAcousticSource(0., sigma, Fdrive)
    elif stim_type == 'acoustic_planar_transducer':
        source = PlanarDiskTransducerSource((0, 0, zdistance), Fdrive, r=r)
        convert_func = lambda x: x * source.relNormalAxisAmp(0.)  # Pa

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


def currentDistance(fiber_type, fiberD, tstim, n_cur, cur_min, cur_max, n_z, z_min, z_max, outdir='.'):

    # Get fiber
    fiber = getFiberClass(fiber_type)(fiberD, 21)

    psource = ExtracellularCurrent((0, z_min), mode='cathode')
    toffset = 20e-3
    pp = PulsedProtocol(tstim, toffset)

    # Get filecode
    filecodes = fiber.filecodes(psource, 1, pp)
    for k in ['nnodes', 'nodeD', 'rs', 'nodeL', 'interD', 'interL', 'I', 'A',
              'nature', 'toffset', 'PRF', 'DC']:
        if k in filecodes:
            del filecodes[k]
    filecodes['fiberD'] = f'fiberD{(fiberD * M_TO_UM):.2f}um'
    filecodes['cur'] = f'cur{(n_cur):.0f}_{(cur_min / MA_TO_A):.2f}mA-{(cur_max / MA_TO_A):.2f}mA'
    filecodes['z'] = f'z{(n_z):.0f}_{(z_min * M_TO_MM):.2f}mm-{(z_max * M_TO_MM):.2f}mm'
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
                data, meta = fiber.simulate(psource, pp)
                ExcitationMatrix[j][i] = fiber.isExcited(data)
        # Save results
        np.savetxt(fpath, ExcitationMatrix)

    # Load results
    ExcitationMatrix = np.loadtxt(fpath)

    return ExcitationMatrix
