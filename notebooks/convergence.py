# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:38:32 2019

@author: Maria
"""

import os
import numpy as np

from PySONIC.utils import logger, si_format
from ExSONIC.core import VextUnmyelinatedSennFiber


def cache(fpath, delimiter=','):
    ''' Add an extra IO memoization functionality to a function using file caching,
        to avoid repetitions of tedious computations with identical inputs.
    '''
    def wrapper_with_args(func):
        def wrapper(*args, **kwargs):            
            if os.path.isfile(fpath):
                out = np.loadtxt(fpath, delimiter=delimiter)
            else:
                out = func(*args, **kwargs)
                np.savetxt(fpath, out, delimiter=delimiter)
            return out
        return wrapper
    return wrapper_with_args


@cache('Ithrs.txt')
def convergence(pneuron, fiberD, rho_a, d_ratio, fiberL, tstim, toffset, PRF, DC, psource, nnodes):
    Ithrs = np.empty(nnodes.size)
    for i, x in enumerate(nnodes):
        if x == 1:
            x = 3
        fiber = VextUnmyelinatedSennFiber(pneuron, fiberD, x, rs=rho_a, fiberL=fiberL, d_ratio=d_ratio)
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithrs[i] = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        fiber.reset()
    return Ithrs
