# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:38:32 2019

@author: Maria
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from ExSONIC.core import IinjUnmyelinatedSennFiber, IntracellularCurrent
from ExSONIC.test import TestFiber
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve
from ExSONIC.utils import chronaxie


class TestSennUnmyelinated(TestFiber):

    def test_unmyelinated(self, is_profiled=False):
        ''' Run SENN unmyelinated fiber simulation.
        '''
        logger.info('Test: SENN unmyelinated model')

        # Fiber model parameters
        pneuron = getPointNeuron('FH')       # mammalian fiber membrane equations
        fiberD = 10e-6                       # fiber diameter (m)
        rho_a = 54.7                         # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                        # axon / fiber diameter ratio
        fiberL = 1e-2                       # fiber length (m) 

        # Intracellular stimulation parameters
        tstim = 10e-6   # s
        toffset = 3e-2  # s
        PRF = 100.      # Hz
        DC = 1.         # -

#        # Create extracellular current source
        psource = IntracellularCurrent(0, mode='anode')

        
        nnodes = np.logspace(0, 4, 100)
        nnodes = np.asarray(np.ceil(nnodes) // 2 * 2 + 1, dtype=int)
        Ithrs = np.empty(nnodes.size)
        print(nnodes)
        fname = 'Ithrs.txt'
        if os.path.isfile(fname):
            Ithrs = np.loadtxt(fname, delimiter=',')
        else:
            for i, x in enumerate(nnodes):
                if x == 1:
                    x = 3
                fiber = IinjUnmyelinatedSennFiber(pneuron, fiberD, x, rs=rho_a, fiberL=fiberL, d_ratio=d_ratio)
                logger.info(f'Running titration for {si_format(tstim)}s pulse')
                Ithrs[i] = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
            np.savetxt('Ithrs.txt', Ithrs, delimiter=',')
        
        nodeL = fiberL / nnodes
        Ithr_ref = Ithrs[-1]
        rel_errors = (Ithrs - Ithr_ref) / Ithr_ref
        max_rel_error = 0.01
        max_nodeL = np.interp(max_rel_error, rel_errors[::-1], nodeL[::-1], left=np.nan, right=np.nan)
        
        plt.plot(nodeL, rel_errors, label='convergence')
        plt.xlabel('node length')
        plt.ylabel('relative error')
        plt.xscale('log')
        plt.yscale('log')
        plt.axhline(max_rel_error, label='error threshold', color='k', linestyle='--')
        plt.axvline(max_nodeL, label=f'max node length ({max_nodeL * 1e6:.2f} um)', color='k')
        plt.legend()

if __name__ == '__main__':
    tester = TestSennUnmyelinated()
    tester.main()