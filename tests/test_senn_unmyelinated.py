# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:38:32 2019

@author: Maria
"""

import numpy as np

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
        nnodes = 45
        rho_a = 54.7                         # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                        # axon / fiber diameter ratio
        #nodeL = 1.5e-6                       # node length (m)
        fiberL = 10e-3                       # fiber length (m) 
        #fiber = IinjUnmyelinatedSennFiber(pneuron, fiberD, nnodes, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)
        fiber = IinjUnmyelinatedSennFiber(pneuron, fiberD, nnodes, rs=rho_a, fiberL=fiberL, d_ratio=d_ratio)

        # Intracellular stimulation parameters
        tstim = 10e-6   # s
        toffset = 3e-3  # s
        PRF = 100.      # Hz
        DC = 1.         # -

        # Create extracellular current source
        psource = IntracellularCurrent(0, mode='anode')

        # Titrate for a specific duration and simulate fiber at threshold current
        #logger.info(f'Running titration for {si_format(tstim)}s pulse')
        #Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        data, meta = fiber.simulate(psource, 0.0000003, tstim, toffset, PRF, DC)

#        # Compare output metrics to reference
#        pulse_sim_metrics = {  # Output metrics
#            'Ithr': Ithr,                             # A
#            'cv': fiber.getConductionVelocity(data),  # m/s
#            'dV': fiber.getSpikeAmp(data)             # mV
#        }

        # Plot membrane potential traces for specific duration at threshold current
        fig1 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()
#
#        # Reset fiber model to avoid NEURON integration errors
#        fiber.reset()
#
#        # Compute and plot strength-duration curve for intracellular injection at central node
#        psource = IntracellularCurrent(nnodes // 2, mode='anode')
#        durations = np.array([10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500], dtype=float) * 1e-6  # s
#        Ithrs_sim = np.array([fiber.titrate(psource, x, toffset, PRF, DC) for x in durations])  # A
#
#        # Plot strength-duration curve
#        fig2 = strengthDurationCurve(fiber, durations, {'sim': Ithrs_sim}, Ifactor=1e9, scale='lin')
#
#        # Compare output metrics to reference
#        SDcurve_sim_metrics = {  # Output metrics
#            'chr': chronaxie(durations, Ithrs_sim)  # s
#        }
#
#        logger.info(f'Comparing metrics for {si_format(tstim)}s intracellular anodic pulse')
#        self.logOutputMetrics(pulse_sim_metrics)
#        logger.info(f'Comparing metrics for strength-duration curves')
#        self.logOutputMetrics(SDcurve_sim_metrics)

if __name__ == '__main__':
    tester = TestSennUnmyelinated()
    tester.main()