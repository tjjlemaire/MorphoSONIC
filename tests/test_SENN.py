# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-29 11:32:45

import numpy as np
import logging
import matplotlib.pyplot as plt

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from PySONIC.test import TestBase
from ExSONIC.core import VextSennFiber, CurrentPointSource
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve


class TestSenn(TestBase):

    def test_Reilly1985(self, is_profiled=False):
        ''' Run SENN fiber simulation with identical parameters as in Reilly 1985, Fig2.,
            and compare resulting threshold current, spike amplitude and conduction velocity.

            Reference: Reilly, J.P., Freeman, V.T., and Larkin, W.D. (1985). Sensory effects
            of transient electrical stimulation--evaluation with a neuroelectric model.
            IEEE Trans Biomed Eng 32, 1001–1011.
        '''

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # FrankenHaeuser-Huxley membrane equations
        fiberD = 20e-6                  # fiber diameter (m)
        nnodes = 11
        rho_a = 110.0                   # axoplasm resistivity (Ohm.cm, from McNeal 1976)
        d_ratio = 0.7                   # axon / fiber diameter ratio (from McNeal 1976)
        nodeL = 2.5e-6                  # node length (m, from McNeal 1976)
        fiber = VextSennFiber(pneuron, fiberD, nnodes, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # Electrode and stimulation parameters (from Reilly 1985)
        rho_e = 300.0      # resistivity of external medium (Ohm.cm, from McNeal 1976)
        z0 = fiber.interL  # point-electrode to fiber distance (m, 1 internode length)
        x0 = 0.            # point-electrode located above central node (m)
        mode = 'cathode'   # cathodic pulse
        tstim = 100e-6     # s
        toffset = 3e-3     # s
        PRF = 100.         # Hz
        DC = 1.            # -

        # Reference outputs (from Reilly 1985)
        Ithr_ref = -0.68e-3  # threshold cathodic excitation current amplitude (A)
        cv_ref = 43.0        # conduction velocity (m/s)
        dV_ref = (105, 115)  # spike amplitude (mV)

        # Create point-source electrode
        psource = CurrentPointSource(x0, z0, rho=rho_e, mode=mode)

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        data, meta = fiber.simulate(psource, Ithr, tstim, toffset, PRF, DC)

        # Compute conduction velocity and spike amplitude from resulting data
        cv = fiber.getConductionVelocity(data)  # m/s
        dVmin, dVmax = fiber.getSpikeAmp(data)  # mV

        # Log output metrics
        logger.info(f'threshold current: Ithr = {si_format(Ithr)}A (ref: {si_format(Ithr_ref)}A)')
        logger.info(f'conduction velocity: v = {cv:.1f} m/s (ref: {cv_ref:.1f} m/s)')
        logger.info(f'spike amplitude: dV = {dVmin:.1f}-{dVmax:.1f} mV (ref: {dV_ref[0]:.1f}-{dV_ref[1]:.1f} mV)')

        # Plot membrane potential traces for specific duration at threshold current
        fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

    def test_Sweeney1987(self, is_profiled=False):
        ''' Run SENN fiber simulation with identical parameters as in Sweeney 1987, Figs 2 & 3.,
            and compare resulting threshold current, spike amplitude and conduction velocity,
            strength-duration curves, rheobase current and chronaxie.

            Reference: Sweeney, J.D., Mortimer, J.T., and Durand, D. (1987). Modeling of
            mammalian myelinated nerve for functional neuromuscular stimulation. IEEE 9th
            Annual Conference of the Engineering in Medicine and Biology Society 3, 1577–1578.
        '''
        return

        # Fiber model parameters
        pneuron = getPointNeuron('sweeney')  # mammalian fiber membrane equations
        fiberD = 10e-6                       # fiber diameter (m)
        nnodes = 19
        rho_a = 54.7                         # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                        # axon / fiber diameter ratio
        nodeL = 1.5e-6                       # node length (m)
        fiber = IinjSennFiber(pneuron, fiberD, nnodes, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # Intracellular stimulation parameters
        tstim = 10e-6   # s
        toffset = 3e-3  # s
        PRF = 100.      # Hz
        DC = 1.         # -

        # Output metrics (from Sweeney 1987)
        cv_ref = 57.0       # conduction velocity (m/s)
        dV_ref = (95, 105)  # spike amplitude (mV)
        tau_ref = 25.9e-6   # chronaxie from SD curve (s)

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        data, meta = fiber.simulate(psource, Ithr, tstim, toffset, PRF, DC)

        # Compute conduction velocity and spike amplitude from resulting data
        cv = fiber.getConductionVelocity(data)  # m/s
        dVmin, dVmax = fiber.getSpikeAmp(data)  # mV

        # Log output metrics
        logger.info(f'threshold current: Ithr = {si_format(Ithr)}A')
        logger.info(f'conduction velocity: v = {cv:.1f} m/s (ref: {cv_ref:.1f} m/s)')
        logger.info(f'spike amplitude: dV = {dVmin:.1f}-{dVmax:.1f} mV (ref: {dV_ref[0]:.1f}-{dV_ref[1]:.1f} mV)')

        # Compute and plot strength-duration curve
        durations = np.logspace(-5, -1, 30)  # s
        Ithrs = np.array([fiber.titrate(psource, x, toffset, PRF, DC) for x in durations])  # A
        tau = psource.chronaxie(durations, Ithrs)  # s
        logger.info(f'strength-duration curve: chronaxie = {si_format(tau, 1)}s (ref = {si_format(tau_ref, 1)}s)')

        # Plot strength-duration curve
        fig = strengthDurationCurve(fiber, psource, durations, Ithrs)


if __name__ == '__main__':
    tester = TestSenn()
    tester.main()
    plt.show()