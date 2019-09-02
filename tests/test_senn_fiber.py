# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-02 15:50:38

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from PySONIC.test import TestBase
from ExSONIC.core import VextSennFiber, IinjSennFiber, ExtracellularCurrent, IntracellularCurrent, AStimSennFiber, NodeAcousticSource
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve
from ExSONIC.utils import chronaxie


class TestSenn(TestBase):

    @staticmethod
    def logOutputMetrics(sim_metrics, ref_metrics=None):
        ''' Log output metrics. '''
        if 'Ithr' in sim_metrics:
            thr_log = f'threshold current: Ithr = {si_format(sim_metrics["Ithr"], 1)}A'
        elif 'Athr' in sim_metrics:
            thr_log = f'threshold US amplitude: Athr = {si_format(sim_metrics["Athr"], 2)}Pa'
        logs = {
            'Ithr': thr_log,
            'cv': f'conduction velocity: v = {sim_metrics["cv"]:.1f} m/s',
            'dV': 'spike amplitude: dV = {:.1f}-{:.1f} mV'.format(*sim_metrics["dV"])
        }
        if ref_metrics is not None:
            if 'Ithr' in ref_metrics:
                logs['Ithr'] += f' (ref: {si_format(ref_metrics["Ithr"], 1)}A)'
            if 'Athr' in ref_metrics:
                logs['Athr'] += f' (ref: {si_format(ref_metrics["Ithr"], 2)}Pa)'
            if 'cv' in ref_metrics:
                logs['cv'] += f' (ref: {ref_metrics["cv"]:.1f} m/s)'
            if 'dV' in ref_metrics:
                logs['dV'] += ' (ref: {:.1f}-{:.1f} mV)'.format(*ref_metrics["dV"])
        for log in logs.values():
            logger.info(log)

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
        ref_metrics = {
            'Ithr': -0.68e-3,  # threshold cathodic excitation current amplitude (A)
            'cv': 43.0,        # conduction velocity (m/s)
            'dV': (105, 115)   # spike amplitude (mV)
        }

        # Create extracellular current source
        psource = ExtracellularCurrent(x0, z0, rho=rho_e, mode=mode)

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        data, meta = fiber.simulate(psource, Ithr, tstim, toffset, PRF, DC)

        # Compute conduction velocity and spike amplitude from resulting data
        cv = fiber.getConductionVelocity(data)  # m/s
        dVmin, dVmax = fiber.getSpikeAmp(data)  # mV
        sim_metrics = {
            'Ithr': Ithr,                             # A
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Log output metrics
        self.logOutputMetrics(sim_metrics, ref_metrics)

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

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # mammalian fiber membrane equations
        fiberD = 10e-6                       # fiber diameter (m)
        nnodes = 19
        rho_a = 54.7                         # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                        # axon / fiber diameter ratio
        nodeL = 1.5e-6                       # node length (m)
        fiber = IinjSennFiber(pneuron, fiberD, nnodes, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # Intracellular stimulation parameters
        tstim = 20e-6   # s
        toffset = 3e-3  # s
        PRF = 100.      # Hz
        DC = 1.         # -

        # Output metrics (from Sweeney 1987)
        ref_metrics = {
            'Ithr': 4.5e-9,   # threshold current (A)
            'cv': 57.0,       # conduction velocity (m/s)
            'dV': (95, 105),  # spike amplitude (mV)
            'tau': 25.9e-6    # chronaxie from SD curve (s)
        }

        # Create extracellular current source
        psource = IntracellularCurrent(nnodes // 2, mode='anode')

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        data, meta = fiber.simulate(psource, Ithr, tstim, toffset, PRF, DC)

        # Compute conduction velocity and spike amplitude from resulting data
        cv = fiber.getConductionVelocity(data)  # m/s
        dVmin, dVmax = fiber.getSpikeAmp(data)  # mV
        sim_metrics = {
            'Ithr': Ithr,                             # A
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Log output metrics
        self.logOutputMetrics(sim_metrics, ref_metrics)

        # Plot membrane potential traces for specific duration at threshold current
        fig1 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Reset fiber model to avoid NEURON integration errors
        fiber.reset()

        # Compute and plot strength-duration curve
        durations = np.array([10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500], dtype=float) * 1e-6  # s
        Ithrs_ref = np.array([4.4, 2.8, 1.95, 1.6, 1.45, 1.35, 1.25, 1.2, 1.2, 1.2, 1.2, 1.2]) * 1e-9  # A
        Ithrs_sim = np.array([fiber.titrate(psource, x, toffset, PRF, DC) for x in durations])  # A
        tau = chronaxie(durations, Ithrs_sim)  # s
        logger.info(f'strength-duration curve: chronaxie = {si_format(tau, 1)}s (ref = {si_format(ref_metrics["tau"], 1)}s)')
        Ithrs = {
            'ref': Ithrs_ref,
            'sim': Ithrs_sim
        }

        # Plot strength-duration curve
        fig2 = strengthDurationCurve(fiber, durations, Ithrs, Ifactor=1e9, scale='lin')

    def test_ASTIM(self, is_profiled=False):
        ''' Run SENN fiber ASTIM simulation. '''

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # mammalian fiber membrane equations
        fiberD = 10e-6                  # fiber diameter (m)
        nnodes = 5
        rho_a = 54.7                    # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                   # axon / fiber diameter ratio
        nodeL = 1.5e-6                  # node length (m)
        a = 32e-9                       # sonophore diameter (m)
        Fdrive = 500e3                  # US frequency (Hz)
        fs = 1                          # sonophore membrane coverage (-)
        fiber = AStimSennFiber(
            pneuron, fiberD, nnodes, a=a, Fdrive=Fdrive, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # US stimulation parameters
        Astim = 100e3   # Pa
        tstim = 200e-6   # s
        toffset = 3e-3  # s
        PRF = 100.      # Hz
        DC = 1.         # -

        # Create extracellular current source
        psource = NodeAcousticSource(nnodes // 2, Fdrive)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Athr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # Pa
        data, meta = fiber.simulate(psource, Athr, tstim, toffset, PRF, DC)

        # Compute conduction velocity and spike amplitude from resulting data
        cv = fiber.getConductionVelocity(data)  # m/s
        dVmin, dVmax = fiber.getSpikeAmp(data)  # mV
        sim_metrics = {
            'Athr': Athr,                             # Pa
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

        # Plot membrane potential traces for specific duration at threshold current
        fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()


if __name__ == '__main__':
    tester = TestSenn()
    tester.main()