# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:46:57

import numpy as np

from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format
from MorphoSONIC.core import *
from MorphoSONIC.models import *
from MorphoSONIC.test import TestFiber
from MorphoSONIC.plt import SectionCompTimeSeries, strengthDurationCurve, plotFieldDistribution
from MorphoSONIC.utils import chronaxie


class TestFiberEstim(TestFiber):

    w = 1e-3          # Gaussian FWHM (m)
    thr_factor = 1.2  # threshold factor at which to display results

    def test_Reilly1985(self, is_profiled=False):
        ''' Run SENN fiber simulation with identical parameters as in Reilly 1985, Fig2.,
           and compare resulting metrics:
               - threshold current for 100 us cathodic pulse
               - conduction velocity
               - spike amplitude
               - rheobase cathodic charge
               - rheobase cathodic current
               - Polarity selectivity ratio P for 1 us and 10 ms cathodic pulses
               - S/D time constant

           Reference: Reilly, J.P., Freeman, V.T., and Larkin, W.D. (1985). Sensory effects
           of transient electrical stimulation--evaluation with a neuroelectric model.
           IEEE Trans Biomed Eng 32, 1001–1011.
        '''
        logger.info('Test: SENN myelinated fiber model validation against Reilly 1985 data')

        # Fiber model (20 um diameter)
        fiberD = 20e-6  # m
        nnodes = 21
        fiber = SennFiber(fiberD, nnodes)

        # Electrode and stimulation parameters (from Reilly 1985)
        rho_e = 300.0      # resistivity of external medium (Ohm.cm, from McNeal 1976)
        z0 = fiber.interL  # point-electrode to fiber distance (m, 1 internode length)
        x0 = 0.            # point-electrode located above central node (m)
        mode = 'cathode'   # cathodic pulse
        tstim = 100e-6     # s
        toffset = 3e-3     # s
        pp = PulsedProtocol(tstim, toffset)

        # Create extracellular current source
        psource = ExtracellularCurrent((x0, z0), rho=rho_e, mode=mode)

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, pp)  # A
        data, meta = fiber.simulate(psource.updatedX(Ithr), pp)

        # Compare output metrics to reference
        pulse_sim_metrics = {  # Output metrics
           'Ithr': Ithr,                             # A
           'cv': fiber.getConductionVelocity(data),  # m/s
           'dV': fiber.getSpikeAmp(data)             # mV
        }
        pulse_ref_metrics = {   # Reference metrics (from Reilly 1985, fig 2)
           'Ithr': -0.68e-3,  # threshold cathodic excitation current amplitude (A)
           'cv': 43.0,        # conduction velocity (m/s)
           'dV': (105, 120)   # spike amplitude (mV)
        }

        # Plot membrane potential traces for specific duration at threshold current
        SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

        # Compute and plot strength-duration curve with both polarities
        fiber.reset()
        durations = np.logspace(0, 4, 5) / S_TO_US  # s
        pps = [PulsedProtocol(x, toffset) for x in durations]
        psources = {k: ExtracellularCurrent((x0, z0), rho=rho_e, mode=k)
                    for k in ['cathode', 'anode']}
        Ithrs = {k: np.array([np.abs(fiber.titrate(v, x)) for x in pps])  # A
                 for k, v in psources.items()}
        I0_ref = 0.36e-3  # A
        Ithrs['cathode ref'] = np.array([96.3, 9.44, 1.89, 1.00, 1.00]) * I0_ref  # A

        # Plot strength-duration curve
        strengthDurationCurve(fiber, durations, Ithrs, yfactor=1 / MA_TO_A, scale='log')

        # Compare output metrics to reference
        SDcurve_sim_metrics = {  # Output metrics
           'Q0': durations[0] * Ithrs['cathode'][0],
           'I0': Ithrs['cathode'][-1],
           'P1us': Ithrs['anode'][0] / Ithrs['cathode'][0],
           'P10ms': Ithrs['anode'][-1] / Ithrs['cathode'][-1]
        }
        SDcurve_ref_metrics = {  # Reference metrics (from Reilly 1985, fig 6 and table 1)
           'Q0': 34.7e-9,  # rheobase cathodic charge Q0 (C)
           'I0': I0_ref,   # rheobase cathodic current (A)
           'P1us': 4.2,    # polarity selectivity ratio P for 1 us pulse
           'P10ms': 5.6    # polarity selectivity ratio P for 10 ms pulse
        }

        # Log metrics
        logger.info(f'Comparing metrics for {si_format(tstim)}s extracellular cathodic pulse')
        self.logOutputMetrics(pulse_sim_metrics, pulse_ref_metrics)
        logger.info(f'Comparing metrics for strength-duration curves')
        self.logOutputMetrics(SDcurve_sim_metrics, SDcurve_ref_metrics)

    def test_Reilly1987(self, is_profiled=False):
        ''' Run SENN fiber simulation with identical parameters as in Reilly 1987 (base-case),
           and compare resulting output indexes:
               - rheobase cathodic charge
               - rheobase cahtodic current
               - Polarity selectivity ratio P for 10 us and 1 ms cathodic pulses
               - S/D time constant

           Reference: Reilly, J.P., and Bauer, R.H. (1987). Application of a neuroelectric model
           to electrocutaneous sensory sensitivity: parameter variation study. IEEE Trans Biomed
           Eng 34, 752–754.
        '''
        logger.info('Test: SENN myelinated fiber model validation against Reilly 1987 data')

        # Fiber model (10 um diameter)
        fiberD = 10e-6  # m
        nnodes = 21
        fiber = SennFiber(fiberD, nnodes)

        # Electrode and stimulation parameters (from Reilly 1987)
        rho_e = 300.0      # resistivity of external medium (Ohm.cm, from McNeal 1976)
        z0 = fiber.interL  # point-electrode to fiber distance (m, 1 internode length)
        x0 = 0.            # point-electrode located above central node (m)
        toffset = 3e-3     # s

        # Create cathodic and anodic extracellular current sources
        psources = {k: ExtracellularCurrent((x0, z0), rho=rho_e, mode=k)
                    for k in ['cathode', 'anode']}

        # Compute and plot strength-duration curve with both polarities
        durations = np.array([1, 5, 10, 50, 100, 500, 1000, 2000], dtype=float) / S_TO_US  # s
        Ithrs = {k: np.array([np.abs(fiber.titrate(v, PulsedProtocol(x, toffset))) for x in durations])  # A
                 for k, v in psources.items()}
        Qthrs = {k: v * durations for k, v in Ithrs.items()}  # C

        # Plot strength-duration curve
        strengthDurationCurve(fiber, durations, Ithrs, yfactor=1 / MA_TO_A, scale='log')

        # Compare output metrics to reference
        i10us, i1ms = 2, 6
        SDcurve_sim_metrics = {  # Output metrics
           'Q0': Qthrs['cathode'].min(),  # C
           'I0': Ithrs['cathode'].min(),  # A
           'P10us': Ithrs['anode'][i10us] / Ithrs['cathode'][i10us],
           'P1ms': Ithrs['anode'][i1ms] / Ithrs['cathode'][i1ms]
        }
        SDcurve_sim_metrics['tau_e'] = SDcurve_sim_metrics['Q0'] / SDcurve_sim_metrics['I0']  # s
        SDcurve_ref_metrics = {  # Reference outputs (from Reilly 1987, table 2)
           'Q0': 15.9e-9,     # rheobase cathodic charge Q0 (C)
           'I0': 0.18e-3,     # rheobase cathodic current (A)
           'P10us': 4.66,     # polarity selectivity ratio P for 10 us
           'P1ms': 5.53,      # polarity selectivity ratio P for 10 us
           'tau_e': 92.3e-6,  # S/D time constant tau_e (s)
        }
        logger.info(f'Comparing metrics for strength-duration curves')
        self.logOutputMetrics(SDcurve_sim_metrics, SDcurve_ref_metrics)

    def test_Sweeney1987(self, is_profiled=False):
        ''' Run SENN fiber simulation with identical parameters as in Sweeney 1987, Figs 2 & 3.,
            and compare resulting threshold current, spike amplitude and conduction velocity,
            strength-duration curves, rheobase current and chronaxie.

            Reference: Sweeney, J.D., Mortimer, J.T., and Durand, D. (1987). Modeling of
            mammalian myelinated nerve for functional neuromuscular stimulation. IEEE 9th
            Annual Conference of the Engineering in Medicine and Biology Society 3, 1577–1578.
        '''
        logger.info('Test: SENN myelinated fiber model validation against Sweeney 1987 data')

        # Fiber model (10 um diameter)
        fiberD = 10e-6  # m
        nnodes = 19
        fiber = SweeneyFiber(fiberD, nnodes)

        # Intracellular stimulation parameters
        tstim = 10e-6   # s
        toffset = 3e-3  # s
        pp = PulsedProtocol(tstim, toffset)

        # Create extracellular current source
        psource = IntracellularCurrent('node0', mode='anode')

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, pp)  # A
        data, meta = fiber.simulate(psource.updatedX(self.thr_factor * Ithr), pp)

        # Compare output metrics to reference
        pulse_sim_metrics = {  # Output metrics
            'Ithr': Ithr,                             # A
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }
        pulse_ref_metrics = {  # Reference metrics (from Sweeney 1987)
            'cv': 57.0,       # conduction velocity (m/s)
            'dV': (90, 95),  # spike amplitude (mV)
        }

        # Plot membrane potential traces for specific duration at threshold current
        SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

        # Reset fiber model to avoid NEURON integration errors
        fiber.reset()

        # Compute and plot strength-duration curve for intracellular injection at central node
        psource = IntracellularCurrent(fiber.central_ID, mode='anode')
        durations = np.array([
            10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500], dtype=float) / S_TO_US  # s
        Ithrs_ref = np.array([
            4.4, 2.8, 1.95, 1.6, 1.45, 1.4, 1.30, 1.25, 1.25, 1.25, 1.25, 1.25]) / A_TO_NA  # A
        Ithrs_sim = np.array([
            fiber.titrate(psource, PulsedProtocol(x, toffset)) for x in durations])  # A
        Qthrs_sim = Ithrs_sim * durations  # C

        # Plot strength-duration curve
        strengthDurationCurve(
            fiber, durations, {'ref': Ithrs_ref, 'sim': Ithrs_sim}, yfactor=A_TO_NA, scale='lin')

        # Compare output metrics to reference
        SDcurve_sim_metrics = {  # Output metrics
            'chr': chronaxie(durations, Ithrs_sim),  # s
            'tau_e': Qthrs_sim.min() / Ithrs_sim.min()  # s
        }
        SDcurve_ref_metrics = {  # Reference metrics (from Sweeney 1987)
            'chr': 25.9e-6,   # chronaxie from S/D curve (s)
            'tau_e': 37.4e-6  # S/D time constant (s)
        }

        logger.info(f'Comparing metrics for {si_format(tstim)}s intracellular anodic pulse')
        self.logOutputMetrics(pulse_sim_metrics, pulse_ref_metrics)
        logger.info(f'Comparing metrics for strength-duration curves')
        self.logOutputMetrics(SDcurve_sim_metrics, SDcurve_ref_metrics)

    def test_Sundt2015(self, is_profiled=False):
        ''' Run C-fiber simulation with identical parameters as in Sundt 2015, Figs 5d
            (only peripheral axon), and compare resulting conduction velocity and spike
            amplitude.

            Reference: Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
            root ganglia in an unmyelinated sensory neuron: a modeling study.
            Journal of Neurophysiology (2015)
        '''
        logger.info('Test: SENN unmyelinated fiber model validation against Sundt 2015 data')

        # Unmyelinated fiber model (0.8 um diameter)
        fiberD = 0.8e-6  # m
        fiber = UnmyelinatedFiber(fiberD, fiberL=5e-3)

        # Stimulation parameters
        tstim = 1e-3     # s
        toffset = 10e-3  # s
        pp = PulsedProtocol(tstim, toffset)
        psource = IntracellularCurrent(fiber.central_ID)
        I = 0.2e-9  # A

        # Simulate fiber
        data, meta = fiber.simulate(psource.updatedX(I), pp)

        # Discard data from end nodes
        # npad = 2
        # ids = fiber.nodeIDs[npad:-npad]
        # data = {k: data[k] for k in ids}

        # Assess excitation
        logger.info('fiber is {}excited'.format({True: '', False: 'not '}[fiber.isExcited(data)]))

        # Plot membrane potential traces for specific duration at threshold current
        SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

        # Compare output metrics to reference
        pulse_ref_metrics = {     # Reference metrics (from Sundt 2015 ModelDB files)
            'cv': 0.44,     # conduction velocity (m/s)
            'dV': (86, 90)  # spike amplitude (mV)
        }
        pulse_sim_metrics = {
            'cv': fiber.getConductionVelocity(data, out='median'),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Reset fiber model to avoid NEURON integration errors
        fiber.reset()

        # Compute and plot strength-duration curve for intracellular injection at central node
        durations = np.logspace(-5, 0, 10)  # s
        Ithrs_sim = np.array([fiber.titrate(psource, PulsedProtocol(x, toffset))
                              for x in durations])  # A
        Qthrs_sim = Ithrs_sim * durations  # C

        # Compute SD curves metrics
        SDcurve_sim_metrics = {
            'chr': chronaxie(durations, Ithrs_sim),  # s
            'tau_e': Qthrs_sim.min() / Ithrs_sim.min()  # s
        }

        # Plot strength-duration curve
        strengthDurationCurve(fiber, durations, {'sim': Ithrs_sim}, yfactor=A_TO_NA, scale='log')

        logger.info(f'Comparing metrics for {si_format(pp.tstim)}s intracellular anodic pulse')
        self.logOutputMetrics(pulse_sim_metrics, pulse_ref_metrics)
        logger.info(f'Computing strength-duration curve metrics')
        self.logOutputMetrics(SDcurve_sim_metrics)

    def gaussian(self, fiber, pp):
        ''' Run myelinated fiber ESTIM simulation with gaussian distribution source. '''
        source = GaussianVoltageSource(
            0., GaussianAcousticSource.from_FWHM(self.w),
            mode='cathode')
        logger.info(f'fiber length = {fiber.length * 1e3:.2f} mm, source FWHM = {self.w * 1e3:.2f} mm')

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        # Vthr = fiber.titrate(source, pp)  # Pa
        Vthr = -100.0 / self.thr_factor
        data, meta = fiber.simulate(source.updatedX(self.thr_factor * Vthr), pp)
        plotFieldDistribution(fiber, source.updatedX(self.thr_factor * Vthr))

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
           'Vthr': Vthr,                             # Pa
           'cv': fiber.getConductionVelocity(data),  # m/s
           'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Plot membrane potential and membrane charge density traces
        SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

        # # Comparative SD curve
        # durations = np.logspace(-5, -3, 20)  # s
        # toffset = 10e-3                     # s
        # pps = [PulsedProtocol(t, toffset) for t in durations]
        # Athrs = np.array([fiber.titrate(source, pp) for pp in pps])

        # # Plot strength-duration curve
        # fig2 = strengthDurationCurve(
        #     fiber, durations, {'myelinated': Athrs}, scale='log',
        #     yname='amplitude', yfactor=PA_TO_KPA, yunit='Pa', plot_chr=False)

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

    def test_gaussianSENN(self, is_profiled=False):
        logger.info('Test: gaussian distribution source on myelinated SENN fiber')
        fiber = SennFiber(10e-6, 21)
        toffset = fiber.AP_travel_time_estimate * 3.0
        pp = PulsedProtocol(100e-6, toffset)
        return self.gaussian(fiber, pp)

    def test_gaussianMRG(self, is_profiled=False):
        logger.info('Test: gaussian distribution source on myelinated MRG fiber')
        fiber = MRGFiber(10e-6, 21)
        toffset = fiber.AP_travel_time_estimate * 10.0
        pp = PulsedProtocol(100e-6, toffset)
        return self.gaussian(fiber, pp)

    def test_gaussianSundt(self, is_profiled=False):
        logger.info('Test: gaussian distribution source on unmyelinated fiber')
        fiber = UnmyelinatedFiber(0.8e-6, fiberL=5e-3)
        toffset = fiber.AP_travel_time_estimate * 1.5
        pp = PulsedProtocol(10e-3, toffset)
        return self.gaussian(fiber, pp)


if __name__ == '__main__':
    tester = TestFiberEstim()
    tester.main()
