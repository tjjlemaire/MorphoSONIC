# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-20 11:51:53

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format, isIterable
from PySONIC.test import TestBase
from ExSONIC.core import VextSennFiber, IinjSennFiber, ExtracellularCurrent, IntracellularCurrent, SonicSennFiber, NodeAcousticSource
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve
from ExSONIC.utils import chronaxie


class TestSenn(TestBase):

    @staticmethod
    def relativeChange(x, xref):
        if isIterable(x):
            x, xref = np.asarray(x), np.asarray(xref)
        return np.mean((x - xref) / xref)

    @classmethod
    def logOutputMetrics(cls, sim_metrics, ref_metrics=None):
        ''' Log output metrics. '''
        logfmts = {
            'Q0': {
                'name': 'rheobase charge',
                'fmt': lambda x: f'{si_format(np.abs(x), 1)}C'
            },
            'I0': {
                'name': 'rheobase current',
                'fmt': lambda x: f'{si_format(np.abs(x), 1)}A'
            },
            'Ithr': {
                'name': 'threshold current',
                'fmt': lambda x: f'{si_format(np.abs(x), 1)}A'
            },
            'cv': {
                'name': 'conduction velocity',
                'fmt': lambda x: f'{x:.1f} m/s'
            },
            'dV': {
                'name': 'spike amplitude',
                'fmt': lambda x: '{:.1f}-{:.1f} mV'.format(*x)
            },
            'Athr': {
                'name': 'threshold US amplitude',
                'fmt': lambda x: f'{si_format(x, 2)}Pa'
            },
            'chr': {
                'name': 'chronaxie',
                'fmt': lambda x: f'{si_format(x, 1)}s'
            },
            'tau_e': {
                'name': 'S/D time constant',
                'fmt': lambda x: f'{si_format(x, 1)}s'
            }
        }
        for t in [1e0, 1e1, 1e3, 1e4]:
            key = f'P{si_format(t * 1e-6, 0, space="")}s'
            desc = f'{si_format(t * 1e-6, 0)}s'
            logfmts[key] = {
                'name': f'polarity selectivity ratio @ {desc}',
                'fmt': lambda x: f'{x:.2f}'
            }

        for k, v in logfmts.items():
            warn = False
            if k in sim_metrics:
                log = f"--- {v['name']}: {k} = {v['fmt'](sim_metrics[k])}"
                if ref_metrics is not None and k in ref_metrics:
                    rel_change = cls.relativeChange(sim_metrics[k], ref_metrics[k])
                    if np.abs(rel_change) > 0.05:
                        warn = True
                    log += f" (ref = {v['fmt'](ref_metrics[k])}, {rel_change * 100:.2f}% change)"
                if warn:
                    logger.warning(log)
                else:
                    logger.info(log)

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
        logger.info('Test: SENN model validation against Reilly 1985 data')

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # FrankenHaeuser-Huxley membrane equations
        fiberD = 20e-6                  # fiber diameter (m)
        nnodes = 21                     # number of nodes
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

        # Create extracellular current source
        psource = ExtracellularCurrent(x0, z0, rho=rho_e, mode=mode)

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        data, meta = fiber.simulate(psource, Ithr, tstim, toffset, PRF, DC)

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
        fig = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Compute and plot strength-duration curve with both polarities
        fiber.reset()
        durations = np.logspace(0, 4, 5) * 1e-6  # s
        psources = {k: ExtracellularCurrent(x0, z0, rho=rho_e, mode=k) for k in ['cathode', 'anode']}
        Ithrs = {k: np.array([np.abs(fiber.titrate(v, x, toffset, PRF, DC)) for x in durations])  # A
                 for k, v in psources.items()}
        I0_ref = 0.36e-3  # A
        Ithrs['cathode ref'] = np.array([96.3, 9.44, 1.89, 1.00, 1.00]) * I0_ref  # A

        # Plot strength-duration curve
        fig2 = strengthDurationCurve(fiber, durations, Ithrs, yfactor=1e3, scale='log')

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
        logger.info('Test: SENN model validation against Reilly 1987 data')

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # FrankenHaeuser-Huxley membrane equations
        fiberD = 10e-6                  # fiber diameter (m)
        nnodes = 21
        rho_a = 110.0                   # axoplasm resistivity (Ohm.cm, from McNeal 1976)
        d_ratio = 0.7                   # axon / fiber diameter ratio (from McNeal 1976)
        nodeL = 2.5e-6                  # node length (m, from McNeal 1976)
        fiber = VextSennFiber(pneuron, fiberD, nnodes, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # Electrode and stimulation parameters (from Reilly 1987)
        rho_e = 300.0      # resistivity of external medium (Ohm.cm, from McNeal 1976)
        z0 = fiber.interL  # point-electrode to fiber distance (m, 1 internode length)
        x0 = 0.            # point-electrode located above central node (m)
        tstim = 100e-6     # s
        toffset = 3e-3     # s
        PRF = 100.         # Hz
        DC = 1.            # -

        # Create cathodic and anodic extracellular current sources
        psources = {k: ExtracellularCurrent(x0, z0, rho=rho_e, mode=k) for k in ['cathode', 'anode']}

        # Compute and plot strength-duration curve with both polarities
        durations = np.array([1, 5, 10, 50, 100, 500, 1000, 2000], dtype=float) * 1e-6  # s
        Ithrs = {k: np.array([np.abs(fiber.titrate(v, x, toffset, PRF, DC)) for x in durations])  # A
                 for k, v in psources.items()}
        Qthrs = {k: v * durations for k, v in Ithrs.items()}  # C

        # Plot strength-duration curve
        fig = strengthDurationCurve(fiber, durations, Ithrs, yfactor=1e3, scale='log')

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
        logger.info('Test: SENN model validation against Sweeney 1987 data')

        # Fiber model parameters
        pneuron = getPointNeuron('SW')  # mammalian fiber membrane equations
        fiberD = 10e-6                  # fiber diameter (m)
        nnodes = 19
        rho_a = 54.7                    # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                   # axon / fiber diameter ratio
        nodeL = 1.5e-6                  # node length (m)
        fiber = IinjSennFiber(pneuron, fiberD, nnodes, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # Intracellular stimulation parameters
        tstim = 10e-6   # s
        toffset = 3e-3  # s
        PRF = 100.      # Hz
        DC = 1.         # -

        # Create extracellular current source
        psource = IntracellularCurrent(0, mode='anode')

        # Titrate for a specific duration and simulate fiber at threshold current
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Ithr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # A
        data, meta = fiber.simulate(psource, 1.2 * Ithr, tstim, toffset, PRF, DC)

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
        fig1 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Reset fiber model to avoid NEURON integration errors
        fiber.reset()

        # Compute and plot strength-duration curve for intracellular injection at central node
        psource = IntracellularCurrent(nnodes // 2, mode='anode')
        durations = np.array([10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500], dtype=float) * 1e-6  # s
        Ithrs_ref = np.array([4.4, 2.8, 1.95, 1.6, 1.45, 1.4, 1.30, 1.25, 1.25, 1.25, 1.25, 1.25]) * 1e-9  # A
        Ithrs_sim = np.array([fiber.titrate(psource, x, toffset, PRF, DC) for x in durations])  # A
        Qthrs_sim = Ithrs_sim * durations  # C

        # Plot strength-duration curve
        fig2 = strengthDurationCurve(fiber, durations, {'ref': Ithrs_ref, 'sim': Ithrs_sim}, yfactor=1e9, scale='lin')

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

    def test_ASTIM(self, is_profiled=False):
        ''' Run SENN fiber ASTIM simulation. '''
        logger.info('Test: SENN model ASTIM simulation')

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
        fiber = SonicSennFiber(
            pneuron, fiberD, nnodes, a=a, Fdrive=Fdrive, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # US stimulation parameters
        Astim = 100e3   # Pa
        tstim = 3e-3   # s
        toffset = 3e-3  # s
        PRF = 100.      # Hz
        DC = 1.         # -
        inode = nnodes // 2  # central node

        # Create extracellular current source
        psource = NodeAcousticSource(inode, Fdrive)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        Athr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # Pa
        data, meta = fiber.simulate(psource, 1.2 * Athr, tstim, toffset, PRF, DC)

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
        fig1 = SectionCompTimeSeries([(data, meta)], 'Qm', fiber.ids).render()
        fig2 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        # Clear fiber model to avoid NEURON integration errors
        fiber.clear()

        # Define durations vector for titrations curves
        durations = np.logspace(-5, -2, 30)  # s

        # Titration curves for different US frequencies
        Athrs_vs_Fdrive = {}
        for y in np.array([20, 500, 4e3]) * 1e3:
            id = f'{si_format(y, 0, space=" ")}Hz'
            fiber = SonicSennFiber(
                pneuron, fiberD, nnodes, a=a, Fdrive=y, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)
            psource = NodeAcousticSource(inode, y)
            Athrs_vs_Fdrive[id] = np.array([fiber.titrate(psource, t, toffset, PRF, DC) for t in durations])  # Pa
            fiber.clear()
        fig3 = strengthDurationCurve(
            fiber, durations, Athrs_vs_Fdrive, yfactor=1e-3, scale='log',
            name='amplitude', unit='Pa', plot_chr=False)

        psource = NodeAcousticSource(inode, Fdrive)

        # Titration curves for different sonophore radii
        Athrs_vs_a = {}
        for y in np.array([16., 32., 64.]) * 1e-9:
            id = f'{si_format(y, 0, space=" ")}m'
            fiber = SonicSennFiber(
                pneuron, fiberD, nnodes, a=y, Fdrive=Fdrive, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)
            Athrs_vs_a[id] = np.array([fiber.titrate(psource, t, toffset, PRF, DC) for t in durations])  # Pa
            fiber.clear()
        fig4 = strengthDurationCurve(
            fiber, durations, Athrs_vs_a, yfactor=1e-3, scale='log',
            name='amplitude', unit='Pa', plot_chr=False)

        # Titration curves for different fiber diameters
        Athrs_vs_fiberD = {}
        for y in np.array([5., 10., 20.]) * 1e-6:
            id = f'{si_format(y, 0, space=" ")}m'
            fiber = SonicSennFiber(
                pneuron, y, nnodes, a=a, Fdrive=Fdrive, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)
            Athrs_vs_fiberD[id] = np.array([fiber.titrate(psource, t, toffset, PRF, DC) for t in durations])  # Pa
            fiber.clear()
        fig5 = strengthDurationCurve(
            fiber, durations, Athrs_vs_fiberD, yfactor=1e-3, scale='log',
            name='amplitude', unit='Pa', plot_chr=False)


if __name__ == '__main__':
    tester = TestSenn()
    tester.main()