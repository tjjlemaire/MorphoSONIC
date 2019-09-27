# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-27 11:27:41

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, si_format
from ExSONIC.test import TestFiber
from ExSONIC.core import SonicSennFiber, NodeAcousticSource, PlanarDiskTransducerSource
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve, strengthDistanceCurve
from ExSONIC.utils import chronaxie


class TestSennAstim(TestFiber):

    def test_centralnode(self, is_profiled=False):
        ''' Run SENN fiber ASTIM simulation. '''
        logger.info('Test: SENN model ASTIM simulation at central node')

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # Frog myelinated node membrane equations
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
            fiber, durations, Athrs_vs_Fdrive, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

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
            fiber, durations, Athrs_vs_a, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

        # Titration curves for different fiber diameters
        Athrs_vs_fiberD = {}
        for y in np.array([5., 10., 20.]) * 1e-6:
            id = f'{si_format(y, 0, space=" ")}m'
            fiber = SonicSennFiber(
                pneuron, y, nnodes, a=a, Fdrive=Fdrive, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)
            Athrs_vs_fiberD[id] = np.array([fiber.titrate(psource, t, toffset, PRF, DC) for t in durations])  # Pa
            fiber.clear()
        fig5 = strengthDurationCurve(
            fiber, durations, Athrs_vs_fiberD, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)


    def test_transducer(self, is_profiled=False):
        ''' Run SENN fiber ASTIM simulation for a flat external transducer. '''
        logger.info('Test: SENN model ASTIM stimulation from external transducer')

        # Fiber model parameters
        pneuron = getPointNeuron('FH')  # Frog myelinated node membrane equations
        fiberD = 10e-6                  # fiber diameter (m)
        nnodes = 15
        rho_a = 54.7                    # axoplasm resistivity (Ohm.cm)
        d_ratio = 0.6                   # axon / fiber diameter ratio
        nodeL = 1.5e-6                  # node length (m)
        a = 32e-9                       # sonophore diameter (m)
        Fdrive = 500e3                  # US frequency (Hz)
        fs = 1                          # sonophore membrane coverage (-)
        fiber = SonicSennFiber(
            pneuron, fiberD, nnodes, a=a, Fdrive=Fdrive, rs=rho_a, nodeL=nodeL, d_ratio=d_ratio)

        # US stimulation parameters
        tstim = 3e-3    # s
        toffset = 3e-3  # s
        PRF = 100.      # Hz
        DC = 1.         # -

        # Transducer parameters
        z0 = fiber.interL  # default transducer z-location, m
        x0 = 0             # transducer initial x-location, m
        rho = 1204.1       # medium density (kg/m3)
        c = 1515.0         # speed of sound (m/s)
        l = 0.01           # length of a flat lxl transducer surface (m)
        theta = 0         # transducer angle of incidence (radians)
        r_tr = 1.27e-3     # transducer radius (m)

        # Create ultrasound source
        psource = PlanarDiskTransducerSource(x0, z0, Fdrive, rho=rho, c=c, theta=theta, r=r_tr)

        # Titrate for a specific duration and simulate fiber at threshold particle velocity
        logger.info(f'Running titration for {si_format(tstim)}s pulse')
        uthr = fiber.titrate(psource, tstim, toffset, PRF, DC)  # m/s
        data, meta = fiber.simulate(psource, 1.2 * uthr, tstim, toffset, PRF, DC)

        # Compute conduction velocity and spike amplitude from resulting data
        sim_metrics = {
            'uthr': uthr,                             # m/s
            'cv': fiber.getConductionVelocity(data),  # m/s
            'dV': fiber.getSpikeAmp(data)             # mV
        }

        # Log output metrics
        self.logOutputMetrics(sim_metrics)

        # Plot membrane potential traces for specific duration at threshold current
        fig1 = SectionCompTimeSeries([(data, meta)], 'Qm', fiber.ids).render()
        fig2 = SectionCompTimeSeries([(data, meta)], 'Vm', fiber.ids).render()

        fiber.reset()

        # Strength-distance curve
        ztr = np.array([0.5, 1.0, 2.0, 4.0]) * fiber.interL # transducer-fiber distances (m)
        uthrs = []
        for dist in ztr:
            psource.z = dist
            logger.info(f'Running titration for {si_format(tstim)}s pulse')
            uthrs.append(fiber.titrate(psource, tstim, toffset, PRF, DC))  # m/s
            fiber.reset()
        fig3 = strengthDistanceCurve(
            fiber, ztr, {'sim': np.array(uthrs)}, scale='lin',
            yname='particle velocity', yfactor=1e0, yunit='m/s')

        # Strength-duration curve
        psource.z = z0
        durations = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1, 3, 5], dtype=float) * 1e-3  # s
        uthrs = np.array([fiber.titrate(psource, x, toffset, PRF, DC) for x in durations])
        fig4 = strengthDurationCurve(
            fiber, durations, {'sim': np.array(uthrs)}, scale='log', plot_chr=False,
            yname='particle velocity', yfactor=1e0, yunit='m/s')


if __name__ == '__main__':
    tester = TestSennAstim()
    tester.main()