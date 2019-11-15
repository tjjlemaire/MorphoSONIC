# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-19 19:30:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-14 23:03:08

import numpy as np

from PySONIC.neurons import getPointNeuron
from PySONIC.core import PulsedProtocol
from PySONIC.utils import logger, si_format
from ExSONIC.test import TestFiber
from ExSONIC.core import SonicFiber, myelinatedFiber, NodeAcousticSource, PlanarDiskTransducerSource
from ExSONIC.plt import SectionCompTimeSeries, strengthDurationCurve, strengthDistanceCurve
from ExSONIC.utils import chronaxie


class TestSennAstim(TestFiber):

    def test_centralnode(self, is_profiled=False):
        ''' Run SENN fiber ASTIM simulation. '''
        logger.info('Test: SENN model ASTIM simulation at central node')

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
        fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
            rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=a, Fdrive=Fdrive, fs=fs)

        # US stimulation parameters
        pp = PulsedProtocol(3e-3, 3e-3)
        inode = nnodes // 2  # central node

        # Create extracellular current source
        psource = NodeAcousticSource(inode, Fdrive)

        # Titrate for a specific duration and simulate fiber at threshold US amplitude
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        Athr = fiber.titrate(psource, pp)  # Pa
        data, meta = fiber.simulate(psource, 1.2 * Athr, pp)

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

    def test_SDcurves_centralnode(self):
        logger.info('Test: SENN model ASTIM simulation at central node - SD curves')

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
        fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
            rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=a, Fdrive=Fdrive, fs=fs)

        # US stimulation parameters
        toffset = 3e-3  # s
        inode = nnodes // 2  # central node

        # Create extracellular current source
        psource = NodeAcousticSource(inode, Fdrive)

        # Define durations vector for titrations curves
        durations = np.logspace(-5, -2, 30)  # s

        # Titration curves for different US frequencies
        Athrs_vs_Fdrive = {}
        for f in np.array([20, 500, 4e3]) * 1e3:
            id = f'{si_format(f, 0, space=" ")}Hz'
            fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
                rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=a, Fdrive=f, fs=fs)
            psource = NodeAcousticSource(inode, f)
            Athrs_vs_Fdrive[id] = np.array([
                fiber.titrate(psource, PulsedProtocol(t, toffset)) for t in durations])  # Pa
            fiber.clear()
        fig3 = strengthDurationCurve(
            fiber, durations, Athrs_vs_Fdrive, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

        psource = NodeAcousticSource(inode, Fdrive)

        # Titration curves for different sonophore radii
        Athrs_vs_a = {}
        for radius in np.array([16., 32., 64.]) * 1e-9:
            id = f'{si_format(radius, 0, space=" ")}m'
            fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
                rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=radius, Fdrive=Fdrive, fs=fs)
            Athrs_vs_a[id] = np.array([
                fiber.titrate(psource, PulsedProtocol(t, toffset)) for t in durations])  # Pa
            fiber.clear()
        fig4 = strengthDurationCurve(
            fiber, durations, Athrs_vs_a, scale='log',
            yname='amplitude', yfactor=1e-3, yunit='Pa', plot_chr=False)

        # Titration curves for different fiber diameters
        Athrs_vs_fiberD = {}
        for D in np.array([5., 10., 20.]) * 1e-6:
            id = f'{si_format(D, 0, space=" ")}m'
            fiber = myelinatedFiber(SonicFiber, pneuron, D, nnodes,
                rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=a, Fdrive=Fdrive, fs=fs)
            Athrs_vs_fiberD[id] = np.array([
                fiber.titrate(psource, PulsedProtocol(t, toffset)) for t in durations])  # Pa
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
        fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
            rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=a, Fdrive=Fdrive, fs=fs)

        # US stimulation parameters
        pp = PulsedProtocol(3e-3, 3e-3)

        # Transducer parameters
        z0 = fiber.interL  # default transducer z-location, m
        x0 = 0             # transducer initial x-location, m
        rho = 1204.1       # medium density (kg/m3)
        c = 1515.0         # speed of sound (m/s)
        l = 0.01           # length of a flat lxl transducer surface (m)
        theta = 0         # transducer angle of incidence (radians)
        r_tr = 1.27e-3     # transducer radius (m)

        # Create ultrasound source
        psource = PlanarDiskTransducerSource((x0, z0), Fdrive, rho=rho, c=c, theta=theta, r=r_tr)

        # Titrate for a specific duration and simulate fiber at threshold particle velocity
        logger.info(f'Running titration for {si_format(pp.tstim)}s pulse')
        uthr = fiber.titrate(psource, pp)  # m/s
        data, meta = fiber.simulate(psource, 1.2 * uthr, pp)

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

    def test_SDcurves_transducer(self):
        logger.info('Test: SENN model ASTIM stimulation from external transducer - SD curves')

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
        fiber = myelinatedFiber(SonicFiber, pneuron, fiberD, nnodes,
            rs=rho_a, nodeL=nodeL, d_ratio=d_ratio, a=a, Fdrive=Fdrive, fs=fs)

        # Transducer parameters
        z0 = fiber.interL  # default transducer z-location, m
        x0 = 0             # transducer initial x-location, m
        rho = 1204.1       # medium density (kg/m3)
        c = 1515.0         # speed of sound (m/s)
        l = 0.01           # length of a flat lxl transducer surface (m)
        theta = 0         # transducer angle of incidence (radians)
        r_tr = 1.27e-3     # transducer radius (m)

        # US stimulation parameters
        tstim = 3e-3    # s
        toffset = 3e-3  # s
        pp = PulsedProtocol(tstim, toffset)

        # Create ultrasound source
        psource = PlanarDiskTransducerSource((x0, z0), Fdrive, rho=rho, c=c, theta=theta, r=r_tr)

        # Strength-distance curve
        ztr = np.array([0.5, 1.0, 2.0, 4.0]) * fiber.interL # transducer-fiber distances (m)
        uthrs = []
        for dist in ztr:
            psource.x = (psource.x[0], dist)
            logger.info(f'Running titration for {si_format(tstim)}s pulse')
            uthrs.append(fiber.titrate(psource, pp))  # m/s
            fiber.reset()
        fig3 = strengthDistanceCurve(
            fiber, ztr, {'sim': np.array(uthrs)}, scale='lin',
            yname='particle velocity', yfactor=1e0, yunit='m/s')

        # Strength-duration curve
        psource.x = (psource.x[0], z0)
        durations = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1, 3, 5], dtype=float) * 1e-3  # s
        uthrs = np.array([
            fiber.titrate(psource, PulsedProtocol(x, toffset)) for x in durations])
        fig4 = strengthDurationCurve(
           fiber, durations, {'sim': np.array(uthrs)}, scale='log', plot_chr=False,
           yname='particle velocity', yfactor=1e0, yunit='m/s')


if __name__ == '__main__':
    tester = TestSennAstim()
    tester.main()