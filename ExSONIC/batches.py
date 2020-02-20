# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-17 12:19:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-20 13:02:39

import abc
import numpy as np

from PySONIC.core import LogBatch, PulsedProtocol
from PySONIC.utils import logger, si_format, isPickable
from PySONIC.postpro import boundDataFrame


class MPIBatch:

    def __init__(self, pc, simfunc):
        self.pc = pc
        self.simfunc = simfunc

    def run(self, queue):
        if self.pc is None or self.pc.nhost() == 1:
            return self.run_serial(queue)
        else:
            return self.run_parallel(queue)

    def run_serial(self, queue):
        return [self.simfunc(x) for x in queue]

    def run_parallel(self, queue):

        global pickable_simfunc
        def pickable_simfunc(*args, **kwargs):
            return self.simfunc(*args, **kwargs)

        if not isPickable(pickable_simfunc):
            raise ValueError('simulation function is not pickable')
        index, output = [], []
        for x in queue:
            self.pc.submit(pickable_simfunc, x)
        while True:
            task_id = self.pc.working()
            if task_id == 0:
                break
            output.append(self.pc.pyret())
            index.append(task_id)
        return [x for _, x in sorted(zip(index, output))]


class StrengthDurationBatch(LogBatch):
    ''' Interface to a strength-duration batch '''

    in_key = 'tstim'
    suffix = 'strengthduration'
    unit = 's'

    def __init__(self, out_key, source, fiber, durations, offset, root='.', convert_func=None):
        ''' Construtor.

            :param out_key: string defining the unique batch output key
            :param source: source object
            :param fiber: fiber model
            :param durations: array of pulse durations
            :param toffset: constant stimulus offset
            :param root: root for IO operations
            :return: array of threshold acoustic amplitudes for each pulse duration
        '''
        self.out_keys = [out_key]
        if convert_func is None:
            self.convert_func = lambda x: x
        else:
            self.convert_func = convert_func
        self.source = source
        self.fiber = fiber
        self.offset = offset
        super().__init__(durations, root=root)

    @property
    def out_keys(self):
        return self._out_keys

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value

    def sourcecode(self):
        return self.source.quickcode

    def corecode(self):
        return f'{self.fiber.quickcode}_source_{self.sourcecode()}'

    def run(self):
        logger.info(f'Computing SD curve for {self.fiber} with {self.source}')
        return super().run()

    def compute(self, t):
        xthr = self.fiber.titrate(self.source, PulsedProtocol(t, self.offset))
        return [self.convert_func(xthr)]


class FiberConvergenceBatch(LogBatch):
    ''' Generic interface to a fiber convergence study. '''

    suffix = 'convergence'
    out_keys = ['nodeL (m)', 'Ithr (A)', 'CV (m/s)', 'dV (mV)']
    unit = 'm'

    def __init__(self, in_key, fiber_func, source_func, inputs, pp, root='.'):
        ''' Construtor.

            :param in_key: string defining the batch input key
            :param fiber_func: function used to create the fiber object
            :param source_func: function used to create the source object
            :param inputs: spatial discretization parameters
            :param pp: pulsing protocol object
            :param root: root for IO operations
        '''
        self.in_key = in_key
        self.fiber_func = fiber_func
        self.source_func = source_func
        self.pp = pp
        super().__init__(inputs, root=root)

    @property
    def in_key(self):
        return self._in_key

    @in_key.setter
    def in_key(self, value):
        self._in_key = value

    def fibercode(self):
        ''' String fully describing the fiber object. '''
        codes = self.getFiber(self.inputs[0]).modelcodes
        for k in ['nnodes', 'interL', 'interD', 'nodeL']:
            if k in codes:
                del codes[k]
        return '_'.join(codes.values())

    def corecode(self):
        return f'{self.fibercode()}_{si_format(self.pp.tstim, 1, "")}s'

    def getFiber(self, x):
        ''' Initialize fiber with specific discretization parameter. '''
        logger.info(f'creating model with {self.in_key} = {si_format(x, 2)}m ...')
        fiber = self.fiber_func(x)
        logger.info(f'resulting node length: {si_format(fiber.nodeL, 2)}m')
        return fiber

    def compute(self, x):
        ''' Create a fiber with a specific discretization pattern, simulate it upon application
            of a specific stimulus, and compute the following output metrics:
                - node length (m)
                - stimulation threshold (in source units)
                - conduction velocity (m/s)
                - spike amplitude (mV)

            :param x: discretization parameter
            :return: tuple with the computed output metrics
        '''
        # Initialize fiber and source
        fiber = self.getFiber(x)
        source = self.source_func(fiber)

        # Perform titration to find threshold excitation amplitude
        logger.info(f'Running titration with intracellular current injected at node {source.inode}')
        Ithr = fiber.titrate(source, self.pp)  # A

        if not np.isnan(Ithr):
            # If fiber is excited
            logger.info(f'Ithr = {si_format(Ithr, 2)}A')

            # Simulate fiber at 1.1 times threshold current
            data, meta = fiber.simulate(source.updatedX(1.1 * Ithr), self.pp)

            # Filter out stimulation artefact from dataframe
            data = {k: boundDataFrame(df, (self.pp.tstim, self.pp.ttotal)) for k, df in data.items()}

            # Compute CV and spike amplitude
            cv = fiber.getConductionVelocity(data, out='median')  # m/s
            dV = fiber.getSpikeAmp(data, out='median')            # mV
            logger.info(f'CV = {cv:.2f} m/s')
            logger.info(f'dV = {dV:.2f} mV')
        else:
            # Otherwise, assign NaN values to them
            cv, dV = np.nan, np.nan

        return fiber.nodeL, Ithr, cv, dV

    def run(self):
        logger.info('running {0} parameter sweep ({1}{3} - {2}{3})'.format(
            self.in_key, *si_format([self.inputs.min(), self.inputs.max()], 2), self.unit))
        out = super().run()
        logger.info('parameter sweep successfully completed')
        return out

