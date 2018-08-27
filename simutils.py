import time
import logging
import pickle
import numpy as np
import pandas as pd

from PySONIC.utils import si_format, logger
from PySONIC.solvers import findPeaks
from PySONIC.constants import *

from sonic0D import Sonic0D

logger.setLevel(logging.INFO)


class AStimWorker():
    ''' Worker class that runs a single A-STIM simulation a given neuron for specific
        stimulation parameters, and save the results in a PKL file. '''

    def __init__(self, wid, batch_dir, log_filepath, neuron, a, Fdrive, Adrive, tstim, toffset,
                 PRF, DC, dt=None, atol=None, nsims=1):
        ''' Class constructor.

            :param wid: worker ID
            :param neuron: neuron object
            :param a: sonophore diameter (m)
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param nsims: total number or simulations
        '''

        self.id = wid
        self.batch_dir = batch_dir
        self.log_filepath = log_filepath
        self.neuron = neuron
        self.Fdrive = Fdrive
        self.Adrive = Adrive
        self.tstim = tstim
        self.toffset = toffset
        self.PRF = PRF
        self.DC = DC
        self.dt = dt
        self.atol = atol
        self.nsims = nsims

    def __call__(self):
        ''' Method that runs the simulation. '''

        # Determine simulation code
        simcode = 'ASTIM_{}_{}_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_{}_NEURON'.format(
            self.neuron.name,
            'CW' if self.DC == 1 else 'PW',
            self.a * 1e9,
            self.Fdrive * 1e-3,
            self.Adrive * 1e-3,
            self.tstim * 1e3,
            'PRF{:.2f}Hz_DC{:.2f}%_'.format(self.PRF, self.DC * 1e2) if self.DC < 1. else ''
        )

        # Get date and time info
        date_str = time.strftime("%Y.%m.%d")
        daytime_str = time.strftime("%H:%M:%S")

        # Run simulation
        tstart = time.time()
        model = Sonic0D(self.neuron, a=self.a, Fdrive=self.Fdrive)
        model.setAdrive(self.Adrive * 1e-3)
        (t, y, stimon) = model.simulate(self.tstim, self.toffset, self.PRF, self.DC,
                                        self.dt, self.atol)
        Qm, Vm, *states = y
        tcomp = time.time() - tstart
        logger.debug('completed in %ss', si_format(tcomp, 2))

        # Store dataframe and metadata
        df = pd.DataFrame({'t': t, 'stimon': stimon, 'Qm': Qm * 1e-5, 'Vm': Vm})
        for j in range(len(self.neuron.states_names)):
            df[self.neuron.states_names[j]] = states[j]
        meta = {'neuron': self.neuron.name, 'a': self.a,
                'Fdrive': self.Fdrive, 'Adrive': self.Adrive, 'phi': np.pi,
                'tstim': self.tstim, 'toffset': self.toffset, 'PRF': self.PRF, 'DC': self.DC,
                'tcomp': tcomp}
        if self.dt is not None:
            meta['dt'] = self.dt
        if self.atol is not None:
            meta['atol'] = self.atol

        # Export into to PKL file
        output_filepath = '{}/{}.pkl'.format(self.batch_dir, simcode)
        with open(output_filepath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': df}, fh)
        logger.debug('simulation data exported to "%s"', output_filepath)

        # Detect spikes on Qm signal
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(Qm, SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_QPROM)
        n_spikes = ipeaks.size
        lat = t[ipeaks[0]] if n_spikes > 0 else 'N/A'
        sr = np.mean(1 / np.diff(t[ipeaks])) if n_spikes > 1 else 'N/A'
        logger.debug('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

        # Export key metrics to log file
        log = {
            'A': date_str,
            'B': daytime_str,
            'C': self.neuron.name,
            'D': self.a * 1e9,
            'E': 0.0,
            'F': self.Fdrive * 1e-3,
            'G': self.Adrive * 1e-3,
            'H': self.tstim * 1e3,
            'I': self.PRF * 1e-3 if self.DC < 1 else 'N/A',
            'J': self.DC,
            'K': 'NEURON',
            'L': t.size,
            'M': round(tcomp, 4),
            'N': n_spikes,
            'O': lat * 1e3 if isinstance(lat, float) else 'N/A',
            'P': sr * 1e-3 if isinstance(sr, float) else 'N/A'
        }

        if xlslog(self.log_filepath, 'Data', log) == 1:
            logger.debug('log exported to "%s"', self.log_filepath)
        else:
            logger.error('log export to "%s" aborted', self.log_filepath)

        return output_filepath

    def __str__(self):
        worker_str = 'A-STIM {} simulation {}/{}: {} neuron, a = {}m, f = {}Hz, A = {}Pa, t = {}s'\
            .format(self.int_method, self.id, self.nsims, self.neuron.name,
                    *si_format([self.a, self.Fdrive], 1, space=' '),
                    si_format(self.Adrive, 2, space=' '), si_format(self.tstim, 1, space=' '))
        if self.DC < 1.0:
            worker_str += ', PRF = {}Hz, DC = {:.2f}%'\
                .format(si_format(self.PRF, 2, space=' '), self.DC * 1e2)
        return worker_str
