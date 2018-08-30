# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-27 14:38:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-30 11:09:19

import os
from neuron import h


def getNmodlDir():
    ''' Return path to directory containing MOD files and compiled mechanisms files. '''
    selfdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(selfdir, 'nmodl')


class IPulse:
    ''' This uses NEURON's event delivery system to control the current supplied by an IClamp.
        A similar strategy can be used to drive discontinuous changes in any other parameter.
        1.  During simulation initialization, the stimulus current is set to 0, and an
        FInitializeHandler is used to launch an event that will arrive at the time when we want the
        first jump of stim.amp to occur.
        2.  Arrival of this event causes proc toggleStim() to be called.
        3.  toggleStim() assigns a new value to stim.amp, and uses the CVode class's event() method to
        launch two new events.  The first of these will come back in the future to turn off the
        stimulus. The second will come back a bit later, to turn it back on again, and start a new
        cycle.
    '''

    def __init__(self, sec, dur, amp, start, interval):

        self.dur = dur  # ms, duration of each pulse
        self.amp = amp  # nA
        self.start = start  # ms, time of first pulse
        self.interval = interval  # ms, interval between pulses (from pulse end to the start of the next one)

        self.stimon = 0
        self.stim = h.IClamp(sec(0.5))

        self.fih = h.FInitializeHandler(self.initialize)
        self.cvode = h.CVode()

        self.initialize()

    def initialize(self):
        self.stimon = 0
        self.stim.amp = 0  # prevent value at end of a run from contaminating the start of the following run
        self.stim.delay = 0  # we want to exert control over amp starting at 0 ms
        self.stim.dur = 1e9  # if we're going to change amp, dur must be long enough to span all our changes
        self.cvode.event(start, self.toggleStim)
        print("launched event that will turn on pulse at ", self.start)

    def toggleStim(self):
        print("t = ", h.t)
        if (self.stimon == 0):
            self.stimon = 1
            self.stim.amp = self.amp
            self.cvode.event(h.t + self.dur, self.toggleStim)
            print("stim.amp = ", self.stim.amp, ", launched event to turn pulse off")
        else:
            self.stimon = 0
            self.stim.amp = 0
            self.cvode.event(h.t + self.interval, self.toggleStim)
            print("stim.amp = ", self.stim.amp, ", launched event to turn next pulse on")

        # we've changed a parameter abruptly so we really should re-initialize cvode
        if self.cvode.active():
            self.cvode.re_init()
        else:
            h.fcurrent()


if __name__ == '__main__':
    dur = 0.1  # ms, duration of each pulse
    amp = 0.1  # nA
    start = 5  # ms, time of first pulse
    interval = 25  # ms, interval between pulses (from pulse end to the start of the next one)

    sec = h.Section(name='section')
    ipulse = IPulse(sec, dur, amp, start, interval)
