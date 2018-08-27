
# This uses NEURON's event delivery system to control the current supplied
# by an IClamp.  A similar strategy can be used to drive discontinuous
# changes in any other parameter.
# 1.  During simulation initialization, the stimulus current is set to 0,
# and an FInitializeHandler is used to launch an event that will arrive
# at the time when we want the first jump of stim.amp to occur.
# 2.  Arrival of this event causes proc seti() to be called.
# 3.  seti() assigns a new value to stim.amp, and uses the CVode class's
# event() method to launch two new events.  The first of these will come
# back in the future to turn off the stimulus.  The second will come back
# a bit later, to turn it back on again, and start a new cycle.

from neuron import h

stim = h.IClamp(0.5)

DUR = 0.1  # ms, duration of each pulse
AMP = 0.1  # nA
START = 5  # ms, time of first pulse
INTERVAL = 25  # ms, interval between pulses (from pulse end to the start of the next one)


fih = h.FInitializeHandler(initi)
cvode = h.CVode()

STIMON = 0


def initi():
    STIMON = 0
    stim.amp = 0  # prevent value at end of a run from contaminating the start of the following run
    stim.delay = 0  # we want to exert control over amp starting at 0 ms
    stim.dur = 1e9  # if we're going to change amp, dur must be long enough to span all our changes
    cvode.event(START, seti)
    print("launched event that will turn on pulse at ", START)


def seti():
    print("t = ", t)
    if (STIMON == 0):
        STIMON = 1
        stim.amp = AMP
        cvode.event(t + DUR, seti)
        print("stim.amp = ", stim.amp, ", launched event to turn pulse off")
    else:
        STIMON = 0
        stim.amp = 0
        cvode.event(t + INTERVAL, seti)
        print("stim.amp = ", stim.amp, ", launched event to turn next pulse on")

    # we've changed a parameter abruptly so we really should re-initialize cvode
    if cvode.active():
        cvode.re_init()
    else:
        fcurrent()
