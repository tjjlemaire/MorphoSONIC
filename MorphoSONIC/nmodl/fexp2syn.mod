NEURON {
	POINT_PROCESS FExp2Syn
	RANGE g, tau1, tau2, e, i, f, tauF
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
	tau1 (ms) < 1e-9, 1e9 >
	tau2 (ms) < 1e-9, 1e9 >
	e 	 (mV)

	: facilitation
    f (1) < 0, 1e9 >
    tauF (ms) < 1e-9, 1e9 >
}

ASSIGNED {
	v (mV)
	i (nA)
	g (umho)
	factor
}

STATE {
	A (umho)
	B (umho)
}

INITIAL {
	LOCAL tpeak
	if (tau1 / tau2 > 0.9999) {
		tau1 = 0.9999 * tau2
	}
	A = 0
	B = 0
	tpeak = (tau1 * tau2) / (tau2 - tau1) * log(tau2 / tau1)
	factor = -exp(-tpeak / tau1) + exp(-tpeak / tau2) factor = 1 / factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g * (v - e)
}

DERIVATIVE state {
	A' = -A / tau1
	B' = -B / tau2
}

NET_RECEIVE(weight (umho), F, tsyn (ms)) {
	INITIAL {
        F = 1
        tsyn = t
	}
    F = 1 + (F - 1) * exp(-(t - tsyn) / tauF)
    tsyn = t
	A = A + weight * factor * F
	B = B + weight * factor * F
    F = F + f
}
