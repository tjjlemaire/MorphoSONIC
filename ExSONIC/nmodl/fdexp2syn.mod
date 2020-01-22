NEURON {
	POINT_PROCESS FDExp2Syn
	RANGE g, tau1, tau2, e, i, f, tauF, d1, tauD1, d2, tauD2
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

    : fast depression
    d1 (1) < 0, 1 >
    tauD1 (ms) < 1e-9, 1e9 >

    : slow depression
    d2  (1) < 0, 1 >
    tauD2 (ms) < 1e-9, 1e9 >
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

NET_RECEIVE(weight (umho), F, D1, D2, tsyn (ms)) {
	INITIAL {
        F = 1
        D1 = 1
        D2 = 1
        tsyn = t
	}
    F = 1 + (F - 1) * exp(-(t - tsyn) / tauF)
    D1 = 1 + (D1 - 1) * exp(-(t - tsyn) / tauD1)
    D2 = 1 + (D2 - 1) * exp(-(t - tsyn) / tauD2)
    tsyn = t
	A = A + weight * factor * F * D1 * D2
	B = B + weight * factor * F * D1 * D2
    F = F + f
    D1 = D1 * d1
    D2 = D2 * d2
}
