TITLE FH membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Xenopus axonal node (governed by
Frankenhaeuser - Huxley equations) upon ultrasonic stimulation, based on the multi-Scale
Optmimized Neuronal Intramembrane Cavitation (SONIC) model.

@Author: Theo Lemaire, EPFL
@Date:   2019-01-09
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (uF) = (microfarad)
    (nC) = (nanocoulomb)
    (kPa) = (kilopascal)

    (molar) = (1/liter)         : moles do not appear in units
    (M)     = (molar)
    (mM)    = (millimolar)
}

NEURON {
	SUFFIX FH

    : Constituting currents
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iP
    NONSPECIFIC_CURRENT iLeak

    : RANGE variables
    RANGE Adrive, Vmeff : section specific
    RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)
}

CONSTANT {
	FARADAY = 9.64853e4 (coul) : Faraday constant (C/mol, moles do not appear in units)
	R = 8.31342  : Universal gas constant (Pa.m^3.mol^-1.K^-1 or J.mol^-1.K^-1)
}


PARAMETER {
    : Parameters set by python/hoc caller
    stimon : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    q10

    : membrane properties
    v0 = -70          (mV)
    pnabar = 8e-3     (cm/s)
	ppbar = .54e-3    (cm/s)
	pkbar = 1.2e-3    (cm/s)
	gleak = 30.3e-3   (S/cm2)
	eleak = -69.74    (mV)
	Tcelsius = 20.00  (degC)

	: ionic concentrations
	nai = 13.74e-3    (M)
    nao = 114.5e-3    (M)
    ki = 120e-3       (M)
    ko = 2.5e-3       (M)
}

STATE {
    : Gating states
    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    p  : iP activation gate
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Vmeff   (mV)
    v        (mV)
    iNa      (mA/cm2)
    iKd      (mA/cm2)
    iP       (mA/cm2)
    iLeak    (mA/cm2)
}

: Function tables to interpolate effective variables
FUNCTION_TABLE V(A(kPa), Q(nC/cm2))       (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2))   (/ms)


COMMENT
FUNCTION V(A(kPa), Q(nC/cm2)) {
	V = Q / cm
}

FUNCTION alpham(A(kPa), Q(nC/cm2)){ LOCAL Vdiff
	:printf("v = %f, Q = %f, cm = %f, Q/cm = %f\n", v, Q, cm, Q / cm)
    Vdiff = Q / cm - v0
    alpham = q10 * 0.36 * vtrap(22. - Vdiff, 3.)
}

FUNCTION betam(A(kPa), Q(nC/cm2)) { LOCAL Vdiff
	Vdiff = Q / cm - v0
	betam = q10 * 0.4 * vtrap(Vdiff - 13., 20.)
}

FUNCTION alphah(A(kPa), Q(nC/cm2)) { LOCAL Vdiff
	Vdiff = Q / cm - v0
	alphah = q10 * 0.1 * vtrap(Vdiff + 10.0, 6.)
}

FUNCTION betah(A(kPa), Q(nC/cm2)) { LOCAL Vdiff
	Vdiff = Q / cm - v0
	betah = q10 * 4.5 / (exp((45. - Vdiff) / 10.) + 1)
}

FUNCTION alphan(A(kPa), Q(nC/cm2)) { LOCAL Vdiff
	Vdiff = Q / cm - v0
	alphan = q10 * 0.02 * vtrap(35. - Vdiff, 10.0)
}

FUNCTION betan(A(kPa), Q(nC/cm2)) { LOCAL Vdiff
	Vdiff = Q / cm - v0
	betan = q10 * 0.05 * vtrap(Vdiff - 10., 10.)
}

FUNCTION alphap(A(kPa), Q(nC/cm2)) { LOCAL Vdiff
	Vdiff = Q / cm - v0
	alphap = q10 * 0.006 * vtrap(40. - Vdiff, 10.0)
}

FUNCTION betap(A(kPa), Q(nC/cm2)) { LOCAL Vdiff
	Vdiff = Q / cm - v0
	betap = q10 * 0.09 * vtrap(Vdiff + 25., 20.)
}
ENDCOMMENT

INITIAL {
    : Set initial states values
    q10 = 3^((Tcelsius - 20) / 10)
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = alphap(0, v) / (alphap(0, v) + betap(0, v))
}

BREAKPOINT {
	LOCAL ghk_na, ghk_k

    : Integrate states
    SOLVE states METHOD cnexp

    : Compute effective membrane potential
    Vmeff = V(Adrive * stimon, v)

    : compute ionic currents
    ghk_na = ghkDrive(Vmeff, nai, nao)
    ghk_k = ghkDrive(Vmeff, ki, ko)
    iNa = pnabar * m * m * h * ghk_na
	iKd = pkbar * n * n * ghk_k
    iP = ppbar * p * p * ghk_na
	iLeak = gleak * (Vmeff - eleak)
}

FUNCTION ghkDrive(v(mV), ci(M), co(M)) {
	:assume a single charge
	LOCAL x, eci, eco
	x = FARADAY * v / (R*(Tcelsius+273.15)) * 1e-3
	eci = ci*efun(-x)
	eco = co*efun(x)
	ghkDrive = FARADAY*(eci - eco)
}

FUNCTION efun(x) {
	efun = x / (exp(x) - 1)
}

FUNCTION vtrap(x, y) {
    vtrap = x / (exp(x / y) - 1)
}

DERIVATIVE states {
    : States derivatives
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    p' = alphap(Adrive * stimon, v) * (1 - p) - betap(Adrive * stimon, v) * p
}
