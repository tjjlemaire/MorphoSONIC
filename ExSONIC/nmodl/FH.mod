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
    pNabar = 8e-3     (cm/s)
	pPbar = .54e-3    (cm/s)
	pkbar = 1.2e-3    (cm/s)
	gLeak = 30.3e-3   (S/cm2)
	ELeak = -69.74    (mV)
	Tcelsius = 20.00  (degC)

	: ionic concentrations
	Nai = 13.74e-3    (M)
    Nao = 114.5e-3    (M)
    Ki = 120e-3       (M)
    Ko = 2.5e-3       (M)
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

INITIAL {
    : Set initial states values
    q10 = 3^((Tcelsius - 20) / 10)
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = alphap(0, v) / (alphap(0, v) + betap(0, v))
}

BREAKPOINT {
	LOCAL ghkNa, ghkK

    : Integrate states
    SOLVE states METHOD cnexp

    : Compute effective membrane potential
    Vmeff = V(Adrive * stimon, v)

    : compute ionic currents
    ghkNa = ghkDrive(Vmeff, Nai, Nao)
    ghkK = ghkDrive(Vmeff, Ki, Ko)
    iNa = pNabar * m * m * h * ghkNa
	iKd = pkbar * n * n * ghkK
    iP = pPbar * p * p * ghkNa
	iLeak = gLeak * (Vmeff - ELeak)
}

FUNCTION ghkDrive(v(mV), Ci(M), Co(M)) {
	:assume a single charge
	LOCAL x, ECi, ECo
	x = FARADAY * v / (R*(Tcelsius+273.15)) * 1e-3
	ECi = Ci*efun(-x)
	ECo = Co*efun(x)
	ghkDrive = FARADAY*(ECi - ECo)
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
