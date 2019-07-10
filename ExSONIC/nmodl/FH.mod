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

    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iP
    NONSPECIFIC_CURRENT iLeak

    RANGE Adrive, Vm : section specific
    RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

CONSTANT {
    Z_Na = 1
    FARADAY = 96485.3
    Rg = 8.31342
    Z_K = 1
}


PARAMETER {
    stimon       : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    pNabar = 8e-3     (cm/s)
	pPbar = .54e-3    (cm/s)
	pkbar = 1.2e-3    (cm/s)
	gLeak = 30.3e-3   (S/cm2)
	ELeak = -69.974    (mV)
	T = 293.15  (K)
	Nai = 13.74e-3    (M)
    Nao = 114.5e-3    (M)
    Ki = 120e-3       (M)
    Ko = 2.5e-3       (M)
}

STATE {
    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    p  : iP activation gate
}

ASSIGNED {
    Vm       (mV)
    v        (mV)
    iNa      (mA/cm2)
    iKd      (mA/cm2)
    iP       (mA/cm2)
    iLeak    (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2))       (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2))   (/ms)

FUNCTION efun(x) {
    efun = x / (exp(x) - 1)
}

FUNCTION ghkDrive(Vm, Z_ion, Cion_in, Cion_out, T) {
    LOCAL x, eCi, eCo
    x = FARADAY * Vm / (Rg * T) * 1e-3
    eCi = Cion_in * efun(-x)
    eCo = Cion_out * efun(x)
    ghkDrive = FARADAY * (eCi - eCo)
}

INITIAL {
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = alphap(0, v) / (alphap(0, v) + betap(0, v))
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    Vm = V(Adrive * stimon, v)
    iNa = pNabar * m * m * h * ghkDrive(Vm, Z_Na, Nai, Nao, T)
    iKd = pkbar * n * n * ghkDrive(Vm, Z_K, Ki, Ko, T)
    iP = pPbar * p * p * ghkDrive(Vm, Z_Na, Nai, Nao, T)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    p' = alphap(Adrive * stimon, v) * (1 - p) - betap(Adrive * stimon, v) * p
}
