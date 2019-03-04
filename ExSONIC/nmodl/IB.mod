TITLE IB neuron membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of an intrinsically bursting neuron upon ultrasonic
stimulation, based on the multi-Scale Optmimized Neuronal Intramembrane Cavitation (SONIC) model.

@Author: Theo Lemaire, EPFL
@Date:   2018-08-21
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (uF) = (microfarad)
    (kPa) = (kilopascal)
}

NEURON {
    SUFFIX IB

    : Constituting currents
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iM
    NONSPECIFIC_CURRENT iCa
    NONSPECIFIC_CURRENT iLeak

    : RANGE variables
    RANGE Adrive, Vmeff : section specific
    RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)
}


PARAMETER {
    : Parameters set by python/hoc caller
    stimon : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    : membrane properties
    cm = 1              (uF/cm2)
    ena = 50            (mV)
    eca = 120           (mV)
    ek = -90            (mV)
    eleak = -70.0       (mV)
    gnabar = 0.050     (S/cm2)
    gkdbar = 0.005     (S/cm2)
    gmbar = 3.0e-5     (S/cm2)
    gcabar = 1.0e-4    (S/cm2)
    gleak = 1.0e-5     (S/cm2)
}

STATE {
    : Gating states
    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    p  : iM activation gate
    q  : iCa activation gate
    r  : iCa inactivation gate
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Vmeff   (mV)
    v       (mV)
    iNa     (mA/cm2)
    iKd     (mA/cm2)
    iM      (mA/cm2)
    iCa     (mA/cm2)
    iLeak   (mA/cm2)
}

: Function tables to interpolate effective variables
FUNCTION_TABLE V(A(kPa), Q(nC/cm2))      (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphaq(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betaq(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphar(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betar(A(kPa), Q(nC/cm2))   (/ms)

INITIAL {
    : Set initial states values
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = alphap(0, v) / (alphap(0, v) + betap(0, v))
    q = alphaq(0, v) / (alphaq(0, v) + betaq(0, v))
    r = alphar(0, v) / (alphar(0, v) + betar(0, v))
}

BREAKPOINT {
    : Integrate states
    SOLVE states METHOD cnexp

    : Compute effective membrane potential
    Vmeff = V(Adrive * stimon, v)

    : Compute ionic currents
    iNa = gnabar * m * m * m * h * (Vmeff - ena)
    iKd = gkdbar * n * n * n * n * (Vmeff - ek)
    iM = gmbar * p * (Vmeff - ek)
    iCa = gcabar * q * q * r * (Vmeff - eca)
    iLeak = gleak * (Vmeff - eleak)
}

DERIVATIVE states {
    : Gating states derivatives
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    p' = alphap(Adrive * stimon, v) * (1 - p) - betap(Adrive * stimon, v) * p
    q' = alphaq(Adrive * stimon, v) * (1 - q) - betaq(Adrive * stimon, v) * q
    r' = alphar(Adrive * stimon, v) * (1 - r) - betar(Adrive * stimon, v) * r
}