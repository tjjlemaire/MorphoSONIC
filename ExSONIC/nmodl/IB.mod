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

    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iM
    NONSPECIFIC_CURRENT iCaL
    NONSPECIFIC_CURRENT iLeak

    RANGE Adrive, Vm : section specific
    RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}


PARAMETER {
    stimon       : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    cm = 1              (uF/cm2)
    ENa = 50            (mV)
    ECa = 120           (mV)
    EK = -90            (mV)
    ELeak = -70.0       (mV)
    gNabar = 0.050     (S/cm2)
    gKdbar = 0.005     (S/cm2)
    gMbar = 3.0e-5     (S/cm2)
    gCaLbar = 1.0e-4    (S/cm2)
    gLeak = 1.0e-5     (S/cm2)
}

STATE {
    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    p  : iM activation gate
    q  : iCaL activation gate
    r  : iCaL inactivation gate
}

ASSIGNED {
    Vm      (mV)
    v       (mV)
    iNa     (mA/cm2)
    iKd     (mA/cm2)
    iM      (mA/cm2)
    iCaL     (mA/cm2)
    iLeak   (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE pinf(A(kPa), Q(nC/cm2)) ()
FUNCTION_TABLE taup(A(kPa), Q(nC/cm2)) (ms)
FUNCTION_TABLE alphaq(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betaq(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphar(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betar(A(kPa), Q(nC/cm2)) (/ms)

INITIAL {
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = pinf(0, v)
    q = alphaq(0, v) / (alphaq(0, v) + betaq(0, v))
    r = alphar(0, v) / (alphar(0, v) + betar(0, v))
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    Vm = V(Adrive * stimon, v)
    iNa = gNabar * m * m * m * h * (Vm - ENa)
    iKd = gKdbar * n * n * n * n * (Vm - EK)
    iM = gMbar * p * (Vm - EK)
    iCaL = gCaLbar * q * q * r * (Vm - ECa)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    p' = (pinf(Adrive * stimon, v) - p) / taup(Adrive * stimon, v)
    q' = alphaq(Adrive * stimon, v) * (1 - q) - betaq(Adrive * stimon, v) * q
    r' = alphar(Adrive * stimon, v) * (1 - r) - betar(Adrive * stimon, v) * r
}