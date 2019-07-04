TITLE FS neuron membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a regular spiking neuron upon ultrasonic
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
    SUFFIX FS


    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iM
    NONSPECIFIC_CURRENT iLeak

    RANGE Adrive, Vm : section specific
    RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
    stimon       : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    cm = 1              (uF/cm2)
    ENa = 50            (mV)
    EK = -90            (mV)
    ELeak = -70.4       (mV)
    gNabar = 0.058      (S/cm2)
    gKdbar = 0.0039     (S/cm2)
    gMbar = 7.87e-5     (S/cm2)
    gLeak = 3.8e-5      (S/cm2)
}

STATE {
    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    p  : iM activation gate
}

ASSIGNED {
    Vm        (mV)
    v         (mV)
    iNa       (mA/cm2)
    iKd       (mA/cm2)
    iM        (mA/cm2)
    iLeak     (mA/cm2)
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

INITIAL {
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = pinf(0, v)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    Vm = V(Adrive * stimon, v)
    iNa = gNabar * m * m * m * h * (Vm - ENa)
    iKd = gKdbar * n * n * n * n * (Vm - EK)
    iM = gMbar * p * (Vm - EK)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    p' = (pinf(Adrive * stimon, v) - p) / taup(Adrive * stimon, v)
}