TITLE sundt membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Unmyelinated C-fiber model.
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-11-22
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX SUsegauto
    NONSPECIFIC_CURRENT iNa : Sodium current.  Gating formalism from Migliore 1995, using 3rd power for m to reproduce 1 ms AP half-width
    NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
    NONSPECIFIC_CURRENT iLeak : non-specific leakage current
    RANGE Adrive, Vm, y, Fdrive, A_t : section specific
    RANGE stimon, detailed    : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
    stimon       : Stimulation state
    Fdrive (kHz) : Stimulation frequency
    Adrive (kPa) : Stimulation amplitude
    detailed     : Simulation type
    gNabar = 0.04 (S/cm2)
    ENa = 55.0 (mV)
    gKdbar = 0.04 (S/cm2)
    EK = -90.0 (mV)
    gLeak = 0.0001 (S/cm2)
    ELeak = -60.069215464991295 (mV)
}

STATE {
    m : iNa activation gate
    h : iNa inactivation gate
    n : iKd activation gate
    l : iKd inactivation gate
}

ASSIGNED {
    v  (nC/cm2)
    Vm (mV)
    iNa (mA/cm2)
    iKd (mA/cm2)
    iLeak (mA/cm2)
    A_t  (kPa)
    y
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphal(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betal(A(kPa), Q(nC/cm2)) (/ms)

INCLUDE "update.inc"

INITIAL {
    update()
    m = alpham(A_t, y) / (alpham(A_t, y) + betam(A_t, y))
    h = alphah(A_t, y) / (alphah(A_t, y) + betah(A_t, y))
    n = alphan(A_t, y) / (alphan(A_t, y) + betan(A_t, y))
    l = alphal(A_t, y) / (alphal(A_t, y) + betal(A_t, y))
}

BREAKPOINT {
    update()
    SOLVE states METHOD cnexp
    iNa = gNabar * m * m * m * h * (Vm - ENa)
    iKd = gKdbar * n * n * n * l * (Vm - EK)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(A_t, y) * (1 - m) - betam(A_t, y) * m
    h' = alphah(A_t, y) * (1 - h) - betah(A_t, y) * h
    n' = alphan(A_t, y) * (1 - n) - betan(A_t, y) * n
    l' = alphal(A_t, y) * (1 - l) - betal(A_t, y) * l
}