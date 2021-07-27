TITLE MRGnode membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Mammalian myelinated fiber node.
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2020-03-05
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX MRGnodeauto
    NONSPECIFIC_CURRENT iNaf : fast Sodium current.
    NONSPECIFIC_CURRENT iNap : persistent Sodium current.
    NONSPECIFIC_CURRENT iKs : slow Potassium current
    NONSPECIFIC_CURRENT iLeak : non-specific leakage current
    RANGE Adrive, Vm, y, Fdrive, A_t : section specific
    RANGE stimon, detailed    : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
    stimon       : Stimulation state
    Fdrive (kHz) : Stimulation frequency
    Adrive (kPa) : Stimulation amplitude
    detailed     : Simulation type
    gNafbar = 3.0 (S/cm2)
    ENa = 50.0 (mV)
    gNapbar = 0.01 (S/cm2)
    gKsbar = 0.08 (S/cm2)
    EK = -90.0 (mV)
    gLeak = 0.007 (S/cm2)
    ELeak = -90.0 (mV)
}

STATE {
    m : iNaf activation gate
    h : iNaf inactivation gate
    p : iNap activation gate
    s : iKs activation gate
}

ASSIGNED {
    v  (nC/cm2)
    Vm (mV)
    iNaf (mA/cm2)
    iNap (mA/cm2)
    iKs (mA/cm2)
    iLeak (mA/cm2)
    A_t  (kPa)
    y
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphas(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betas(A(kPa), Q(nC/cm2)) (/ms)

INCLUDE "update.inc"

INITIAL {
    update()
    m = alpham(A_t, y) / (alpham(A_t, y) + betam(A_t, y))
    h = alphah(A_t, y) / (alphah(A_t, y) + betah(A_t, y))
    p = alphap(A_t, y) / (alphap(A_t, y) + betap(A_t, y))
    s = alphas(A_t, y) / (alphas(A_t, y) + betas(A_t, y))
}

BREAKPOINT {
    update()
    SOLVE states METHOD cnexp
    iNaf = gNafbar * m * m * m * h * (Vm - ENa)
    iNap = gNapbar * p * p * p * (Vm - ENa)
    iKs = gKsbar * s * (Vm - EK)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(A_t, y) * (1 - m) - betam(A_t, y) * m
    h' = alphah(A_t, y) * (1 - h) - betah(A_t, y) * h
    p' = alphap(A_t, y) * (1 - p) - betap(A_t, y) * p
    s' = alphas(A_t, y) * (1 - s) - betas(A_t, y) * s
}