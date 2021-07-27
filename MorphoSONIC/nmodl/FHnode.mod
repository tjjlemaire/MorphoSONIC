TITLE FH membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Xenopus myelinated fiber node
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-07-05
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

CONSTANT {
    Z_Na = 1
    FARADAY = 96485.3
    Rg = 8.31342
    Z_K = 1
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX FHnodeauto
    NONSPECIFIC_CURRENT iNa : Sodium current
    NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
    NONSPECIFIC_CURRENT iP : non-specific delayed current
    NONSPECIFIC_CURRENT iLeak : non-specific leakage current
    RANGE Adrive, Vm, y, Fdrive, A_t : section specific
    RANGE stimon, detailed     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
    stimon       : Stimulation state
    Fdrive (kHz) : Stimulation frequency
    Adrive (kPa) : Stimulation amplitude
    detailed     : Simulation type
    pNabar = 8e-09 (10 m/ms)
    Nai = 0.01374 (M)
    Nao = 0.1145 (M)
    T = 293.15 ()
    pKbar = 1.2e-09 (10 m/ms)
    Ki = 0.12 (M)
    Ko = 0.0025 (M)
    pPbar = 5.4e-10 (10 m/ms)
    gLeak = 0.03003 (S/cm2)
    ELeak = -69.974 (mV)
}

STATE {
    m : iNa activation gate
    h : iNa inactivation gate
    n : iKd gate
    p : iP gate
}

ASSIGNED {
    v  (nC/cm2)
    Vm (mV)
    iNa (mA/cm2)
    iKd (mA/cm2)
    iP (mA/cm2)
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
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2)) (/ms)

FUNCTION efun(x) {
    efun = x / (exp(x) - 1)
}

FUNCTION ghkDrive(Vm, Z_ion, Cion_in, Cion_out, T) {
    LOCAL x, eCin, eCout
    x = Z_ion * FARADAY * Vm / (Rg * T) * 1e-3
    eCin = Cion_in * efun(-x)
    eCout = Cion_out * efun(x)
    ghkDrive = FARADAY * (eCin - eCout) * 1e6
}

INCLUDE "update.inc"

INITIAL {
    update()
    m = alpham(A_t, y) / (alpham(A_t, y) + betam(A_t, y))
    h = alphah(A_t, y) / (alphah(A_t, y) + betah(A_t, y))
    n = alphan(A_t, y) / (alphan(A_t, y) + betan(A_t, y))
    p = alphap(A_t, y) / (alphap(A_t, y) + betap(A_t, y))
}

BREAKPOINT {
    update()
    SOLVE states METHOD cnexp
    iNa = pNabar * m * m * h * ghkDrive(Vm, Z_Na, Nai, Nao, T)
    iKd = pKbar * n * n * ghkDrive(Vm, Z_K, Ki, Ko, T)
    iP = pPbar * p * p * ghkDrive(Vm, Z_Na, Nai, Nao, T)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(A_t, y) * (1 - m) - betam(A_t, y) * m
    h' = alphah(A_t, y) * (1 - h) - betah(A_t, y) * h
    n' = alphan(A_t, y) * (1 - n) - betan(A_t, y) * n
    p' = alphap(A_t, y) * (1 - p) - betap(A_t, y) * p
}