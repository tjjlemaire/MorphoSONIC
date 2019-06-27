TITLE TC neuron membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a thalamo-cortical neuron upon ultrasonic
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
    (nC) = (nanocoulomb)
    (kPa) = (kilopascal)

    (molar) = (1/liter)         : moles do not appear in units
    (M)     = (molar)
    (mM)    = (millimolar)
    (um)    = (micron)
    (msM)   = (ms mM)
}

NEURON {
    SUFFIX TC

    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iKl
    NONSPECIFIC_CURRENT iH
    NONSPECIFIC_CURRENT iCaT
    NONSPECIFIC_CURRENT iLeak

    RANGE Adrive, Vm : section specific
    RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

CONSTANT {
    FARADAY = 96494     (coul) : moles do not appear in units
}


PARAMETER {
    stimon       : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    cm = 1              (uF/cm2)
    ENa = 50            (mV)
    ECa = 120           (mV)
    EK = -90            (mV)
    EH = -40            (mV)
    ELeak = -70         (mV)
    gNabar = 0.09       (S/cm2)
    gKdbar = 0.01       (S/cm2)
    gKLeak = 1.38e-5       (S/cm2)
    gCaTbar = 0.002      (S/cm2)
    gHbar = 1.75e-5     (S/cm2)
    gLeak = 1e-5        (S/cm2)

    k1 = 2.5e19         (1/M*M*M*M*ms)    : CB protein Calcium-driven activation rate
    k2 = 0.0004         (1/ms)            : CB protein inactivation rate
    k3 = 0.1            (1/ms)            : CB protein iH channel binding rate
    k4  = 0.001         (1/ms)            : CB protein iH channel unbinding rate
    nca = 4                               : number of Calcium binding sites on CB protein

    depth = 1e-7        (m)   : depth of shell
    taur = 5            (ms)  : rate of calcium removal
    camin = 5e-8        (M)   : minimal intracellular Calcium concentration

}

STATE {
    m        : iNa activation gate
    h        : iNa inactivation gate
    n        : iKd activation gate
    s        : iCaT activation gate
    u        : iCaT inactivation gate
    C1       : iH channel closed state
    O1       : iH channel open state
    P0       : proportion of unbound CB protein
    Cai (M)  : submembrane Calcium concentration
}

ASSIGNED {
    Vm       (mV)
    v        (mV)
    iNa      (mA/cm2)
    iKd      (mA/cm2)
    iKl      (mA/cm2)
    iCaT      (mA/cm2)
    iH       (mA/cm2)
    iLeak    (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2))       (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphas(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betas(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphau(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betau(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphao(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betao(A(kPa), Q(nC/cm2))   (/ms)

FUNCTION npow(x, n) {
    : Raise a quantity to a given power exponent
    npow = x^n
}

FUNCTION iondrive(i (mA/cm2), val, d(nm)) (M/ms) {
    : Compute the change in submembrane ionic concentration resulting from a given ionic current
    iondrive = -1e-5 * i / (val * FARADAY * d)
}

INITIAL {
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    s = alphas(0, v) / (alphas(0, v) + betas(0, v))
    u = alphau(0, v) / (alphau(0, v) + betau(0, v))
    iCaT = gCaTbar * s * s * u * (V(0, v) - ECa)
    Cai = camin + taur * iondrive(iCaT, 2, depth)
    P0 = k2 / (k2 + k1 * npow(Cai, nca))
    O1 = k4 / (k3 * (1 - P0) + k4 * (1 + betao(0, v) / alphao(0, v)))
    C1 = betao(0, v) / alphao(0, v) * O1
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    if(O1 < 0.) {O1 = 0.}
    if(O1 > 1.) {O1 = 1.}
    if(C1 < 0.) {C1 = 0.}
    if(C1 > 1.) {C1 = 1.}
    Vm = V(Adrive * stimon, v)
    iNa = gNabar * m * m * m * h * (Vm - ENa)
    iKd = gKdbar * n * n * n * n * (Vm - EK)
    iKl = gKLeak * (Vm - EK)
    iCaT = gCaTbar * s * s * u * (Vm - ECa)
    iH = gHbar * (O1 + 2 * (1 - O1 - C1)) * (Vm - EH)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    s' = alphas(Adrive * stimon, v) * (1 - s) - betas(Adrive * stimon, v) * s
    u' = alphau(Adrive * stimon, v) * (1 - u) - betau(Adrive * stimon, v) * u
    Cai' = (camin - Cai) / taur + iondrive(iCaT, 2, depth)
    P0' = k2 * (1 - P0) - k1 * P0 * npow(Cai, nca)
    C1' = betao(Adrive * stimon, v) * O1 - alphao(Adrive * stimon, v) * C1
    O1' = alphao(Adrive * stimon, v) * C1 - betao(Adrive * stimon, v) * O1 - k3 * O1 * (1 - P0) + k4 * (1 - O1 - C1)
}


