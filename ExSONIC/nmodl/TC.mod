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

    : Constituting currents
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iKl
    NONSPECIFIC_CURRENT iH
    NONSPECIFIC_CURRENT iCa
    NONSPECIFIC_CURRENT iLeak

    : RANGE variables
    RANGE Adrive, Vmeff : section specific
    RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)
}

CONSTANT {
    FARADAY = 96494     (coul) : moles do not appear in units
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
    eh = -40            (mV)
    eleak = -70         (mV)
    gnabar = 0.09       (S/cm2)
    gkdbar = 0.01       (S/cm2)
    gkl = 1.38e-5       (S/cm2)
    gcabar = 0.002      (S/cm2)
    ghbar = 1.75e-5     (S/cm2)
    gleak = 1e-5        (S/cm2)

    : iH Calcium dependence properties
    k1 = 2.5e19         (1/M*M*M*M*ms)    : CB protein Calcium-driven activation rate
    k2 = 0.0004         (1/ms)            : CB protein inactivation rate
    k3 = 0.1            (1/ms)            : CB protein iH channel binding rate
    k4  = 0.001         (1/ms)            : CB protein iH channel unbinding rate
    nca = 4                               : number of Calcium binding sites on CB protein

    : submembrane Calcium evolution properties
    depth = 1e-7        (m)   : depth of shell
    taur = 5            (ms)   : rate of calcium removal
    camin = 5e-8        (M)   : minimal intracellular Calcium concentration

}

STATE {
    : Gating states
    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    s  : iCa activation gate
    u  : iCa inactivation gate
    C1  : iH channel closed state
    O1  : iH channel open state

    C_Ca (M) : submembrane Calcium concentration
    P0       : proportion of unbound CB protein
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Vmeff    (mV)
    v        (mV)
    iNa      (mA/cm2)
    iKd      (mA/cm2)
    iKl      (mA/cm2)
    iCa      (mA/cm2)
    iH       (mA/cm2)
    iLeak    (mA/cm2)
}

: Function tables to interpolate effective variables
FUNCTION_TABLE V(A(kPa), Q(nC/cm2))      (mV)
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
    : Set initial states values
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    s = alphas(0, v) / (alphas(0, v) + betas(0, v))
    u = alphau(0, v) / (alphau(0, v) + betau(0, v))

    : Compute steady-state Calcium concentration
    iCa = gcabar * s * s * u * (V(0, v) - eca)
    C_Ca = camin + taur * iondrive(iCa, 2, depth)

    : Compute steady values for the kinetics system of Ih
    P0 = k2 / (k2 + k1 * npow(C_Ca, nca))
    O1 = k4 / (k3 * (1 - P0) + k4 * (1 + betao(0, v) / alphao(0, v)))
    C1 = betao(0, v) / alphao(0, v) * O1
}

BREAKPOINT {
    : Integrate states
    SOLVE states METHOD cnexp

    : Check iH states and restrict them if needed
    if(O1 < 0.) {O1 = 0.}
    if(O1 > 1.) {O1 = 1.}
    if(C1 < 0.) {C1 = 0.}
    if(C1 > 1.) {C1 = 1.}

    : Compute effective membrane potential
    Vmeff = V(Adrive * stimon, v)

    : compute ionic currents
    iNa = gnabar * m * m * m * h * (Vmeff - ena)
    iKd = gkdbar * n * n * n * n * (Vmeff - ek)
    iKl = gkl * (Vmeff - ek)
    iCa = gcabar * s * s * u * (Vmeff - eca)
    iH = ghbar * (O1 + 2 * (1 - O1 - C1)) * (Vmeff - eh)
    iLeak = gleak * (Vmeff - eleak)
}

DERIVATIVE states {
    : Gating states derivatives
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    s' = alphas(Adrive * stimon, v) * (1 - s) - betas(Adrive * stimon, v) * s
    u' = alphau(Adrive * stimon, v) * (1 - u) - betau(Adrive * stimon, v) * u

    : Compute derivatives of variables for the kinetics system of Ih
    C_Ca' = (camin - C_Ca) / taur + iondrive(iCa, 2, depth)
    P0' = k2 * (1 - P0) - k1 * P0 * npow(C_Ca, nca)
    C1' = betao(Adrive * stimon, v) * O1 - alphao(Adrive * stimon, v) * C1
    O1' = alphao(Adrive * stimon, v) * C1 - betao(Adrive * stimon, v) * O1 - k3 * O1 * (1 - P0) + k4 * (1 - O1 - C1)
}


