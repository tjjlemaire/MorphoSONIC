TITLE STN neuron membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a sub-thalamic nucleus neuron upon ultrasonic
stimulation, based on the multi-Scale Optmimized Neuronal Intramembrane Cavitation (SONIC) model.

@Author: Theo Lemaire, EPFL
@Date:   2019-03-05
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
    SUFFIX STN

    : Constituting currents
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iA
    NONSPECIFIC_CURRENT iCaT
    NONSPECIFIC_CURRENT iCaL
    NONSPECIFIC_CURRENT iKCa
    NONSPECIFIC_CURRENT iLeak

    : RANGE variables
    RANGE Adrive, Vmeff : section specific
    RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)
}


CONSTANT {
    FARADAY = 96494     (coul)     : moles do not appear in units
    R = 8.31342         (J/mol/K)  : Universal gas constant
}


PARAMETER {
    : Parameters set by python/hoc caller
    stimon : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    : membrane properties
    cm = 1              (uF/cm2)
    ena = 60            (mV)
    ek = -90            (mV)
    eleak = -60.0       (mV)
    gnabar = 0.049     (S/cm2)
    gkdbar = 0.057     (S/cm2)
    gAbar = 0.005      (S/cm2)
    gcaTbar = 0.005    (S/cm2)
    gcaLbar = 0.015    (S/cm2)
    gkcabar = 0.001    (S/cm2)
    gleak = 1.9e-5     (S/cm2)

    thetax_d2 = 1e-7   (M)
    kx_d2 = 2e-8       (M)
    thetax_r = 1.7e-7  (M)
    kx_r = -8e-8       (M)

    tau_d2 = 130       (ms)
    tau_r = 2          (ms)

    C_Ca0 = 5e-9       (M)
    C_Ca_out = 2e-3    (M)
    KCa = 2            (1/ms)

    T = 306.15         (K)

    depth              (m)  : depth of sub-membrane space
    i2CCa  :conversion factor from Calcium current (mA/cm2) to Calcium concentration derivative (M/ms)
    eca                (mV)
}

STATE {
    : Gating states
    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    a  : iA activation gate
    b  : iA inactivation gate
    p  : iCaT activation gate
    q  : iCaT inactivation gate
    c  : iCaL activation gate
    d1 : iCaL inactivation gate
    d2 : iCaL inactivation gate
    r  : iCaK activation gate

    C_Ca (M) : submembrane Calcium concentration
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Vmeff   (mV)
    v       (mV)
    iNa     (mA/cm2)
    iKd     (mA/cm2)
    iA      (mA/cm2)
    iCaT    (mA/cm2)
    iCaL    (mA/cm2)
    iKCa    (mA/cm2)
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
FUNCTION_TABLE alphaa(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betaa(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphab(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betab(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphaq(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betaq(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphac(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betac(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphad1(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betad1(A(kPa), Q(nC/cm2))   (/ms)


FUNCTION xinf(var, theta, k) {
    xinf = 1 / (1 + exp((var - theta) / k))
}

FUNCTION d2inf(Cai (M)) {
    d2inf = xinf(Cai, thetax_d2, kx_d2)
}

FUNCTION rinf(Cai (M)) {
    rinf = xinf(Cai, thetax_r, kx_r)
}

FUNCTION nernst(z, Cin (M), Cout (M), T (K)) (mV) {
    nernst = (R * T) / (z * FARADAY) * log(Cout / Cin) * 1e3
}


INITIAL {
    : Set initial states values
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = alphap(0, v) / (alphap(0, v) + betap(0, v))
    q = alphaq(0, v) / (alphaq(0, v) + betaq(0, v))
    a = alphaa(0, v) / (alphaa(0, v) + betaa(0, v))
    b = alphab(0, v) / (alphab(0, v) + betab(0, v))
    c = alphac(0, v) / (alphac(0, v) + betac(0, v))
    d1 = alphad1(0, v) / (alphad1(0, v) + betad1(0, v))
    d2 = d2inf(C_Ca0)
    r = rinf(C_Ca0)
    C_Ca = C_Ca0

    eca = nernst(2, C_Ca0, C_Ca_out, T)
    Vmeff = V(0, v)
    iCaT = gcaTbar * p * p * q * (Vmeff - eca)
    iCaL = gcaLbar * c * c * d1 * d2 * (Vmeff - eca)
    depth = -(iCaT + iCaL) / (2 * FARADAY * KCa * C_Ca) * 1e-5
    i2CCa = 1e-5 / (2 * depth * FARADAY)
}

BREAKPOINT {
    : Integrate states
    SOLVE states METHOD cnexp

    : Compute effective membrane potential
    Vmeff = V(Adrive * stimon, v)

    : Compute Calcium reversal potential
    eca = nernst(2, C_Ca, C_Ca_out, T)

    : Compute ionic currents
    iNa = gnabar * m * m * m * h * (Vmeff - ena)
    iKd = gkdbar * n * n * n * n * (Vmeff - ek)
    iA = gAbar * a * a * b * (Vmeff - ek)
    iCaT = gcaTbar * p * p * q * (Vmeff - eca)
    iCaL = gcaLbar * c * c * d1 * d2 * (Vmeff - eca)
    iKCa = gkcabar * r * r * (Vmeff - ek)
    iLeak = gleak * (Vmeff - eleak)
}

DERIVATIVE states {
    : Gating states derivatives
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    p' = alphap(Adrive * stimon, v) * (1 - p) - betap(Adrive * stimon, v) * p
    q' = alphaq(Adrive * stimon, v) * (1 - q) - betaq(Adrive * stimon, v) * q
    a' = alphaa(Adrive * stimon, v) * (1 - a) - betaa(Adrive * stimon, v) * a
    b' = alphab(Adrive * stimon, v) * (1 - b) - betab(Adrive * stimon, v) * b
    c' = alphac(Adrive * stimon, v) * (1 - c) - betac(Adrive * stimon, v) * c
    d1' = alphad1(Adrive * stimon, v) * (1 - d1) - betad1(Adrive * stimon, v) * d1

    d2' = (d2inf(C_Ca) - d2) / tau_d2
    r' = (rinf(C_Ca) - r) / tau_r

    iCaT = gcaTbar * p * p * q * (Vmeff - eca)
    iCaL = gcaLbar * c * c * d1 * d2 * (Vmeff - eca)
    C_Ca' = - i2CCa * (iCaT + iCaL) - C_Ca * KCa
}