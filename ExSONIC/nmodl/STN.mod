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

    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iA
    NONSPECIFIC_CURRENT iCaT
    NONSPECIFIC_CURRENT iCaL
    NONSPECIFIC_CURRENT iKCa
    NONSPECIFIC_CURRENT iLeak

    RANGE Adrive, Vm : section specific
    RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}


CONSTANT {
    FARADAY = 96494     (coul)     : moles do not appear in units
    R = 8.31342         (J/mol/K)  : Universal gas constant
}


PARAMETER {
    stimon       : Stimulation state
    Adrive (kPa) : Stimulation amplitude

    cm = 1              (uF/cm2)
    ENa = 60            (mV)
    EK = -90            (mV)
    ELeak = -60.0       (mV)
    gNabar = 0.049     (S/cm2)
    gKdbar = 0.057     (S/cm2)
    gAbar = 0.005      (S/cm2)
    gCaTbar = 0.005    (S/cm2)
    gCaLbar = 0.015    (S/cm2)
    gKCabar = 0.001    (S/cm2)
    gLeak = 3.5e-4     (S/cm2)

    thetax_d2 = 1e-7   (M)
    kx_d2 = 2e-8       (M)
    thetax_r = 1.7e-7  (M)
    kx_r = -8e-8       (M)
    tau_d2 = 130       (ms)
    tau_r = 2          (ms)
    Cai0 = 5e-9        (M)
    Cao = 2e-3         (M)
    KCa = 2            (1/ms)
    T = 306.15         (K)
    depth              (m)  : depth of sub-membrane space
    iCa2Cai                 :conversion factor from iCa (mA/cm2) to Cai derivative (M/ms)
    ECa                (mV)
}

STATE {
    m       : iNa activation gate
    h       : iNa inactivation gate
    n       : iKd activation gate
    a       : iA activation gate
    b       : iA inactivation gate
    p       : iCaT activation gate
    q       : iCaT inactivation gate
    c       : iCaL activation gate
    d1      : iCaL inactivation gate
    d2      : iCaL inactivation gate
    r       : iCaK activation gate
    Cai (M) : submembrane Calcium concentration
}

ASSIGNED {
    Vm      (mV)
    v       (mV)
    iNa     (mA/cm2)
    iKd     (mA/cm2)
    iA      (mA/cm2)
    iCaT    (mA/cm2)
    iCaL    (mA/cm2)
    iKCa    (mA/cm2)
    iLeak   (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2))       (mV)
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
    m = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p = alphap(0, v) / (alphap(0, v) + betap(0, v))
    q = alphaq(0, v) / (alphaq(0, v) + betaq(0, v))
    a = alphaa(0, v) / (alphaa(0, v) + betaa(0, v))
    b = alphab(0, v) / (alphab(0, v) + betab(0, v))
    c = alphac(0, v) / (alphac(0, v) + betac(0, v))
    d1 = alphad1(0, v) / (alphad1(0, v) + betad1(0, v))
    d2 = d2inf(Cai0)
    r = rinf(Cai0)
    Cai = Cai0

    ECa = nernst(2, Cai0, Cao, T)
    Vm = V(0, v)
    iCaT = gCaTbar * p * p * q * (Vm - ECa)
    iCaL = gCaLbar * c * c * d1 * d2 * (Vm - ECa)
    depth = -(iCaT + iCaL) / (2 * FARADAY * KCa * Cai) * 1e-5
    iCa2Cai = 1e-5 / (2 * depth * FARADAY)
    :printf("depth = %.2e, iCa2Cai = %.2e \n", depth, iCa2Cai)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    Vm = V(Adrive * stimon, v)
    ECa = nernst(2, Cai, Cao, T)
    iNa = gNabar * m * m * m * h * (Vm - ENa)
    iKd = gKdbar * n * n * n * n * (Vm - EK)
    iA = gAbar * a * a * b * (Vm - EK)
    iCaT = gCaTbar * p * p * q * (Vm - ECa)
    iCaL = gCaLbar * c * c * d1 * d2 * (Vm - ECa)
    iKCa = gKCabar * r * r * (Vm - EK)
    iLeak = gLeak * (Vm - ELeak)
}

DERIVATIVE states {
    m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
    h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
    n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
    p' = alphap(Adrive * stimon, v) * (1 - p) - betap(Adrive * stimon, v) * p
    q' = alphaq(Adrive * stimon, v) * (1 - q) - betaq(Adrive * stimon, v) * q
    a' = alphaa(Adrive * stimon, v) * (1 - a) - betaa(Adrive * stimon, v) * a
    b' = alphab(Adrive * stimon, v) * (1 - b) - betab(Adrive * stimon, v) * b
    c' = alphac(Adrive * stimon, v) * (1 - c) - betac(Adrive * stimon, v) * c
    d1' = alphad1(Adrive * stimon, v) * (1 - d1) - betad1(Adrive * stimon, v) * d1
    d2' = (d2inf(Cai) - d2) / tau_d2
    r' = (rinf(Cai) - r) / tau_r
    ECa = nernst(2, Cai, Cao, T)
    iCaT = gCaTbar * p * p * q * (Vm - ECa)
    iCaL = gCaLbar * c * c * d1 * d2 * (Vm - ECa)
    Cai' = - iCa2Cai * (iCaT + iCaL) - Cai * KCa
}
