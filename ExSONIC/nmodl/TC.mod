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
    RANGE Adrive, fs, Vmeff : section specific
    RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)
}

CONSTANT {
    FARADAY = 96494     (coul) : moles do not appear in units
}


PARAMETER {
    : Parameters set by python/hoc caller
    stimon : Stimulation state
    fs : Fraction of membrane covered by sonophore
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
    : Standard gating states
    m_0  : iNa activation gate
    h_0  : iNa inactivation gate
    n_0  : iKd activation gate
    s_0  : iCa activation gate
    u_0  : iCa inactivation gate
    C1_0  : iH channel closed state
    O1_0  : iH channel open state

    : US-modulated gating state
    m_US  : iNa activation gate
    h_US  : iNa inactivation gate
    n_US  : iKd activation gate
    s_US  : iCa activation gate
    u_US  : iCa inactivation gate
    C1_US  : iH channel closed state
    O1_US  : iH channel open state

    C_Ca_0 (M) : submembrane Calcium concentration
    P0_0       : proportion of unbound CB protein
    C_Ca_US (M) : submembrane Calcium concentration
    P0_US       : proportion of unbound CB protein
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Vmeff_0   (mV)
    Vmeff_US  (mV)
    Vmeff   (mV)
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
    m_0 = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h_0 = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n_0 = alphan(0, v) / (alphan(0, v) + betan(0, v))
    s_0 = alphas(0, v) / (alphas(0, v) + betas(0, v))
    u_0 = alphau(0, v) / (alphau(0, v) + betau(0, v))

    m_US = m_0
    h_US = h_0
    n_US = n_0
    s_US = s_0
    u_US = u_0

    : Compute steady-state Calcium concentration
    iCa = gcabar * s_0 * s_0 * u_0 * (V(0, v) - eca)
    C_Ca_0 = camin + taur * iondrive(iCa, 2, depth)

    C_Ca_US = C_Ca_0

    : Compute steady values for the kinetics system of Ih
    P0_0 = k2 / (k2 + k1 * npow(C_Ca_0, nca))
    O1_0 = k4 / (k3 * (1 - P0_0) + k4 * (1 + betao(0, v) / alphao(0, v)))
    C1_0 = betao(0, v) / alphao(0, v) * O1_0

    O1_US = O1_0
    C1_US = C1_0
    P0_US = P0_0
}

BREAKPOINT {
    : Integrate states
    SOLVE states METHOD cnexp

    : Check iH states and restrict them if needed
    if(O1_0 < 0.) {O1_0 = 0.}
    if(O1_0 > 1.) {O1_0 = 1.}
    if(C1_0 < 0.) {C1_0 = 0.}
    if(C1_0 > 1.) {C1_0 = 1.}

    if(O1_US < 0.) {O1_US = 0.}
    if(O1_US > 1.) {O1_US = 1.}
    if(C1_US < 0.) {C1_US = 0.}
    if(C1_US > 1.) {C1_US = 1.}

    : Compute effective membrane potential
    Vmeff_0 = V(0, v)
    Vmeff_US = V(Adrive * stimon, v)
    Vmeff = (1 - fs) * Vmeff_0 + fs * Vmeff_US

    : compute ionic currents
    iNa = gnabar * ((1 - fs) * m_0 * m_0 * m_0 * h_0 * (Vmeff_0 - ena) + fs * m_US * m_US * m_US * h_US * (Vmeff_US - ena))
    iKd = gkdbar * ((1 - fs) * n_0 * n_0 * n_0 * n_0 * (Vmeff_0 - ek) + fs * n_US * n_US * n_US * n_US * (Vmeff_US - ek))
    iKl = gkl * ((1 - fs) * (Vmeff_0 - ek) + fs * (Vmeff_US - ek))
    iCa = gcabar * ((1 - fs) * s_0 * s_0 * u_0 * (Vmeff_0 - eca) + fs * s_US * s_US * u_US * (Vmeff_US - eca))
    iH = ghbar * ((1 - fs) * (O1_0 + 2 * (1 - O1_0 - C1_0)) * (Vmeff_0 - eh) + fs * (O1_US + 2 * (1 - O1_US - C1_US)) * (Vmeff_US - eh))
    iLeak = gleak * ((1 - fs) * (Vmeff_0 - eleak) + fs * (Vmeff_US - eleak))
}

DERIVATIVE states {
    : Standard states derivatives
    m_0' = alpham(0, v) * (1 - m_0) - betam(0, v) * m_0
    h_0' = alphah(0, v) * (1 - h_0) - betah(0, v) * h_0
    n_0' = alphan(0, v) * (1 - n_0) - betan(0, v) * n_0
    s_0' = alphas(0, v) * (1 - s_0) - betas(0, v) * s_0
    u_0' = alphau(0, v) * (1 - u_0) - betau(0, v) * u_0

    : US-modulated states derivatives
    m_US' = alpham(Adrive * stimon, v) * (1 - m_US) - betam(Adrive * stimon, v) * m_US
    h_US' = alphah(Adrive * stimon, v) * (1 - h_US) - betah(Adrive * stimon, v) * h_US
    n_US' = alphan(Adrive * stimon, v) * (1 - n_US) - betan(Adrive * stimon, v) * n_US
    s_US' = alphas(Adrive * stimon, v) * (1 - s_US) - betas(Adrive * stimon, v) * s_US
    u_US' = alphau(Adrive * stimon, v) * (1 - u_US) - betau(Adrive * stimon, v) * u_US

    : Compute derivatives of variables for the kinetics system of Ih
    C_Ca_0' = (camin - C_Ca_0) / taur + iondrive(iCa, 2, depth)
    P0_0' = k2 * (1 - P0_0) - k1 * P0_0 * npow(C_Ca_0, nca)
    C1_0' = betao(0, v) * O1_0 - alphao(0, v) * C1_0
    O1_0' = alphao(0, v) * C1_0 - betao(0, v) * O1_0 - k3 * O1_0 * (1 - P0_0) + k4 * (1 - O1_0 - C1_0)

    C_Ca_US' = (camin - C_Ca_US) / taur + iondrive(iCa, 2, depth)
    P0_US' = k2 * (1 - P0_US) - k1 * P0_US * npow(C_Ca_US, nca)
    C1_US' = betao(Adrive * stimon, v) * O1_US - alphao(Adrive * stimon, v) * C1_US
    O1_US' = alphao(Adrive * stimon, v) * C1_US - betao(Adrive * stimon, v) * O1_US - k3 * O1_US * (1 - P0_US) + k4 * (1 - O1_US - C1_US)
}


