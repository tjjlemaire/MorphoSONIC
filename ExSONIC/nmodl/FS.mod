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

    : Constituting currents
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iM
    NONSPECIFIC_CURRENT iLeak

    : RANGE variables
    RANGE Adrive, fs, Vmeff : section specific
    RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
    : Parameters set by python/hoc caller
    stimon : Stimulation state
    fs : Fraction of membrane covered by sonophore
    Adrive (kPa) : Stimulation amplitude

    : membrane properties
    cm = 1              (uF/cm2)
    ena = 50            (mV)
    ek = -90            (mV)
    eleak = -70.4       (mV)
    gnabar = 0.058      (S/cm2)
    gkdbar = 0.0039     (S/cm2)
    gmbar = 7.87e-5     (S/cm2)
    gleak = 3.8e-5      (S/cm2)
}

STATE {
    : Standard gating states
    m_0  : iNa activation gate
    h_0  : iNa inactivation gate
    n_0  : iKd activation gate
    p_0  : iM activation gate

    : US-modulated gating state
    m_US  : iNa activation gate
    h_US  : iNa inactivation gate
    n_US  : iKd activation gate
    p_US  : iM activation gate
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Vmeff_0   (mV)
    Vmeff_US  (mV)
    Vmeff     (mV)
    v         (mV)
    iNa       (mA/cm2)
    iKd       (mA/cm2)
    iM        (mA/cm2)
    iLeak     (mA/cm2)
}

: Function tables to interpolate effective variables
FUNCTION_TABLE V(A(kPa), Q(nC/cm2))      (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2))   (/ms)
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2))  (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2))   (/ms)

INITIAL {
    : Set initial states values
    m_0 = alpham(0, v) / (alpham(0, v) + betam(0, v))
    h_0 = alphah(0, v) / (alphah(0, v) + betah(0, v))
    n_0 = alphan(0, v) / (alphan(0, v) + betan(0, v))
    p_0 = alphap(0, v) / (alphap(0, v) + betap(0, v))

    m_US = m_0
    h_US = h_0
    n_US = n_0
    p_US = p_0
}

BREAKPOINT {
    : Integrate states
    SOLVE states METHOD cnexp

    : Compute effective membrane potentials
    Vmeff_0 = V(0, v)
    Vmeff_US = V(Adrive * stimon, v)
    Vmeff = (1 - fs) * Vmeff_0 + fs * Vmeff_US

    : Compute ionic currents
    iNa = gnabar * ((1 - fs) * m_0 * m_0 * m_0 * h_0 * (Vmeff_0 - ena) + fs * m_US * m_US * m_US * h_US * (Vmeff_US - ena))
    iKd = gkdbar * ((1 - fs) * n_0 * n_0 * n_0 * n_0 * (Vmeff_0 - ek) + fs * n_US * n_US * n_US * n_US * (Vmeff_US - ek))
    iM = gmbar * ((1 - fs) * p_0 * (Vmeff_0 - ek) + fs * p_US * (Vmeff_US - ek))
    iLeak = gleak * ((1 - fs) * (Vmeff_0 - eleak) + fs * (Vmeff_US - eleak))
}

DERIVATIVE states {
    : Standard states derivatives
    m_0' = alpham(0, v) * (1 - m_0) - betam(0, v) * m_0
    h_0' = alphah(0, v) * (1 - h_0) - betah(0, v) * h_0
    n_0' = alphan(0, v) * (1 - n_0) - betan(0, v) * n_0
    p_0' = alphap(0, v) * (1 - p_0) - betap(0, v) * p_0

    : US-modulated states derivatives
    m_US' = alpham(Adrive * stimon, v) * (1 - m_US) - betam(Adrive * stimon, v) * m_US
    h_US' = alphah(Adrive * stimon, v) * (1 - h_US) - betah(Adrive * stimon, v) * h_US
    n_US' = alphan(Adrive * stimon, v) * (1 - n_US) - betan(Adrive * stimon, v) * n_US
    p_US' = alphap(Adrive * stimon, v) * (1 - p_US) - betap(Adrive * stimon, v) * p_US
}