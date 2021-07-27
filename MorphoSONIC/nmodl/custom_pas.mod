TITLE Custom passive current

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX custom_pas
    NONSPECIFIC_CURRENT i : passive leakage current
    RANGE g, e
    RANGE Adrive, Vm, y, Fdrive, A_t : section specific
    RANGE stimon, detailed    : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
    stimon       : Stimulation state
    Fdrive (kHz) : Stimulation frequency
    Adrive (kPa) : Stimulation amplitude
    detailed     : Simulation type
    g      (S/cm2)
    e      (mV)
}

ASSIGNED {
    v   (nC/cm2)
    Vm  (mV)
    i   (mA/cm2)
    A_t  (kPa)
    y
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)

INCLUDE "update.inc"

BREAKPOINT {
    update()
    i = g * (Vm - e)
}