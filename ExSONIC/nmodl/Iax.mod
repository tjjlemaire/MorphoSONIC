TITLE Axial current as density mechanism

COMMENT
Implementation of the contribution of axial current at a node connected in series with
two neighboring nodes, using POINTERS towards a reference voltage variable at the central
and neighboring nodes.

@Author: Theo Lemaire, EPFL
@Date:   2018-08-21
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

NEURON  {
    SUFFIX Iax
    NONSPECIFIC_CURRENT iax
    RANGE R, Rprev, Rnext, Am
    POINTER V, Vprev, Vnext
}

PARAMETER {
    R       (ohm)
    Rprev   (ohm)
    Rnext   (ohm)
    Am      (cm2)
}

ASSIGNED {
    V       (mV)
    Vprev   (mV)
    Vnext   (mV)
    iax     (mA/cm2)
}

BREAKPOINT {
    iax = 2 / Am * ((V - Vprev) / (R + Rprev) + (V - Vnext) / (R + Rnext))
}