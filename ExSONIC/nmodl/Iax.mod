TITLE Axial current as point process

COMMENT
Implementation of the contribution of axial current at a node connected with
a given number of neighboring nodes, using POINTERS towards a reference voltage variable
at the central and neighboring nodes.

@Author: Theo Lemaire, EPFL
@Date:   2018-08-21
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

DEFINE MAX_CON 2  : max number of axial connections

NEURON  {
    POINT_PROCESS Iax
    NONSPECIFIC_CURRENT iax
    RANGE R, Rother
    POINTER V, Vother, Vext, Vextother
}

UNITS {
    (nA) = (nanoamp)
}

PARAMETER {
    R                   (ohm)
    Rother[MAX_CON]     (ohm)
}

ASSIGNED {
    V          (mV)
    Vother     (mV)
    Vext       (mV)
    Vextother  (mV)
    iax        (nA)
}

BREAKPOINT {
    iax = 0
    FROM i=0 TO MAX_CON-1 {
        iax = iax + ((V + Vext) - (get_Vother(i) + get_Vextother(i))) / (R + Rother[i])
    }
    iax = 2 * iax * 1e6
}

INCLUDE "Vother.inc"
INCLUDE "Vextother.inc"