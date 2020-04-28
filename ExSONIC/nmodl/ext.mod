TITLE Custom extracellular mechanism as point process

COMMENT
Custom implementation of the extracellular mechanism.

@Author: Theo Lemaire, EPFL
@Date:   2020-04-22
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

DEFINE MAX_CON 2  : max number of axial connections

NEURON  {
    POINT_PROCESS custom_extracellular
    RANGE xg, xc, xr, xrother0, xrother1, e_extracellular, NLEVELS
    POINTER Vother, Vextother, iax
}

CONSTANT {
    MA_CM2_TO_UA_CM2 = 1e3
}

PARAMETER {
    NLEVELS
    xg[2]              (S/cm2)
    xc[2]              (uF/cm2)
    xr[2]              (ohm)
    xrother0[MAX_CON]  (ohm)
    xrother1[MAX_CON]  (ohm)
    e_extracellular    (mV)
    Am                 (cm2)
}

ASSIGNED {
    Vother         (mV)
    Vextother      (mV)
    iax            (nA)
    iaxdensity     (mA/cm2)
    ixraxial[2]    (mA/cm2)
    itransverse[2] (mA/cm2)
}

STATE {
    V0      (mV)
    V1      (mV)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
}

INITIAL {
    if (NLEVELS == 0) {
        V0 = e_extracellular
        V1 = 0.
    }
    else {
        if (NLEVELS == 1) {
            V0 = e_extracellular
            :V0 = e_extracellular + (ixraxial[0]) / xg[0]
            V1 = 0
        }
        else {
            V0 = 0
            V1 = 0
        }
    }
}

DERIVATIVE states {
    if (NLEVELS == 0) {
        currents0()
        V1' = 0
        V0' = (e_extracellular - V0) * 1e6 :-itransverse[0] * 1e3
    } else {
        if (NLEVELS == 1) {
            currents1()
            :V0' = (iaxdensity + ixraxial[0] + itransverse[0]) / xc[0] * 1e3
            V1' = 0
            V0' = -itransverse[0] * 1e3
            :V0' = -xg[0] * (V0 - e_extracellular) / xc[0] * 1e-3
        }
        else {
            currents2()
            V1' = (iaxdensity + ixraxial[0] + ixraxial[1] + itransverse[1]) / xc[1] * 1e3
            V0' = (iaxdensity + ixraxial[0] + itransverse[0]) / xc[0] * 1e3 + V1'
        }
    }
}


PROCEDURE currents0() {
    itransverse[0] = xg[0] * (V0 - e_extracellular)
    iaxdensity = 0
    ixraxial[0] = 0
    ixraxial[1] = 0
    itransverse[1] = 0
}

PROCEDURE currents1() {
    itransverse[0] = xg[0] * (V0 - e_extracellular)
    iaxdensity = iax * 1e-6 / Am
    ixraxial[0] = 0
    FROM i=0 TO MAX_CON-1 {
        ixraxial[0] = ixraxial[0] + (V0 - get_Vother(i)) / (xr[0] + xrother0[i])
    }
    ixraxial[0] = 2 * ixraxial[0] / Am
    ixraxial[1] = 0
    itransverse[1] = 0
}

PROCEDURE currents2() {
    itransverse[0] = xg[0] * (V0 - V1)
    iaxdensity = iax * 1e-6 / Am
    ixraxial[0] = 0
    ixraxial[1] = 0
    FROM i=0 TO MAX_CON-1 {
        ixraxial[0] = ixraxial[0] + (V0 - get_Vother(i)) / (xr[0] + xrother0[i])
        ixraxial[1] = ixraxial[1] + (V1 - get_Vextother(i)) / (xr[1] + xrother1[i])
    }
    ixraxial[0] = 2 * ixraxial[0] / Am
    ixraxial[1] = 2 * ixraxial[1] / Am
    itransverse[1] = xg[1] * (V1 - e_extracellular)
}


INCLUDE "Vother.inc"
INCLUDE "Vextother.inc"