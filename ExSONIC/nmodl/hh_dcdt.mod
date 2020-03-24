TITLE HH varying capacitance

COMMENT
modified to take into account sodium gating capacitance (perhaps wrongly)
But provides an example of how to manage variable capacitance using
the relation q = c*v so that i = (c*dv/dt) + (dc/dt * v)
The effect of changing capacitance on the (c*dv/dt) term is accomplished
via a POINTER to the compartment cm (set up in hoc) where the c pointer
is assigned a value in the BEFORE BREAKPOINT block. The effect of
the (dc/dt * v) term is accomplished in the BREAKPOINT block.
To allow a better undertanding of the role of the two terms of d(c*v)/dt,
the flag use_dc_dt can be used to turn the second term on or off.

SUFFIX hhdcdt
POINTER c

ASSIGNED {
    c   (uF/cm2)
    idc (mA/cm2)
}

FUNCTION_TABLE c(A, Q) (mV)

INITIAL{
    idc = 0
}

BEFORE BREAKPOINT {
    c = c(A, Q)
}

BREAKPOINT {
    idc = dcdt * v * 0.001
    iLeak = iLeak + idc
}

ENDCOMMENT