TITLE One-layer custom extracellular mechanism

COMMENT
Custom implementation of the extracellular mechanism with one layer (representing the periaxonal space).

From "treeset.c"

The current balance equations are sum of outward currents = sum of inward currents

For an internal node:

cm*dvm/dt + i(vm) = is(vi) + ai_j*(vi_j - vi)

For an external node (one layer):

cx*dvx/dt + gx*(vx - ex) = cm*dvm/dt + i(vm) + ax_j*(vx_j - vx)

where vm = vi - vx, and every term has the dimensions mA/cm2.

The implicit linear form for the compartment internal node is:
cm/dt*(vmnew - vmold) + (di/dvm*(vmnew - vmold) + i(vmold))
    = (dis/dvi*(vinew - viold) + is(viold)) + ai_j*(vinew_j - vinew)

and for the compartment external node is:

cx/dt*(vxnew - vxold) - gx(vxnew - ex)
    = cm/dt*(vmnew - vmold) + (di/dvm*(vmnew - vmold) + i(vmold))
    + ax_j*(vxnew_j - vxnew) = 0

and this forms the matrix equation row (for an internal node):

    (cm/dt + di/dvm - dis/dvi + ai_j)*vinew
    - ai_j*vinew_j
    -(cm/dt + di/dvm)*vxnew
    =
    cm/dt*vmold + di/dvm*vmold - i(vmold) - dis/dvi*viold + is(viold)

where old is present value of vm, vi, or vx and new will be found by solving this matrix
equation (which is of the form G*v = rhs).


cx*dvx/dt + gx*(vx - ex) = iax + iperiax
-> cx/dt * dvx + gx * (vx - ex) = iax + iperiax
-> cx/dt * (vxnew - vxold) + gx * (vxnew - ex) = iax + iperiax
-> cx/dt * vxnew - cx/dt * vxold + gx * vxnew - gx * ex = iax + iperiax
-> (cx/dt + gx) * vxnew - cx/dt * vxold - gx * ex = iax + iperiax
-> (cx/dt + gx) * vxnew = iax + iperiax + gx * ex + cx/dt * vxold
-> (cx/dt + gx) * vxnew = iax + iperiax + gx * ex + cx/dt * vxold
-> vxnew = (iax + iperiax + gx * ex + cx/dt * vxold) / (cx/dt + gx)

@Author: Theo Lemaire, EPFL
@Date:   2020-04-22
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

DEFINE MAX_CON 2  : max number of axial connections

NEURON  {
    POINT_PROCESS ext
    RANGE gax, xg, xc, Am, cm0, e_extracellular, V0
    POINTER Vother, iax
}

PARAMETER {
    gax[MAX_CON]      (S/cm2)
    xg                (S/cm2)
    xc                (mF/cm2)
    Am                (cm2)
    cm0               (uF/cm2)
    e_extracellular   (mV)
}

ASSIGNED {
    V0             (mV)
    tlast          (ms)
    mydt           (ms)
    V0last         (mV)
    Vother         (mV)
    iax            (nA)
    iaxdensity     (mA/cm2)
    ixraxial       (mA/cm2)
    itransverse    (mA/cm2)
}

PROCEDURE updateV0() {
    currents()
    mydt = t - tlast
    tlast = t
    V0last = V0
    V0 = (xc / mydt * V0last + xg * e_extracellular - iaxdensity * cm0 - ixraxial) / (xc / mydt + xg)
    :printf("t = %.3f ms, mydt = %.3f ms, gc = %.3e S/cm2, xg = %.3e S/cm2 \n", t, mydt, xc / mydt, xg)
}

PROCEDURE currents() {
    iaxdensity = iax / Am * 1e-6
    ixraxial = 0
    FROM i=0 TO MAX_CON-1 {
        ixraxial = ixraxial + gax[i] * (V0 - get_Vother(i))
    }
    :ixraxial = bound(ixraxial, 1e-2)
}

BREAKPOINT {
    updateV0()
}

INCLUDE "Vother.inc"
INCLUDE "bound.inc"