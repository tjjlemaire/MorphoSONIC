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
    RANGE Gax, cfac
    POINTER V, Vother, Vext, Vextother
}

UNITS {
    (nA) = (nanoamp)
}

PARAMETER {
    Gax[MAX_CON]  (uS)
}

ASSIGNED {
    V             (mV)
    Vother        (mV)
    Vext          (mV)
    Vextother     (mV)
    iax           (nA)
}

BREAKPOINT {
    compute()
}

PROCEDURE compute() {
    iax = 0
    FROM i=0 TO MAX_CON-1 {
        iax = iax + Gax[i] * (V - get_Vother(i) + Vext - get_Vextother(i))
    }
}

PROCEDURE declare_Vother() {
    LOCAL n
    n = MAX_CON
    VERBATIM
    {
        double*** pd = (double***)(&(_p_Vother));
        *pd = (double**)hoc_Ecalloc((size_t)_ln, sizeof(double*));
    }
    ENDVERBATIM
}

PROCEDURE set_Vother(i) {
    VERBATIM
    {
        double** pd = (double**)_p_Vother;
        pd[(int)_li] = hoc_pgetarg(2);
    }
    ENDVERBATIM
}

FUNCTION get_Vother(i) {
    VERBATIM
    {
        double** pd = (double**)_p_Vother;
        _lget_Vother = *pd[(int)_li];
    }
    ENDVERBATIM
}

PROCEDURE declare_Vextother() {
    LOCAL n
    n = MAX_CON
    VERBATIM
    {
        double*** pd = (double***)(&(_p_Vextother));
        *pd = (double**)hoc_Ecalloc((size_t)_ln, sizeof(double*));
    }
    ENDVERBATIM
}

PROCEDURE set_Vextother(i) {
    VERBATIM
    {
        double** pd = (double**)_p_Vextother;
        pd[(int)_li] = hoc_pgetarg(2);
    }
    ENDVERBATIM
}

FUNCTION get_Vextother(i) {
    VERBATIM
    {
        double** pd = (double**)_p_Vextother;
        _lget_Vextother = *pd[(int)_li];
    }
    ENDVERBATIM
}
