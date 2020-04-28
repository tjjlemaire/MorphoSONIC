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