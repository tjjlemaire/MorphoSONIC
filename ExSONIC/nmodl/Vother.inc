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