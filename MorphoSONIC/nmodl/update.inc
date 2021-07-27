FUNCTION_TABLE Cm_table(A(kPa), t(/ms)) (uF/cm2)

PROCEDURE update() { LOCAL Cm
    if (detailed) {
        Cm = Cm_table(Adrive * stimon, fmod(t, 1 / Fdrive))
        A_t = 0.               : zero amplitude for lookups
        Vm = v / Cm            : voltage taken from Qm/Cm ratio
        y = Vm                 : voltage dependent rates
    } else {
        A_t = Adrive * stimon  : time-dependent amplitude for lookups
        Vm = V(A_t, v)         : charge density dependent voltage
        y = v                  : charge density dependent rates
    }
}