TITLE FH membrane mechanism

COMMENT
Equations governing the effective membrane dynamics of a Xenopus myelinated fiber node
upon electrical / ultrasonic stimulation, based on the SONIC model.

Reference: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019).
Understanding ultrasound neuromodulation using a computationally efficient
and interpretable model of intramembrane cavitation. J. Neural Eng.

@Author: Theo Lemaire, EPFL
@Date: 2019-07-04
@Email: theo.lemaire@epfl.ch
ENDCOMMENT

CONSTANT {
   Z_Na = 1
   FARADAY = 96485.3
   Rg = 8.31342
   Z_K = 1
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
   SUFFIX FHauto
   NONSPECIFIC_CURRENT iNa : Sodium current
   NONSPECIFIC_CURRENT iKd : delayed-rectifier Potassium current
   NONSPECIFIC_CURRENT iP : non-specific delayed current
   NONSPECIFIC_CURRENT iLeak : non-specific leakage current
   RANGE Adrive, Vm : section specific
   RANGE stimon     : common to all sections (but set as RANGE to be accessible from caller)
}

PARAMETER {
   stimon       : Stimulation state
   Adrive (kPa) : Stimulation amplitude
   cm = 2.0 (uF/cm2)
   pNabar = 0.008 (cm/s)
   Nai = 0.01374 (M)
   Nao = 0.1145 ()
   T = 293.15 ()
   pKbar = 0.0012000000000000001 (cm/s)
   Ki = 0.12 ()
   Ko = 0.0025 ()
   pPbar = 0.00054 (cm/s)
   gLeak = 0.03003 (S/cm2)
   ELeak = -69.974 (mV)
}

STATE {
   m : iNa activation gate
   h : iNa inactivation gate
   n : iKd gate
   p : iP gate
}

ASSIGNED {
   v  (nC/cm2)
   Vm (mV)
   iNa (mA/cm2)
   iKd (mA/cm2)
   iP (mA/cm2)
   iLeak (mA/cm2)
}

FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)
FUNCTION_TABLE alpham(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betam(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betah(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betan(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE alphap(A(kPa), Q(nC/cm2)) (/ms)
FUNCTION_TABLE betap(A(kPa), Q(nC/cm2)) (/ms)

FUNCTION efun(x) {
    efun = x / (exp(x) - 1)
}

FUNCTION ghkDrive(Vm, Z_ion, Cion_in, Cion_out, T) {
    LOCAL x, eCin, eCout
    x = Z_ion * FARADAY * Vm / (Rg * T) * 1e-3
    eCin = Cion_in * efun(-x)
    eCout = Cion_out * efun(x)
    ghkDrive = FARADAY * (eCin - eCout) * 1e6
}

INITIAL {
   n = alphan(0, v) / (alphan(0, v) + betan(0, v))
   p = alphap(0, v) / (alphap(0, v) + betap(0, v))
   h = alphah(0, v) / (alphah(0, v) + betah(0, v))
   m = alpham(0, v) / (alpham(0, v) + betam(0, v))
}

BREAKPOINT {
   SOLVE states METHOD cnexp
   Vm = V(Adrive * stimon, v)
   iP = pPbar * p * p * ghkDrive(Vm, Z_Na, Nai, Nao, T)
   iLeak = gLeak * (Vm - ELeak)
   iKd = pKbar * n * n * ghkDrive(Vm, Z_K, Ki, Ko, T)
   iNa = pNabar * m * m * h * ghkDrive(Vm, Z_Na, Nai, Nao, T)
}

DERIVATIVE states {
   n' = alphan(Adrive * stimon, v) * (1 - n) - betan(Adrive * stimon, v) * n
   p' = alphap(Adrive * stimon, v) * (1 - p) - betap(Adrive * stimon, v) * p
   h' = alphah(Adrive * stimon, v) * (1 - h) - betah(Adrive * stimon, v) * h
   m' = alpham(Adrive * stimon, v) * (1 - m) - betam(Adrive * stimon, v) * m
}