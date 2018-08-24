The original **NICE** model involves a bi-directionally coupled electromechanical system to
represent the interaction between mechanical acoustic waves and the electrical response
of a neuron. However, given the intrinsic complexity of this model, it does not make sense
to implement the full electro-mechanical model in *NEURON*.

We have developped a ***multi-Scale Optimized Neuronal Intramembrane Cavitation* (SONIC) model**, in which membrane capacitance and ion channel rate constants depend on the local amplitude of the acoustic perturbation (*A*, in kPa) and on the membrane charge density (*Q = cm v*, in nC/cm2) of the compartment. These time-varying quantities are computed by bilinear interpolation in the (*A, Q*) space using two-dimensional lookup tables pre-computed in Python and inserted here as FUNCTION_TABLE instances.

The standard equation used in NEURON to compute total membrane current assumes a
constant membrane capacitance:

        im = d(cm v)/dt + i_ion(v, t) = cm dv/dt + i_ion(v, t)                  (1)

However in the presence of a time-varying capacitance, this relationship becomes
more complex:

        im = d(cm v)/dt + i_ion(v, t) = cm dv/dt + v dcm/dt + i_ion(v, t)       (2)

While such complexity can potentially be implemented in NEURON by dynamically updating the
parameter *cm* and definining an additional membrane current *idcdt = v dcm/dt*, sharp changes
in capacitance will induce tremendous values of *idcdt* which may hinder the numerical
integration of *v*.

However, we can redefine our system to use membrane charge density *Q* (rather than *v*) as its
differential variable, using a simple variable change (*Q = cm v*, *dQ/dt = cm dv/dt + v dcm/dt*):

        im = dQ/dt + i_ion(Q/cm, t)                                             (3)

This new differential system is far less sensitive to changes in membrane capacitance, and thereby
far easier to integrate. Morevoer, it removes the need to compute dcm/dt for which we don't have
a closed-form expression.

Since *NEURON*'s built-in equations cannot be modified, this model implementation uses a workaround
in which the specific membrane capacitance cm is set to 1 in the hoc/python caller, such that *Q = v*,
in order to use *v* as a dummy variable to integrate the evolution of membrane charge density.

The time-varying capacitance *Cmeff* is effectively computed at each iteration in the BREAKPOINT block
(from the pre-computed interpolation table), from which we deduct the instantaneous membrane
potential *Vmeff = Q/Cmeff* and use it to compute ionic currents. Using appropriate recording vectors
in the hoc/python caller, both *Q* and Vmeff can be retrieved upon simulation completion.

This workaround workaround solution entails an important limitation: through the implicit
redefinition of v, axial currents between connected sections are computed based on differences
in membrane charge density, not differences in membrane potential. Since the 2 quantities can be
very different, important errors would appear in multi-compartmental models.

Therefore, for simple models consisting of several sections connected in series, the contribution
of axial currents can be defined as a separate density mechanism that uses pointers to the values
of *Vmeff* stored in this mechanism at the local and neighboring sections in order to compute an
additional current density iax, treated as a membrane current by NEURON.