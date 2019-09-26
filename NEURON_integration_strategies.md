# Rationale for a customized *NEURON* integration scheme

## Integrating membrane dynamics with a varying capacitance

The original **NICE** model involves a bi-directionally coupled electromechanical system to represent the interaction between mechanical acoustic waves and the electrical response of a neuron. However, given the intrinsic complexity of this model, it does not make sense to implement the full electro-mechanical model in *NEURON*.

We have developed a ***multi-Scale Optimized Neuronal Intramembrane Cavitation* (SONIC) model**, in which membrane capacitance and ion channel rate constants depend on the local amplitude of the acoustic perturbation ($A$, in kPa) and on the membrane charge density ($Q = c_m \cdot v$, in $nC/cm^2$) of the compartment. These time-varying quantities are computed by bilinear interpolation in the $(A, Q)$ space using two-dimensional lookup tables precomputed in Python and inserted here as *FUNCTION_TABLE* instances.

The standard equation used in *NEURON* to compute total membrane current assumes a constant membrane capacitance (1):

$$i_m = \frac{d c_m v}{dt} + i_{ion}(v, t) = c_m \frac{dv}{dt} + i_{ion}(v, t)$$

However in the presence of a time-varying capacitance, this relationship becomes more complex (2):

$$i_m = \frac{d c_m v}{dt} + i_{ion}(v, t) = c_m \frac{dv}{dt} + v \frac{d c_m}{dt} + i_{ion}(v, t)$$

While such complexity can potentially be implemented in NEURON by dynamically updating the parameter $c_m$ and defining an additional membrane current $i_{dcdt} = v \cdot d c_m/dt$, sharp changes in capacitance will induce tremendous values of $i_{dcdt}$ which may hinder the numerical integration of $v$.

However, we can redefine our system to use membrane charge density $Q$ (rather than $v$) as its differential variable, using a simple variable change ($Q = c_m \cdot v$, $\frac{dQ}{dt} = c_m \frac{dv}{dt} + v \frac{d c_m}{dt}$) (3):

$$i_m = \frac{dQ}{dt} + i_{ion}\big(\frac{Q}{cm}, t \big)$$

This new differential system is far less sensitive to changes in membrane capacitance, and thereby far easier to integrate. Moreover, it removes the need to compute $d c_m / dt$ for which we don't have a closed-form expression.

Since *NEURON*'s built-in equations cannot be modified, this model implementation uses a workaround in which the specific membrane capacitance $c_m$ is set to 1 in the *hoc/python* caller, such that $Q = v$, in order to use $v$ as a dummy variable to integrate the evolution of membrane charge density.

The time-varying capacitance $C_m^*(t)$ is effectively computed at each iteration in the *BREAKPOINT* block (from the precomputed interpolation table), from which we deduct the instantaneous membrane potential $V_m^* = Q / C_m^*$ and use it to compute ionic currents. Using appropriate recording vectors in the *hoc/python* caller, both $Q$ and $V_m^*$ can be retrieved upon simulation completion.

## Connecting different sections with various intracellular coupling schemes

### The Q-V discrepancy

The simulation of spatially-extended models implies the solving of partial differential equations, where connected compartments influence each other's states via a coupling force, driven by spatial differences in a key state variable.

In the case of neurons, the coupling force is **intracellular current**, and at the morphological scale, this current is driven by the **gradient in intracellular potential** across nodes. Assuming a constant extracellular potential, that is equivalent to the **gradient in transmembrane potential**. This is the standard assumption used in *NEURON* when sections are coupled together via the "connect" function.

In this context, the charge casting of local membrane dynamics entails an important limitation: through the implicit redefinition of $v$, intracellular currents between sections connected with the standard "connect" function are computed based on differences in membrane charge density, not differences in membrane potential.

### A custom connection scheme across NMODL and Python

In order to work around that algorithmic discrepancy and provide more flexibility on the choice of the state variable governing intracellular coupling, we implemented a custom **definition of intracellular currents as a separate density mechanism**. That mechanism uses pointers that can be set to point to values of any state variable at the relevant section (either $v$ or a variable computed as part of another distributed membrane mechanism). It then computes the resulting intracellular current and normalizes it by the section's own membrane surface area in order to return a current density that is unit-consistent with other density mechanisms (such as ionic currents). 

In order to facilitate section-connection with various coupling variables in a user-friendly manner, a `SeriesConnector` class was implemented that closely mimics *NEURON*'s built-in "connect" function, also called from within the Python interface.

### Restricting axial resistance above a minimum

One notable drawback of this workaround is that the resulting PDE scheme is solved without proper computation of the Jacobian matrix, which can significantly slow down integration or even generate divergence in solutions where intracellular currents become too high. For that reason, an additional parameter *rmin* can be set upon instantiation of `SeriesConnector` objects, that modifies the intracellular resistivity of attached sections in order to restrict intracellular resistance above a certain threshold and thereby ensure that intracellular currents stay within a reasonable order of magnitude. This restriction has no significant impact on simulation results, since it is typically encountered when intracellular currents dominate over membrane currents by several orders of magnitude, and where total synchronization of membrane potentials across the sections of interest is observed with and without resistance bounding.

### Adapting resistivity for charge-based connection schemes:

One should note that if the custom connection scheme is to be used with $v$ (an alias for membrane charge density) as coupling variable, intracellular resistivity must be multiplied by membrane capacitance to ensure the self-consistency of the charge-based differential scheme, where intracellular currents are defined by:

$I_{ax} = \frac{\Delta V}{R} = \frac{\Delta V}{R \cdot c_m}$


### Considerations for application to SONIC model expansions

In practice, given the very localized nature of sonophores, their impact on the global membrane electrical state, and hence on the intracellular electrical driving forces, is very difficult to assess. Hence, different coupling schemes should be compared, based on either LIFUS-modulated (i.e. effective) membrane potential, unaffected membrane potential (i.e. $Q / C_{m0}$), or a partially affected membrane potential, in order to check their impact on predicted neural responses.