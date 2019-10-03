# On the representation of radially symmetric model in NEURON

## Rationale

*NEURON* provides a great environment to design and simulate spatially-extended neuron models with multiple compartments. However, it only implements linear spatial discretization, where morphological sections are approximated by cylindrical compartments and connected in a cable-like topological organization. That is not suited for the representation of radially-symmetric models.

Due to this fixed morphological representation mode, a precise geometrical conversion scheme must be defined in order to enable the simulation of radially-symmetric models with *NEURON*. Particularly, such a conversion scheme must ensure that the axial and membrane currents are correct.

## Radially-symmetric model

Consider two radially symmetric membrane sections: a central, circular membrane patch (compartment $C$) of radius $a$, surrounded by a peripheral circular membrane section (compartment $P$) expanding from $a$ to an outer radius $b$. Compartments $C$ and $P$ are modeled electrically by voltage gated RC circuits (representing their respective local transmembrane dynamics), linked to ground in the extracellular medium, and connected to each other within a sub-membrane intracellular space of depth $h$ by a cylindrical resistor $R_{C, P}$.

### Membrane currents

The membrane surface areas of both compartments are given by:
- $A_{m ,C} = \pi \cdot a^2$
- $A_{m ,P} = \pi \cdot (b^2 - a^2)$

Hence, given the membrane currents densities in each compartment, respectively $i_{m, C}$ and $i_{m, P}$ membrane currents are given by:
- $I_{m, C} = i_{m, C} \cdot A_{m, C} = i_{m, C} \cdot \pi \cdot a^2$
- $I_{m, P} = i_{m, P} \cdot A_{m, P} = i_{m, P} \cdot \pi \cdot (b^2 - a^2)$

### Intracellular current

The "intracellular surface area" of a given radial cross-section at a radius *r* from the center is given by:

$A_r = 2\pi \cdot r\cdot h$

Moreover, given an intracellular radial resistivity $\rho$ and a radial electric field $E_r$, the intracellular radial current density $J_r$ at a distance $r$ equals:

$J_r = \frac{E_r}{\rho} = \frac{1}{\rho} \cdot \frac{dV}{dr}$

Hence, the total intracellular current flowing through a radial cross-section of radius $r$ is given by:

$I_r = J_r \cdot A_r = \frac{2\pi \cdot r\cdot h}{\rho} \cdot \frac{dV}{dr}$

From which we can isolate the local potential variation as:

$dV = I_r \cdot \frac{\rho}{2\pi \cdot r \cdot h} dr$

Integrating that variation between two radii $r_1$ and $r_2$, we have:

$V(r_2) - V(r_1)
= \int_{r_1}^{r_2} dV
= \int_{r_1}^{r_2} I_r \cdot \frac{\rho}{2\pi \cdot r \cdot h} dr
= I_r \cdot \frac{\rho}{2\pi \cdot h} \int_{r_1}^{r_2} \frac{dr}{r}
= I_r \cdot \frac{\rho}{2\pi \cdot h} \cdot \ln(\frac{r_2}{r_1})
= I_r \cdot R_{r_1,r_2}
$

where $R_{r_1,r_2}$ represents the intracellular radial resistor between $r_1$ and $r_2$.

Now, to compute the radial intracellular current between the two compartments of our model, we can assume an intracellular resistor that spans between the middle radial coordinates of compartments $C$ and $P$, i.e. respectively $\frac{a}{2}$ and $\frac{a + b}{2}$:

$R_{C, P}
= R_{\frac{a}{2},\frac{a + b}{2}}
= \frac{\rho}{2\pi \cdot h} \cdot \ln(\frac{\frac{a + b}{2}}{\frac{a}{2}})
= \frac{\rho}{2\pi \cdot h} \cdot \ln(\frac{a + b}{a})$

Finally, we can expres the intracellular current spreading radially between compartments $C$ and $P$ as:

$I_r
= \frac{V_P - V_C}{R_{C, P}}
= \frac{V_P - V_C}{\frac{\rho}{2\pi \cdot h} \cdot \ln(\frac{a + b}{a})}$

## Standard cable model

The simplest way to represent two connected compartments in *NEURON* is through a standard cable model containing two cylindrical sections of diameters $d_1$ and $d_2$ and lengths $L_1$ and $L_2$, respectively. These sections are also modeled electrically by voltage gated RC circuits (representing their respective local transmembrane dynamics), linked to ground in the extracellular medium, and connected to each other by a cylindrical resistor $R_{1, 2}$.

### Membrane currents

The membrane surface areas of both cylinders are given by:
- $A_{m, 1} = \pi \cdot d_1 \cdot L_1$
- $A_{m, 2} = \pi \cdot d_2 \cdot L_2$

Hence, given the membrane currents densities in each compartment, respectively $i_{m, 1}$ and $i_{m, 1}$ membrane currents are given by:
- $I_{m, 1} = i_{m, 1} \cdot A_{m, 1} = i_{m, 1} \cdot \pi \cdot d_1 \cdot L_1$
- $I_{m, 2} = i_{m, 2} \cdot A_{m, 2} = i_{m, 2} \cdot \pi \cdot d_2 \cdot L_2$

### Intracellular current

Given a resistivity $\rho$, the longitudinal resistance of a cylindrical segment of diameter $d$ and length $L$ is given by:

$R = \frac{4 \cdot \rho \cdot L}{\pi \cdot d^2}$

To compute the longitudinal  intracellular current between the two compartments of our model, we can assume that RC nodes are located at the mid-point of each section along the longitudinal axis. Hence we have:

$I_{ax}
= \frac{V_2 - V_1}{\frac{R_1}{2} + \frac{R_2}{2}}
= \frac{V_2 - V_1}{\frac{2 \rho}{\pi}\big(\frac{L_1}{d_1^2} + \frac{L_2}{d_2^2}\big)}
$

## Conversion

In order to accurately represent the radially-symmetric model in NEURON, the geometries of the two cylindrical cable compartments must defined such that, for identical values of transmembrane potentials ($V_1 = V_C$ and $V_2 = V_P$), spatially integrated membrane and intracellular currents are equivalent in both models, i.e.:

- $I_{m, C} = I_{m, 1}$
- $I_{m, P} = I_{m, 2}$
- $I_r = I_{ax}$

that is:

- $\pi \cdot a^2 \cdot i_{m,C} = \pi \cdot d_1 \cdot L_1 \cdot i_{m, 1}$
- $\pi \cdot (b^2 - a^2) \cdot i_{m, P} = \pi \cdot d_2 \cdot L_2 \cdot i_{m, 2}$
- $\frac{V_P - V_C}{\frac{\rho}{2\pi \cdot h} \cdot \ln(\frac{a + b}{a})} = \frac{V_2 - V_1}{\frac{2 \rho}{\pi}\big(\frac{L_1}{d_1^2} + \frac{L_2}{d_2^2}\big)}$

given $V_1 = V_P$, and hence $i_{m, 1} = i_{m, C}$ and $i_{m, 2} = i_{m, P}$.

After simplification, we obtain the following system of equivalences between the 3 parameters of the radially-symmetric model ($a$, $b$ and $h$) and the 4 parameters of the cable model ($d_1$, $L_1$, $d_2$ and $L_2$):

- $a^2 = d_1 \cdot L_1$
- $b^2 - a^2 = d_2 \cdot L_2$
- $\ln(\frac{a + b}{a}) = 4 \cdot h \big(\frac{L_1}{d_1^2} + \frac{L_2}{d_2^2}\big)$

In order to solve the system, we must equalize the number of unknowns with the number of equations. Therefore, we impose identical diameters to the two compartments of the cable model ($d_1 = d_2 = d$), yielding the following system:

- $a^2 = d \cdot L_1$
- $b^2 - a^2 = d \cdot L_2$
- $\ln(\frac{a + b}{a}) = \frac{4 \cdot h}{d^2}(L_1 + L_2)$

After algebraic resolution, we obtain:

- $d = \sqrt[3]{\frac{4 \cdot h \cdot b^2}{\ln(\frac{a + b}{a})}}$
- $L_1 = \frac{a^2}{d}$
- $L_2 = \frac{b^2 - a^2}{d}$
