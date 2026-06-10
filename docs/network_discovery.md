# Network Discovery

Developing a modeling of a chemical network is intellectually challenging and quite time consuming. This document describes how a workflow that discovers a reaction network from timecourse data of the concentrations of chemical species.

## Definitions and Notation

* The **state variables** are species concentrations, which vary over time. The name of the $i$-th state variable is $S_i$ and its concentration at time $t$ is $x_i (t)$. We use $x_i$ if time is implied.
* A **polynomial term** is a product of state variables, each of which may be raised to a power. For example, the term idetified as $S_i S_j$, the product of two state variables, has the value at time $t$ of $x_i (t) x_j (t)$.
* A **rate law** is an algebraic expression consisting of a **kinetic constant** times a polynomial term.
* A **mass action** rate law has a function form of a polynomial term. Suppose that the left-hand side of the reaction is $p S_i + q S_j$, which means that $p$ moles of species $S_i$ are combined with $q$ modles of species $S_j$. Then the mass action rate law is $k x_i^p x_j^q$, where $k$ is a kinetic constant.

## Assumptions

* **Data assumptions**
  * Concentration data are provided for all chemical species.
  * There is no noise in the data.
* **Chemical network assumptions**
  * All rate laws are mass action with no more than two reactants.
  * Each reaction is uniquely defined by its set of reactants paired with their stoichiometries.

## Observations

From the foregoing, several observations are useful.

1. Given the rate law for a reaction, we know its reactants and their stoichiometries. This follows from mass action kinetics.
2. The number of non-boundary reactions is equal to the number of distinct polynomials used in rate laws.

## Algorithm

The algorithm has two parts: discovering the system of differential equations and constructing the reaction network.

### Discovering the System of Differential Equations

The algorithm takes as input:

* A dataframe of time courses of state variables. The index of the dataframe is the time instants.
* A list of boundary species. (This can be implied by the fact that boundary species do not change value.)
  
Its output is a dataframe ${\bf D}$ structured as:

* Index $m$ (row) consists of polynomial terms denoted by $P_m$. For example, $P_m$ could be the string $S_i S_j$. The index has $M$ elements, and $P_0 =1$ (the context term).
* Column $i$ is predicted concentration for chemical species $i$. Its value at time $t$ is denoted by $hat{\dot{x}}_i (t)$.
* Let $c_{m,i}$ denote the value in row $m$ and column $i$ of ${\bf D}$. The estimated differential equation is $\frac{d S_i}{d t} = \sum_m c_{m,i} P_m$.

The algorithm proceeds in the following steps:

1. Construct a preliminary system of differential equations.

   1. Use ``NetworkRateDiscovery`` in @network_rate_discovery.py to construct a system of differential equations.

   2. ``NetworkRateDiscovery.summary()`` returns a dataframe that we denote as ${\bf D^{\prime}}$ that has the same structure as ${\bf D}$.

2. Construct ${\bf D}$ by fitting to species concentrations, not to their derivatives. We do this to improve estimation accuracy. We initialize ${\bf D}$ to ${\bf D}^{\prime}$. We proceed by iterating over each species $S_i$, which is a column in ${\bf D}$:

   1. Construct a vector $v$ of length $M$ whose values are the coefficients obtained in Step (1) for $\{k_{m,i}\}$
   2. If $S_i$ is not a boundary species, then $v(0) = 0$ (the constant term); otherwise, $v(0) = 1$.
   3. Use ``lmfit`` to optimize the values of the non-zero elements of $v$. The residuals are calculated as follows:

      1. Construct a differential equation for $\dot{S}_i$ using the terms in $v$ for $P_m$.
      2. Numerically integrate the differential equation to obtain values at the times provided in the input data.
      3. The residuals are the squared error between these estimates and the data.

   4. Set column $i$ of ${\bf D}$ to $v$.

### Constructing the Reaction Network

The algorithm takes as input the dataframe ${\bf D}$.

The output of the algorithm is a set of reactions. There will be one reaction for each value of $P_m \neq 1$. (The $P_m$ are rows in ${\bf D}$.)

For each $P_m$:

   1. If $P_m = 1$, we may have boundary reactions:

      1. Let $c_{m,i}$ be the value of cell in the $m$-th row and $i$-th column of ${\bf D}$.
      2. For each species index $i$:
         1. If $c_{m,i} < 0$:
            1. Add the reaction $S_i \rightarrow \empty$ with rate law |c_{m,i}|$.
         2. If $c_{m,i} > 0$:
            1. Add the reaction $empty \rightarrow S_i$ with rate law c_{m,i}$.

   2. Else there is a single reaction for $P_m$ that we call $R_m$.

      1. Find the reactants and their stoichiometries.
         1. Let $\cal{S}_R = \{S_{i_k} \}$ be the species in $P_m$. These are the reactants of the reaction constructed for row $P_m$. Let $N_R = |\cal{S}_R$.
         2. The stoichiometry of $S_{i_k}$ is $p_{i_k}$ (the exponent of $S_{i_k}$ in $P_m$).
      2. Find the kinetic constant for the reaction
         1. Let $S^{\star} = S_{i_1}$ with stoichiometry $p^{\star} = p_{i_1}$.
         2. Define $k^{\star} = |c^{\star}/p^{\star}|$, where $c^{\star}$ is the value in the column for $S^{\star}$ in the row for $P_m$ in ${\bf D}$.
         3. $k^{\star}$ is the kinetic constant for the reaction for row $P_m$.
      3. Find the products and their stoichiometries.
         1. Let $\cal{S}_P = \{S_{j_n}\}$ be the set of species for which its cell in row $P_m$ has a value greater than 0.
         2. Define $q_n = c_{j_n}/k^{\star}$, where $c_{j_n}$ is the value of cell for $S_{j_n}$ in row $P_m$.  Let $N_P = |\cal{S}_P|$.
      4. Construct $R_m$, the reaction for $P_m$
         1. $LHS = p_{i_1} S_{i_1} + \cdots + p_{i_{N_R}} S_{i_{N_R}}$
         2. $RHS = q_{i_1} S_{j_1} + \cdots + q_{j_{N_P}} S_{j_{N_P}}$
         3. $R_m$ has the rate law $k^{\star} P_m$.
         4. Add $R_m$ to the network.
