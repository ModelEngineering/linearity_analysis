# Calculating Characteristic Using Coefficient of Variation

Finding the characteristic time of a system is challenging for biological models because of the wide range of magnitudes present in systems.
By characteristic time, we mean a time span for a simulation in which the systems dynamics are in full evidence. This is somewhat subjective, and it is non-trival because systems may have multiple time scales.

Herein, we consider using coefficient of variation of state variables over the time course. One appeal is that coefficient of variation is unitless and so we have fair comparisons between time courses. However, we don't one time state variable to dominate the calculation. For this reason, the objective function calculates the median value of the coefficient of variable for the state variables.

The algorithm makes use of existing Python optimizers, such as ``lmfit``. It looks for the time value that maximizes the median value of the coefficient of variation of the time courses.