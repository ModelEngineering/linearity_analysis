# Find Simulation End Time

Biological models work at many different time constants. To evaluate linearity, we need to do simulations in the natural time constants of the model.

# Algorithm

The algorithm proceeds in two phases. The first is to find a time at which the model is at steady state. The second is to find the time when it first enters steady state.

1. Find steady values of floating species. Use the ``Tellurium`` method ``steadystate()``.
2. Find a time when the system is in steady state. Starting with 1 s, run a simulation with that end time, and compare the floating species ending concentrations with the steady state values (normalizing by the mean of the steady state value). If it's not at steady state, then double the end time.
3. Find the first time when the system enters steady state. Reduce the end times until it is no longer at steady state. Then, the previous time is the first entry into steady state.