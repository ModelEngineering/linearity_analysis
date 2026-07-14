# Tasks

0. Consider rule where choose characteristic time to when encounter the first local minima for std of normal. How does this do relative to Max CV? This is called First Local Minima (FLM).
   1. Start with runtime of simulation = 1.
      1. Compute the std metric
      2. Calculate its diff to find the first local minima.
      3. After the decrease, find the local minima. If None, then double the runtime of the simulation.
1. Assess characteristic times.
2. Assess predictions by piecewise
3. Not properly handling boundary species. Can estimate constants well, but not including the bias terms.
