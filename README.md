# Analyze the accuracy of a linear approximations for model kinetics

## Objectives

* Assess how well an SBML is approximated by one or more linear models based on
  * Consistency of Jacobian
  * Accuracy of reproducing a non-linear simulation
* Identify "linearity bottleneck" reactions, those that must limit the viability of a linear approximation

# Tuning SystemDiscovery

* Need sufficient data, maybe in the 1000s.
* Reduce threshold (sensitivity) to get more sparse coefficients
* Set includebias = True
* Don't include boundary species

## Analyses


## Versions
