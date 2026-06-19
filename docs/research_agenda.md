# Research Agenda

## How well can BioModels be modelled by a system of linear differential equaitons?

- [x] Fit first order monomial to all models and calculate $R^2$ for species time courses.

  - Minimum $R^2$ is model $R^2$
  - Individual $R^2$ are for species.

- [x] Evaluate the density of coefficients in the Jacobian and forcing inputs.
- [ ] For the linear models, assess their dimensionality to see if dimension reduction is possible
- [ ] Characterize the linear models based on the what is being modelled and possibly other characteristics.
  
  ## How robust is linearity to perturbations of initial values?

- [x] repeat the linear studies with perturbations of $\pm 5\%$, $\pm 19\%$, $\pm 20\%$, and $\pm 50\%$.
- [ ] Can robustness be improved by training the regression on perturbation data?

## What are the main reasons for nonlinear behavior?

- [ ] Analyze the nonlinear models to determine which species are nonlinear and how/when the Jacobian changes to look at reactions.

## Are some nonlinear models piecewise linear?

- [ ] Use a standard package for partitioning regressions to see if linearity can be achieved in segements of the time course.
  - [ ] Normalize Jacobians
  - [ ] k-means cluster with cluster distances with minimum cluster size.
  - [ ] SystemDiscovery for each cluster
  - [ ] Prediction using: (a) ${\bf x} (t)$ using SystemDiscover of cluster at $t$ similarly for adjacent time points; (b) apply gaussian kernel on points; (c) denormalize
