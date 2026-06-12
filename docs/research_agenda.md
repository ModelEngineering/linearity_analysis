# Research Agenda

## To what extent is BioModels linear?
- [x] Fit first order monomial to all models and calculate $R^2$ for species time courses.
    - Minimum $R^2$ is model $R^2$
    - Individual $R^2$ are for species.
- [x] Evaluate the density of coefficients in the Jacobian.
- [ ] For the linear models, assess their dimensionality to see if dimension reduction is possible
- [ ] Characterize the linear models based on the what is being model and possibly other characteristics.
  
  ## How robust is linearity to perturbations to initial values?
- [ ] repeat the linear studies with perturbations of $\pm 5\%$, $\pm 19\%$, $\pm 20\%$, and $\pm 50\%$.

## Are some nonlinear models piecewise linear?
- [ ] Use a standard package for partitioning regressions to see if linearity can be achieved in segements of the time course.