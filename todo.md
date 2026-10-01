# Tasks

0. PiecewiseAnalyzer
   1. Plot CDF frac changepoints for man max_fractional_reduction (MFR)
   2. Plot a single model with changepoints for different MFR
   3. Plot CAD for MFR, singular, groups

1. Possible narratives:
   1. How many changepoints are needed to effectively linearize models in biomodels? This is an upper bound since we don't have a good algorithm for changepoint detection in terms of non-linearity. It is also an approximation since models with a large number of species are limited in the number of changepoints that they can accommodate because of data required for estimating the Jacobian.