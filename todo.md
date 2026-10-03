# Tasks

0. PiecewiseAnalyzer
   1. Plot CDF frac changepoints for man max_fractional_reduction (MFR)
   2. Plot a single model with changepoints for different MFR
   3. Plot CAD for MFR, singular, groups
   4. getChangepointsFromFile
      1. Preparation: file selects "model" from all PWPrediction files
      2. In PSD: create a DF of this file; check parameters to match prediction; return changepoints if match, else None.
   
# Plots

1. Plot CAF for each model with an accuracy constraint by choosing the max_fractional_reduction that achieves that accuarcy. Display AUC, max_fractional_reduction and also plot the timecourse with changepoints.
