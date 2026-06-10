# Linear Predictions

## Feature description

This feature evaluates the accuracy of predicting trajectories in chemical
reaction networks (CRNs) using multiple linear models. A linear model is
specified by its Jacobian, initial values of species concentrations, and
forced inputs specified by boundary species or floating species with 0th order rate laws.

## Implementation plan

1. Create a class LinearPredictor that is constructed from a JacobianCollection, initial values for Species, and forced inputs
   1. It calculates the mean of the Jacobians in the collection
   2. It predicts floating concentrations at future times using the linear system specified by the Jacobian and forced inputs.
2. Create a class MultipleLinearPredictor.
   1. It is constructed from a ClusteredJacobianCollection and LRoadrunner
   2. It uses LRoadrunner to calculate forced inputs
   3. It determines initial concentrations of all species
   4. For jacobian_collection in jacobian_collections
      1. Find initial values of species (which may be what was produced on the last iteration)
      2. Create a LinearPredictor
      3. Predict species concentrations for the last timepoint of this JacobianCollection.
