# Tasks

1. Return to the original objective: evaluating the linearity of BioModels. Compare the accuracy of (a) linear model and (b) quadratic models. Consider segmentation as well (refer to references). Application to linearity analysis through Jacobians, MCA.
   1. Timecourse represents the results of a simulation; inputs a model.
   2. Simple network discovery at 2 to 3 thresholds (in units of std in normalized space): 0.01, 0.1, 1.0.
   3. Plot histogram of min $R^2$ for model species.
   4. Title: How Linear is BioModels?
2. ModelIterator iterates over the serialized models.
3. Construct piece-wise segments
4. Not properly handling boundary species. Can estimate constants well, but not including the bias terms.
5. Consider CRN construction?
6. Can I do better with manual splitting?
7. Review slow_subspace_prediction.py for possible use in splitting
8. Bug in 577 & 599. Split = 2 results in an unstable system
9. Evaluate consistency of forcing inputs.
10. Re-run no split linear analyses
