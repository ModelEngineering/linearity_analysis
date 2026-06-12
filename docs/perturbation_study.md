# Perturbation Study

This study explores how perturbation initial values affects the quality of linear fits for models and for species.

In @data/evaluate_monomial_models-0.01, we see that about 15% of the models have a deg1_min of at least 0.9 on their training data. This study compares these results with perturbations of initial values. In the studies, the training data are unperturbed time courses. Separately, time courses are constructed where initial values have been changed by $\pm 5\%$, $\pm 10\%$, $\pm 20\%$, $\pm 50\%$ with a perturbation species fraction of 1.0. A separate ``Timecourse`` is constructed for each perturbation (at total of 8), and then the model trained on unperturbed data is used to predict perturbed data. Please use and/or add capabilities to``SystemDiscovery.analyzePerturbations`` to do this analysis. The result is a CSV file with the columns: model_name, threshold, r2_0, r2_-05, r2_-10, r2_-20, r2_-50, r2_+05, r2_+10, r2_+20, r2_+50. The output file path is new argument to ``analyzePerturbations``.
* Revise and/or rewrite @perturbation_study.py.
* As needed, make use of existing classes and functions in @src/. Use code in @scripts/ as a guide for implementation, but do not import these modules.
* The $R_2$ value for a model is the minimum $R^2$ value for all species in the model.
* The poly_degree is 1.
* Do $R^2$ clamping so that $0 \leq R^2 \leq 1$.
