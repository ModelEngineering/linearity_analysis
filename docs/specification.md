# Specification for Model Linearity Analysis

## Architecture

* Class LinearAnalyzer. This class reads and runs models, collecting their Jacobians and constructs a heatmap for the result.
  * Constructed with an SBML or Antimony model and a default specification of the simulate options for start (0), end(10), and number of points (100)
  * Method collectJacobians. Runs the model, collecting a Jacobian at each timepoint. The Jacobian should contain only floating species in the model.
  * Method plot creates a heatmap for the Jacobian matrix where the entries are the coefficient of variation for each Jacobian.
  * Class method processBioModels for the SBML models in /Users/jlheller/home/Technical/repos/temp-biomodels/final

## Architecture Updates

* Add the following methods
  * Method makeJacobianCVs returns an array of CVs for cells in the Jacobian
  * Class method processBioModelsCVs invokes makeJacobianCVs for the SBML models in /Users/jlheller/home/Technical/repos/temp-biomodels/final and returns a dictionary of model identifiers whose key is the output of makeJacobianCVs
  
## Coding styles

* Functions are camel case.
* Variables are lower case with underline separators
* Document all functions
* Use typing in function signatures.

## Revised coding styles

* All dictionaries have the suffix "_dct"
* All arrays have the suffix "_arr"
* All dataframes have the suffix "_df"
* All series have the suffic "_ser"
* All and only lists end in "s"

## Tests

* There is a test for each method of every class

## Repository Structure

* All code is in src
* All tests are in tests
