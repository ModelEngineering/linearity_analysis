# Jacobian Estimator

## Description

The JacobianEstimator estimates the Jacobian and forcing inputs from a time series for a system of linear differential equations. That is, given the vector ${\bf x}(t)$, it estimates the matrix ${\bf A}$ and the forcing input vector ${\bf u}$ that is the best fit for
$$
\dot{\bf x}(t) = {\bf A} {\bf x}(t) + {\bf u}
$$

``JacobianEstimator`` should be structured in a manner similar to ``SystemDiscovery``.

## Implementation

### Constructor

    def __init__(self, timecourse_df: pd.DataFrame)
``timecourse_df`` has as its index time and the column names are state variables. The constructor creates the following instance state variables:

* ``self.timecourse_df`` is the argument passed
* ``self.dtimecourse_df`` is the derivative of the state variable for times starting at index 0. It is calculated as ``(self.timecourse_df.values[i+1, :] - self.timecourse_df.values[i,:])/(self.timecourse_df.index[i+1] - self.timecourse_df.index[i])``. Thus, if there are $N$ values in ``self.timecourse_df``, there will be $N-1$ in the derivative.

#### Validation

The constructor raises ``TypeError`` if ``timecourse_df`` is not a ``pd.DataFrame``. It raises ``ValueError`` if ``timecourse_df`` is empty, has no columns (no state variables), has an index (time) that is not strictly monotonically increasing, or contains any ``NaN`` or infinite values.

### fit

    def fit(alpha: float)

Fit estimates ${\bf A}$ and ${\bf u}$ using lasso and its tuning parameter $alpha$.

#### Guard

``predict`` raises ``RuntimeError`` if called before ``fit`` has been called.

### predict

    def predict(x: np.ndarray) -> np.ndarray

predicts the derivative given the state variable.
