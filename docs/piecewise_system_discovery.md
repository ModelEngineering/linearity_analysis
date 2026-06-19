# Piecewise System Discovery

This documents a class that implements piecewise linear approximations to model systems with time-varying dynamics. Instead of fitting a single global set of ODE coefficients, it detects change points in the time series and fits a separate `SystemDiscovery` model to each segment, blending predictions across segment boundaries with a Gaussian kernel.

## Motivation

Many biochemical systems exhibit regime shifts (e.g., activation/inactivation of a pathway, saturation effects, or external stimuli). A single linear model may not capture these transitions well. Piecewise approximation allows the model to adapt its coefficients locally while maintaining smooth transitions.

## Class constructor

```python
class PiecewiseSystemDiscovery(object):
    def __init__(
        self,
        timecourse: Timecourse,
        num_change_point: int = 2,
        min_segment_length: int = 100,
        change_point_threshold: float = 0.1,
        fit_kernel_bandwidth: float=1.0,
        predict_kernel_bandwidth: float=1.0,
        **kwargs,
    ) -> None:
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timecourse` | `Timecourse` | *(required)* | Timecourse object.
| `num_change_point` | `int` | `2` | Number of change points.
| `min_segment_length` | `int` | `100` | Minimum number of time points per segment. Candidate change points that would result in segments shorter than this are rejected to ensure stable fits. |
| `change_point_threshold` | `float` | `0.1` | Threshold for detecting a changepoint between adjacent entries.
| `fit_kernel_bandwidth` | `float` | `1.0` | Normalizing term for the Gaussian kernel used to smooth the change-point signal (see `fit()` step 3). |
| `predict_kernel_bandwidth` | `float` | `1.0` | Normalizing term for the Gaussian kernel used to blend segment models in `predict_derivative()`. |
| `**kwargs` | | | Forwarded to each per-segment `SystemDiscovery(...)` constructor (e.g. `threshold`, `poly_degree`, `bias_species`). `is_normalize` is always forced to `False` regardless of `kwargs` — see `fit()` step 5. |

### Internal state

After `fit()` is called, the following attributes are available:

- `self._segment_models: list[SystemDiscovery]` — One fitted `SystemDiscovery` per segment. There are n+1 segments if there are n change points. Each is fit with `is_normalize=False` because the data was already normalized in step 1.
- `self._segment_boundaries: list[tuple[float, float]]` — `(start_time, end_time)` for each segment.
- `self._scaler`: A `Scaler` (see `src/scaler.py`) fit on the entire (unsegmented) timecourse. Used in `fit()` step 1 to produce the scaled timecourse that segment models are fit on. Unrelated to the per-entry Jacobian normalization in step 2, which exists only to compute the change-point signal and is discarded afterward.

## `fit()` method

```python
def fit(self) -> "PiecewiseSystemDiscovery"
```

Returns `self`, matching `SystemDiscovery.fit()`'s chaining convention.

### Algorithm

1. **Scale the timecourse**. Use `self._scaler` to scale `timecourse.timecourse_df`. This scaled DataFrame — not the original — is what gets sliced per segment in step 5.

2. **Create normalized Jacobians**: Using the raw Jacobians from the timecourse (before scaling in step 1), let ${\bf A} = {a_{ij}}$. The entry is normalized to account for the magnitudes of the column state variable, $x_j$, and the row state variable, $x_i$. Let $s_i, s_j$ be the standard deviations of these state variables over their full timecourse. Then, $a_{ij} \rightarrow a_{ij} \frac{s_j}{s_i}$. This normalization is independent of `self._scaler` / step 1 — it exists solely to make Jacobian entries comparable for change-point detection, not to scale data used for fitting.

3. **Compute change-point signal**: Calculate the Frobenius distance between time-adjacent normalized Jacobians and divide this by $n^2$, where $n$ is the number of state variables (chemical species). Then smooth these differences over time using a Gaussian kernel with bandwidth `fit_kernel_bandwidth`. (`predict_kernel_bandwidth` is not used here — it applies only to `predict_derivative()`.) The result is one signal value per time point (the first time point has no signal, since it has no predecessor).

4. **Detect change points**: Sort all candidate time points by signal value, descending. Maintain an initially-empty list of accepted change-point times, kept sorted by time. Walk the sorted candidates in descending-signal order:
   1. If the candidate's signal < `change_point_threshold`, stop (all remaining candidates have smaller or equal signal).
   2. Determine where the candidate's time would be inserted into the accepted list (by time, not by signal rank). Check the lengths (in time points) of the two segments this would create — bounded by the candidate's time-adjacent neighbors among the already-accepted points, or by the timecourse start/end if there are no neighbors on that side. If either resulting segment would be shorter than `min_segment_length`, reject the candidate and continue to the next.
   3. Otherwise, insert the candidate into the accepted list (in time order).
   4. If the accepted list has reached `num_change_point` entries, stop.
   5. Otherwise continue to the next candidate.

   The final change points are the accepted list, in time order. No merging is performed — a candidate that doesn't fit is simply skipped, never combined with a neighbor.

5. **Fit per-segment models**: For each segment defined by the accepted change points:
   1. Slice the **scaled** timecourse DataFrame from step 1 to the segment's `[start_time, end_time)` (a plain `DataFrame.loc[...]` slice — `Timecourse` has no `makeSubmodel()` equivalent to `Trajectory.makeSubmodel()`).
   2. Construct `SystemDiscovery(segment_df, is_normalize=False, **kwargs)`. `is_normalize=False` is always passed, overriding any `is_normalize` key present in `kwargs`, since the data is already scaled at the `PiecewiseSystemDiscovery` level — re-normalizing per segment would scale against each segment's own (shorter, less representative) statistics instead of the full timecourse's.

### Edge cases handled

- **0 change points detected**: If no change point is accepted, use the entire timecourse as a single segment.
- **All segments too short**: If every candidate is rejected for violating `min_segment_length` (so 0 change points survive step 4), this is handled by the same path as the previous bullet — use the entire timecourse as a single segment.

## `predict_derivative(t, x)` method

```python
def predict_derivative(self, t: float, x: np.ndarray) -> np.ndarray
```

This is a method of the `PiecewiseSystemDiscovery` class. It predicts $\frac{d{\bf x}(t)}{dt}$ given the current state ${\bf x}(t)$ at time $t$, blending across all fitted segment models. Returns a denormalized (physical-units) derivative.

### Algorithm

1. Compute a Gaussian weight for each segment model based on the distance from `t` to the segment's midpoint:

   ```text
   w_i = exp( -0.5 * (t - midpoint_i)^2 / sigma^2 )
   ```

   where `sigma` is `predict_kernel_bandwidth` and `midpoint_i` is the mean of `self._segment_boundaries[i]`.
2. For each segment model `i`, obtain its one-step derivative at `x`:

   ```text
   dx_dt_i = segment_model_i.predictOneStepDerivative(x)
   ```

3. Blend: `dx_dt = sum(w_i * dx_dt_i) / sum(w_i)`.
4. Return `dx_dt`.

### Prerequisite: new `SystemDiscovery` method

`SystemDiscovery` currently has no public, non-integrating, single-state derivative evaluator — only `predict(test_df)`, which integrates over a full time grid. Step 2 above requires adding a small public method to `SystemDiscovery`, e.g.:

```python
def predictOneStepDerivative(self, x: np.ndarray) -> np.ndarray:
    """Evaluate the fitted ODE's right-hand side at a single state x (no integration)."""
    self._require_fitted()
    z = self._normalizer.normalize(x)
    dz_dt = self.model.predict(z.reshape(1, -1))[0]
    return self._normalizer.denormalize(dz_dt)
```

This factors out the `rhs` closure already used internally by `SystemDiscovery._simulate()` (`src/system_discovery.py:864-868`), without changing `_simulate`'s behavior. This is a prerequisite change outside the scope of `PiecewiseSystemDiscovery` itself, called out here so it isn't missed during implementation.

(Note: since segment models are always fit with `is_normalize=False`, `normalize`/`denormalize` are identity operations in practice for `PiecewiseSystemDiscovery`'s use — but `predictOneStepDerivative` is written generically, matching `_simulate`'s existing pattern, rather than assuming the caller never normalizes.)

## `predict()` method

```python
def predict(self, test_df: pd.DataFrame = NULL_DF) -> pd.DataFrame:
```

### Algorithm

1. Determine the initial condition and time grid: If `test_df` is provided, use `test_df.values[0]` as `x0` and `test_df.index` as the time grid. Otherwise, use the training initial condition and the training time grid.

2. Integrate forward in a single continuous solve from `x0` over the full time grid, using `self.predict_derivative` as the right-hand side (analogous to `SystemDiscovery._simulate`'s use of `solve_ivp`, e.g. `solve_ivp(fun=self.predict_derivative, t_span=(t[0], t[-1]), y0=x0, t_eval=t, ...)`). The state passed into `predict_derivative` at each evaluation is the integrator's current (predicted) state — there is no periodic restart from the observed trajectory, unlike `LinearPredictor`'s windowed integration. This is a deliberate difference: continuous integration lets the Gaussian blending in `predict_derivative` produce smooth transitions across segment boundaries, which a restart-every-`num_step` scheme would interrupt.

3. Return a `pd.DataFrame` with the predicted trajectories, same format as `SystemDiscovery.predict()`.

### Notes

- The Gaussian kernel ensures smooth transitions: segments whose time range is far from `t` contribute weights that approach (but never exactly reach) zero. There is no separate discrete rule for out-of-range extrapolation — when `test_df`'s time range falls outside the training range, the nearest segment's weight simply dominates the (still continuous) blend as long as `predict_kernel_bandwidth` is small relative to the extrapolation distance.

## `score()` method

```python
def score(self) -> ScoreInfo:
```

Returns a `ScoreInfo` object (the `system_discovery.ScoreInfo` dataclass: `min`, `median`, `max`, `values`, `num_nonzero_term`) with R² statistics aggregated across all segments, weighted by segment length (number of time points). The $R^2$ is computed on the estimates of the derivatives compared with their true values.

- For each segment, compute the $R^2$ values (one per species) using the parent method (`SystemDiscovery.score()`) on the segment's own data.
- Weighted aggregation:
  ```
  weighted_values = concat([segment.score().values repeated segment_length times for each segment])
  ```
  `min`, `median`, and `max` are then computed from `weighted_values`.
- `num_nonzero_term` is the sum of `segment.score().num_nonzero_term` across all segments. Note this is a simple sum, not deduplicated by species — a term appearing in multiple segments' equations is counted once per segment.
- Raises `RuntimeError` if called before `fit()`, matching `SystemDiscovery`'s `_require_fitted()` convention (likewise for `predict()` and `printEquations()`).

## `printEquations()` and `__str__()`

Prints the discovered ODE for each segment, prefixed with the segment's time range:
```
[Segment 1: t in [0.0, 12.5)]
  dA/dt = -0.342 * A + 0.018
  dB/dt = 0.342 * A - 0.115 * B

[Segment 2: t in [12.5, 25.0)]
  dA/dt = -0.521 * A + 0.032
  dB/dt = 0.521 * A - 0.203 * B + 0.041 * A * B
```
