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
| `**kwargs` | | | Forwarded to each per-segment `SystemDiscovery(...)` constructor (e.g. `threshold`, `poly_degree`, `bias_species`, `is_normalize`) unmodified — see `fit()` step 5. |

### Internal state

After `fit()` is called, the following attributes are available:

- `self._segment_models: list[SystemDiscovery]` — One fitted `SystemDiscovery` per segment. There are n+1 segments if there are n change points. Each is fit on its own raw (unscaled) data slice; `SystemDiscovery`'s own per-segment `Scaler` (default `is_normalize=True`) normalizes for fitting and denormalizes its output back to physical units, so each segment's coefficients and derivatives are self-consistently in physical units.
- `self._segment_boundaries: list[tuple[float, float]]` — `(start_time, end_time)` for each segment.

**Note on a prior design (corrected):** an earlier version of this design scaled the entire timecourse once with a single global `Scaler`, sliced the *scaled* DataFrame per segment, and forced `is_normalize=False` on every segment model (reasoning: the data was "already normalized," so segments shouldn't re-normalize). That is a real bug, not a stylistic choice: `is_normalize=False` makes `SystemDiscovery` treat its input as final physical units and never denormalize its output. Since the input was actually pre-scaled by `x / s_i` per species, every cross-species coefficient came out inflated by the ratio `s_i/s_j` of the two species' global standard deviations (self-coefficients were unaffected, since the scaling cancels on the diagonal — confirmed numerically: a 2-species fixture with global `std(S1)/std(S2) = 2.823` produced cross-term coefficients inflated by `2.824`, while self-decay terms fit exactly). `predict_derivative()`/`predict()` then integrated these still-scaled derivatives directly against true physical-unit states, corrupting every prediction. There is no global `self._scaler` in the corrected design — nothing else in this design needs one.

## `fit()` method

```python
def fit(self) -> "PiecewiseSystemDiscovery"
```

Returns `self`, matching `SystemDiscovery.fit()`'s chaining convention.

### Algorithm

1. **Create normalized Jacobians**: Using the raw Jacobians from the timecourse, let ${\bf A} = {a_{ij}}$. The entry is normalized to account for the magnitudes of the column state variable, $x_j$, and the row state variable, $x_i$. Let $s_i, s_j$ be the standard deviations of these state variables over their full timecourse. Then, $a_{ij} \rightarrow a_{ij} \frac{s_j}{s_i}$. This normalization exists solely to make Jacobian entries comparable for change-point detection — it is local to this step and has no bearing on how segment models are later fit.

2. **Compute change-point signal**: Calculate the Frobenius distance between time-adjacent normalized Jacobians and divide this by $n^2$, where $n$ is the number of state variables (chemical species). Then smooth these differences over time using a Gaussian kernel with bandwidth `fit_kernel_bandwidth`. (`predict_kernel_bandwidth` is not used here — it applies only to `predict_derivative()`.) The result is one signal value per time point (the first time point has no signal, since it has no predecessor).

3. **Detect change points**: Sort all candidate time points by signal value, descending. Maintain an initially-empty list of accepted change-point times, kept sorted by time. Walk the sorted candidates in descending-signal order:
   1. If the candidate's signal < `change_point_threshold`, stop (all remaining candidates have smaller or equal signal).
   2. Determine where the candidate's time would be inserted into the accepted list (by time, not by signal rank). Check the lengths (in time points) of the two segments this would create — bounded by the candidate's time-adjacent neighbors among the already-accepted points, or by the timecourse start/end if there are no neighbors on that side. If either resulting segment would be shorter than `min_segment_length`, reject the candidate and continue to the next.
   3. Otherwise, insert the candidate into the accepted list (in time order).
   4. If the accepted list has reached `num_change_point` entries, stop.
   5. Otherwise continue to the next candidate.

   The final change points are the accepted list, in time order. No merging is performed — a candidate that doesn't fit is simply skipped, never combined with a neighbor.

4. **Fit per-segment models**: For each segment defined by the accepted change points:
   1. Slice the **raw** (unscaled) timecourse DataFrame to the segment's `[start_time, end_time)` (a plain `DataFrame.loc[...]` slice — `Timecourse` has no `makeSubmodel()` equivalent to `Trajectory.makeSubmodel()`).
   2. Construct `SystemDiscovery(segment_df, **kwargs)`, passing `kwargs` through unmodified. `SystemDiscovery`'s own default (`is_normalize=True`) applies unless the caller explicitly overrides it via `kwargs`; each segment normalizes using its own local statistics and denormalizes its own output, so coefficients and derivatives come out in physical units without any cross-segment scale dependency.

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

(Note: segment models default to `is_normalize=True` — each segment's own `normalize`/`denormalize` round-trip is what keeps its coefficients and derivatives in physical units. `predictOneStepDerivative` is written generically, matching `_simulate`'s existing pattern, so it works correctly whether or not a given segment model normalizes.)

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
