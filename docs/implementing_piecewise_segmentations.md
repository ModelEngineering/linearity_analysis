# Piece-wise Linear Approximation for Affine Differential Equations

### The `TPWL` class — core algorithm

```python
model = TPWL(f, jac, delta=0.4, weighting="nearest")
model.train(x0, t_span, t_eval)   # Phase 1: collect expansion points
model.simulate(x0_test, t_span, t_eval)  # Phase 2: run surrogate
```

**Phase 1 — Training (point selection):**  
Integrates the full nonlinear ODE once. At each step, if the current state is farther than `delta` from all existing expansion points, a new point `sᵢ` is added with its local Jacobian `Aᵢ = df/dx|_{sᵢ}` and affine offset `bᵢ = f(sᵢ) - Aᵢ sᵢ`, so that `Aᵢx + bᵢ = f(sᵢ)` exactly at the expansion point.

**Phase 2 — Online evaluation:**  
For a new state `x`, the TPWL right-hand side is:

```
f_TPWL(x) = Σᵢ wᵢ(x) · (Aᵢx + bᵢ)
```

Two weighting strategies are implemented:
- **Nearest** — hard switch: only the closest expansion point contributes
- **Gaussian** — soft blending: `wᵢ ∝ exp(-α · ‖x − sᵢ‖²)`

---

## Results Summary

| System | Points | Nearest error | Gaussian error |
|---|---|---|---|
| Van der Pol (mu=1.5) | 36 | 0.10 | 0.22 |
| Lorenz (chaotic) | 70 | 0.048 | 0.064 |
| Duffing (double-well) | 56 | 0.14 | 0.10 |

The delta sensitivity study (second figure) shows the key trade-off: halving `delta` roughly doubles the number of expansion points and reduces error by ~3–10×, until round-off and step-size errors dominate at very small `delta`.

**Notable observations:**
- For Van der Pol, nearest-neighbor outperforms Gaussian here — common when the nonlinearity is sharp enough that mixing remote linear models hurts
- For Lorenz, 70 expansion points give <5% mean error on a chaotic trajectory with the same initial condition — impressive given the model is just blended linear systems
- The Lorenz Jacobians have `max Re(eig) = +11.5`, reflecting the locally unstable directions of the strange attractor — the TPWL correctly captures this local instability:0
- 