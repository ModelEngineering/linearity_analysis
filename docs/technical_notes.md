# Technical Notes

## Aims

1. Understand an existing network in terms of the roles of reactions.
2. Identify modules of reaction networks, networks that produce the same behavior. This only requires a good fit. It does not require solving the "inverse problem".
3. Generate a reaction network that produces a desired timecourse.
4. Evaluate sufficiency of models: linear, quadratic, full.
5. More effective ways to partition timecourses into simpler models. Use a quadratic fit to determine when quadratic terms can be dropped and so fit is linear.

## Network Discovery

1. SINDy has problems with scaling and interpretation because of its generality. For reaction networks, it likely suffices to consider polynomials of order 0, 1, 2. So, we could do a linear regression on terms $1, S_1, \cdots, S_N, S_1 S_1, S_1, S_2, \cdots, S_N S_N$. We just need to use an orthogonal form of the quadratic to avoid co-linearities. Also, we need to have enough data. Probably want to LASSO to provide sparsity. See Appendex A1.
2. Based on this the proposed algorithm is
   1. Calculate derivatives from data if needed (difference divided by $dt$).
   2. Calculate orthogonal predictors (A2)
   3. Do regression. Note that this requires a sufficient number of points because of all of the predictors. Use LASSO to obtain a sparse result.
   4. For all quadratic terms, remove linear terms contained in quadratic.
   5. Do regression on: 1, remaining linear terms, quadratics. These are the final coefficients in the differential equations.
3. Is it possible to generate a canonical quadratic representation of a network. Maybe, just for quadratic networks? If so, we could compare networks based on their canonical representation. One approach would be to make the discovery threshold as large as possible with $R^2$ not dropping below a prescirbed value. Another constraint is that the discovery threshold must be low enough so that there is only one reaction per monomial. The presence of a reaction is indicated by having coefficients that are integral multiples for the monomial.

## Summarizing

1. Reaction network behavior can be summarized by a system of quandratic differential equations (SQDE).
2. Method for constructing a reaction network that is equivalent to a SQDE:
   1. Obtain a table whose rows are quadratic monomials and columns are differentials of species concentrations. The table should only contain rows whose values are either 0 or an integral multiple of each other.
   2. Construct a reaction for each row
      1. Rate law is the smallest coefficient in the row times the monomial
      2. Reactants have negative values with a stoichiometry of its coefficient divided by the kinetic constant
      3. Products are positive values with stoichiometry constructed in the same way.

## Tasks

1. Develop a robust implementation of a quadratic fit using orthogonal independent variables. It should provide high quality fits for quadratic networks.

# Appendices

## A1: Orthogonal Polynomials for Regression

To make linear and quadratic terms orthogonal in regression, use **orthogonal polynomials** — specifically, transform your predictors using the **Gram-Schmidt orthogonalization** applied to the polynomial basis.

---

### The Problem

For a predictor $x$, the raw basis $\{1, x, x^2\}$ is highly collinear. The orthogonal polynomial basis $\{P_0(x), P_1(x), P_2(x)\}$ satisfies:

$$\sum_{i=1}^n P_j(x_i) \cdot P_k(x_i) = 0 \quad \text{for } j \neq k$$

---

### The Transformation

Using Gram-Schmidt, construct:

$$P_0(x) = 1$$

$$P_1(x) = x - \bar{x}$$

$$P_2(x) = (x - \bar{x})^2 - \frac{\sum(x_i - \bar{x})^4}{\sum(x_i - \bar{x})^2} \cdot 1 - \frac{[\sum(x_i-\bar{x})^3]^2}{\sum(x_i-\bar{x})^2 \cdot n}$$

The simplified key results:

- **Linear term:** $\tilde{x} = x - \bar{x}$ (centering is sufficient for orthogonality between $P_0$ and $P_1$)
- **Quadratic term:** $\tilde{x}^2 = (x - \bar{x})^2 - \widehat{\text{proj}}_{P_1}[(x-\bar{x})^2]$, which removes the component of $(x-\bar{x})^2$ correlated with $P_1$

For **symmetric/uniform** $x$ distributions, $(x - \bar{x})$ and $(x - \bar{x})^2$ are already orthogonal, so **centering alone suffices**.

---

### Practical Steps

1. **Center:** $z_i = x_i - \bar{x}$
2. **Compute the correction:** $\alpha = \frac{\sum z_i^3}{\sum z_i^2}$
3. **Orthogonal quadratic:** $q_i = z_i^2 - \alpha z_i - \frac{\sum z_i^2}{n}$

Then regress on $\{1, z_i, q_i\}$.

---

### In Practice

Most statistical software handles this automatically:

| Software | Command |
|---|---|
| **R** | `poly(x, degree=2)` — returns orthogonal polynomials by default |
| **Python (numpy)** | `numpy.polynomial.polynomial.polyvander()` + QR decomposition |
| **Python (sklearn)** | `PolynomialFeatures` + center/scale manually, or use `numpy.polynomial` |
| **Stata** | `orthpoly x, deg(2) generate(p*)` |

---

### Benefits

- **Numerical stability:** eliminates multicollinearity between $x$ and $x^2$
- **Interpretability:** coefficients of $P_1$ and $P_2$ are independent — you can test the significance of the quadratic term without the linear term's coefficient changing
- **Same fitted values:** the model fit is identical to the raw polynomial regression; only the parameterization changes

The key insight is that **centering solves most of the problem** — the residual orthogonalization of the quadratic term matters most when your $x$ distribution is noticeably asymmetric.

## A2: Multiple Predictors

With predictors $x_1, x_2, \ldots, x_p$, you need to orthogonalize a basis that includes constants, linear terms, and quadratic terms (both squared and cross-product terms). The Gram-Schmidt logic is identical — you just have more vectors to orthogonalize against.

---

### The Raw Basis

For $p$ variables, the full quadratic basis has $1 + p + \binom{p+1}{2}$ terms:

$$\{1,\; x_1, \ldots, x_p,\; x_1^2, \ldots, x_p^2,\; x_1 x_2,\; x_1 x_3, \ldots, x_{p-1}x_p\}$$

---

### Step 1: Center All Variables

$$z_j = x_j - \bar{x}_j \quad \text{for each } j = 1, \ldots, p$$

This immediately gives $P_0 = 1$ orthogonal to all $z_j$, since $\sum_i z_{ij} = 0$ by construction.

---

### Step 2: Orthogonalize the Linear Terms Against Each Other

The centered variables $z_1, \ldots, z_p$ are not generally mutually orthogonal — they may be correlated. Apply Gram-Schmidt across them:

$$P_1^{(1)} = z_1$$

$$P_1^{(2)} = z_2 - \frac{\langle z_2, P_1^{(1)}\rangle}{\langle P_1^{(1)}, P_1^{(1)}\rangle} P_1^{(1)}$$

$$P_1^{(j)} = z_j - \sum_{k=1}^{j-1} \frac{\langle z_j, P_1^{(k)}\rangle}{\langle P_1^{(k)}, P_1^{(k)}\rangle} P_1^{(k)}$$

The projection coefficient here is:

$$\frac{\langle z_j, P_1^{(k)}\rangle}{\langle P_1^{(k)}, P_1^{(k)}\rangle} = \frac{\sum_i z_{ij} P_1^{(k)}(x_i)}{\sum_i [P_1^{(k)}(x_i)]^2}$$

This is exactly the **regression coefficient** from regressing $z_j$ on $P_1^{(k)}$.

---

### Step 3: Orthogonalize Each Quadratic Term Against Everything Built So Far

For each quadratic candidate $v$ (either $z_j^2$ or $z_j z_k$), project out all previously accepted basis vectors:

$$P_2^{(v)} = v - \sum_{\ell \in \text{prior}} \frac{\langle v, P_\ell \rangle}{\langle P_\ell, P_\ell \rangle} P_\ell$$

where "prior" includes $P_0$, all orthogonalized linear terms $P_1^{(j)}$, and any quadratic terms already accepted. Specifically:

**For squared terms $z_j^2$:**

$$P_2^{(jj)} = z_j^2 - \underbrace{\frac{\sum z_{ij}^2}{n}}_{\text{proj onto }P_0} - \sum_{k=1}^{p} \underbrace{\frac{\sum_i z_{ij}^2 \cdot P_1^{(k)}(x_i)}{\sum_i [P_1^{(k)}(x_i)]^2}}_{\alpha_{jk}} P_1^{(k)} - \sum_{\text{prior quad}} (\cdots)$$

**For cross terms $z_j z_k$:**

$$P_2^{(jk)} = z_j z_k - \frac{\sum_i z_{ij}z_{ik}}{n} - \sum_{\ell} \frac{\langle z_j z_k, P_1^{(\ell)}\rangle}{\langle P_1^{(\ell)}, P_1^{(\ell)}\rangle} P_1^{(\ell)} - \sum_{\text{prior quad}}(\cdots)$$

Note the projection onto $P_0$ gives $\overline{z_j z_k} = \widehat{\text{Cov}}(x_j, x_k)$, the sample covariance.

---

### The Full Ordering

A natural ordering for Gram-Schmidt is:

$$\underbrace{P_0}_{\text{constant}} \;\to\; \underbrace{P_1^{(1)}, \ldots, P_1^{(p)}}_{\text{linear}} \;\to\; \underbrace{P_2^{(11)}, P_2^{(12)}, \ldots, P_2^{(pp)}}_{\text{quadratic}}$$

Each term is orthogonalized against **all predecessors** in this sequence.

---

### Matrix Formulation

Stack the raw basis vectors as columns of a matrix $V$ (each column is one basis function evaluated at all $n$ data points). Then Gram-Schmidt is just **QR decomposition**:

$$V = QR$$

where $Q$ has orthonormal columns — these are your orthogonal polynomial basis vectors, and $R^{-1}$ gives the transformation from the original basis to the orthogonal one. In practice:

```python
import numpy as np

# Build raw design matrix V: [1, z1, z2, z1^2, z2^2, z1*z2]
Z = X - X.mean(axis=0)         # center all variables
V = np.column_stack([
    np.ones(n),
    Z,
    Z**2,
    Z[:, 0] * Z[:, 1]           # cross term (extend for more variables)
])

Q, R = np.linalg.qr(V, mode='reduced')   # Q columns are orthonormal basis
```

Regressing $y$ on $Q$ gives coefficients that are **fully independent** — adding or removing any term doesn't affect the others.

---

### What Changes Relative to the Single-Variable Case

| Feature | Single variable | Multiple variables |
|---|---|---|
| Centering orthogonalizes | $P_0$ vs $P_1$ | $P_0$ vs all $P_1^{(j)}$ |
| Linear terms | Automatically orthogonal to $P_0$ | Must orthogonalize against **each other** |
| Skewness correction | Scalar $\alpha = m_3/m_2$ | Vector of projections per linear basis vector |
| Cross terms | None | Must project out $P_0$, all linear terms, and prior quadratics |
| Symmetry simplification | $\alpha=0$ if $x$ symmetric | Cross terms vanish if $x_j, x_k$ uncorrelated **and** joint distribution symmetric |

---

### Key Insight

The projection coefficient at every step is just an **OLS regression coefficient** — the amount of the new candidate explained by a previously accepted basis vector. Gram-Schmidt is therefore equivalent to a sequential series of residualizations, and the final orthogonal basis spans exactly the same column space as the original polynomial terms.
