# Change Point Detector

## Background

Detecting change points is critical to adapting systems to nonstationarities. The detector provides an efficiet solution for a sequence of univariate data. Given a number of change points, the detector determines how to partition the sequence of the univariate data so as to minimize the adjusted sum of squares (subtracting the mean) of the partitions. Alternatively, the algorithm inputs the minimum change in required w.r.t. the total sum of squares, and then finds change points until this criteria is violated.

## Mathematical Background

We make the following notation.

* A **sequence** $S$ is a set of contiguous positive integers. The length of the sequence is denoted by $|S|$.
$f_S$ is the first element of the sequence, and $l_S = f_S + |S| -1$ is the
last element of $S$.
* $U$ is a **subsequence** of the sequence $S$ if $U$ is a sequence and
  * $f_U \in [f_S, l_S]$;
  * $l_U \in [f_U, l_S]$;
* $\cal{P}$ is a **partition** of the sequence $S$ iff $\cal{P} = \{ S_1, \cdots, S_K \}$ where: (a) $S_k$ is a subsequence of $S$; (b) $l_{S_k} + 1 = f_{S_{k+1}}$ for $k < K$; and (c) $l_{S_K} = |S| - f_S - 1$; and (d) $f_{S_1} = f_S$.
* $x_i$ is the $i$-th element in the univariate data.
* Let $S$ be a sequence.
  * $sum(X, S) = \sum^{i=l_S}_{i=f_S}x_i$
  * $sum(XX, S) = \sum^{i=l_S}_{i=f_S}x^2_i$

Let $S$ be a sequence and
$\cal{P}$ be a partition of $S$.

* $\sum_{S_i \in \cal{P}} sum(X, S_i) = sum(X, S)$
* $\sum_{S_i \in \cal{P}} sum(XX, S_i) = sum(XX, S)$

Next, we calculate $A_S$, the adjusted sum of squares for the sequence $S$
$$
\begin{align}
A_S & = & \sum_{i=f_S}^{l_S} \left(x_i - \frac{sum(X, S)}{|S|} \right)^2 \\
& = & sum(XX, S) -  \frac{sum(X,S)^2}{|S|}
\end{align}
$$
The adjusted sum of squares for a partition is the sum of the adjusted sum of squares for each subsequence in the partition. Let $\cal{P}$ be a partition of $S$, and $A_{\cal{P}}$ be the adjusted sum of squares for the partition.
$$
A_{\cal{P}} = \sum_{S_i \in \cal{P}} A_{S_i}
$$

Now we consider the sum of adjusted sum of squares for a partition of $S$.
Consider
the partition $\cal{P} = \{S_1, S_2 \}$.
$$
\begin{align}
A_{S_1} + A_{S_2} & = &
sum(XX, S_1) -  \frac{sum(X,S_1)^2}{|S_1|}
+ sum(XX, S_2) -  \frac{sum(X,S_2)^2}{|S_2|} \\
& = &
sum(XX, S) -  \frac{sum(X,S_1)^2}{|S_1|} -  \frac{sum(X,S_2)^2}{|S_2|} \\
\end{align}
$$
Thus, we minimize
$A_{S_1} + A_{S_2}$ if we maximize
$\frac{sum(X,S_1)^2}{|S_1|} +  \frac{sum(X,S_2)^2}{|S_2|}$.
We refer to such a partition as the **minimal partition** of the sequence $S$.

## Finding the minimal partition

Our objective is to find the minimal partition of a subsequence $S$ that lies within $[1, \cdots, N]$.

The inputs to the algorithm are:

* The data values, $x_i$.
* The sums for each element in the subsequence
  * $y_n = y_{n-1} + x_n$, where $y_0 = 0$
* The complement sums for $y_n$
  * $z_n = y_N - y_{n-1}$

The algorithm proceeds as follows.

1. For $n \in [f_S, l_S]$, calculate the signal for the partition, $A_{S_{n,1}} + A_{S_{n,2}}$.
   1. $y^{\prime}_n = y_n - y_{f_S-1}$
   2. $z^{\prime}_n = z_n - y_N + y_{l_S}$
   3. $s_n = \frac{y_{n-1}^{\prime}y^{\prime}_{n-1}}{n} + \frac{z_n^{\prime}z_n^{\prime} }{(N-n)}$
2. Find the change point: $n^{\star} = argmax_n s_n$

## Critiques
I have reviewed the mathematical derivations in `docs/changepoint_detector.md`. There are several notation errors, typos, and logic inconsistencies in the "Finding the minimal partition" section and the partition definition.

### Summary of Findings

#### 1. Partition Definition Errors
In the definition of a partition $\cal{P} = \{ S_1, \cdots, S_K \}$ of sequence $S$:
- **Condition (c):** It states $l_{S_K} = |S|$. This is only true if $f_S = 1$. It should be **$l_{S_K} = l_S$**.
- **Condition (d):** It states $f_{S_1} = 1$. This is only true if the sequence starts at 1. It should be **$f_{S_1} = f_S$**.

#### 2. Algorithm Errors (Finding the minimal partition)

**A. Typographical Errors in $s_n$ Formula**
The expression $s_n = \frac{y_{n-1}^{\prime}y^{\prime}_{n-1}}{n} + \frac{z_n^{\prime}z_n^{\prime} }{(N-n)}$ contains typos. It is likely intended to be:
$$s_n = \frac{(y'_{n-1})^2}{\text{length}_1} + \frac{(z'_n)^2}{\text{length}_2}$$
The notation $y_{n-1}^{\prime}y^{\prime}_{n-1}$ is confusing and looks like a multiplication of two different variables or a typo for squaring.

**B. Incorrect Denominators (Logic Error)**
The denominators $n$ and $N-n$ assume that the sequence $S$ is exactly $[1, N]$ and the split occurs at $n$. However, the algorithm is defined for a **subsequence** $S$ within $[1, N]$.
- If the first partition is $S_1 = [f_S, n-1]$, its length is $|S_1| = (n-1) - f_S + 1 = \mathbf{n - f_S}$.
- If the second partition is $S_2 = [n, l_S]$, its length is $|S_2| = l_S - n + 1$.
Using $n$ and $N-n$ will lead to incorrect results for any subsequence that doesn't start at 1 and end at $N$.

**C. Index Inconsistency**
- Step 1.1 calculates $y'_n = y_n - y_{f_S-1}$.
- Step 1.3 uses $y'_{n-1}$.
While this is mathematically consistent if you want $S_1$ to end at $n-1$, the loop $n \in [f_S, l_S]$ and the calculation of $y'_n$ inside the loop make the usage of $y'_{n-1}$ awkward (it refers to a value from the previous iteration or a value not yet calculated for $n=f_S$).

**D. Loop Range and Edge Cases**
The loop $n \in [f_S, l_S]$ allows $n = f_S$. If $n = f_S$, then $S_1 = [f_S, f_S-1]$, which is an empty set. This leads to $|S_1| = 0$, causing a **division by zero** in the $s_n$ calculation. The range should be $n \in [f_S + 1, l_S]$ to ensure both partitions have at least one element.

### Recommended Corrections

**Revised Algorithm Section:**

1. For $n \in [f_S+1, l_S]$:
   1. Calculate sum of first partition: $y'_{n-1} = y_{n-1} - y_{f_S-1}$
   2. Calculate sum of second partition: $z'_n = y_{l_S} - y_{n-1}$
   3. Calculate signal: $s_n = \frac{(y'_{n-1})^2}{n - f_S} + \frac{(z'_n)^2}{l_S - n + 1}$
2. Find the change point: $n^{\star} = \text{argmax}_n s_n$