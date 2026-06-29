# Change Point Detector

## Background

Detecting change points is critical to adapting systems to nonstationarities. The detector provides an efficient solution for a sequence of univariate data. Given a number of change points, the detector determines how to partition the sequence of the univariate data so as to minimize the adjusted sum of squares (subtracting the mean) of the partitions. Alternatively, the algorithm inputs the minimum reduction in the sum of squares required w.r.t. the total sum of squares, and then finds change points until this criterion is violated.

## Mathematical Background

We make the following notation.

* A **sequence** $S$ is a set of contiguous positive integers. The length of the sequence is denoted by $|S|$.
$f_S$ is the first element of the sequence, and $l_S = f_S + |S| -1$ is the
last element of $S$.
* $U$ is a **subsequence** of the sequence $S$ if $U$ is a sequence and
  * $f_U \in [f_S, l_S]$;
  * $l_U \in [f_U, l_S]$;
* $\mathcal{P}$ is a **partition** of the sequence $S$ iff $\mathcal{P} = \{ S_1, \cdots, S_K \}$ where: (a) $S_k$ is a subsequence of $S$; (b) $l_{S_k} + 1 = f_{S_{k+1}}$ for $k < K$; (c) $l_{S_K} = l_S$; and (d) $f_{S_1} = f_S$.
* $x_i$ is the $i$-th element in the univariate data.
* Let $S$ be a sequence.
  * $sum(X, S) = \sum_{i=f_S}^{l_S}x_i$
  * $sum(XX, S) = \sum_{i=f_S}^{l_S}x^2_i$

Let $S$ be a sequence and
$\mathcal{P}$ be a partition of $S$.

* $\sum_{S_i \in \mathcal{P}} sum(X, S_i) = sum(X, S)$
* $\sum_{S_i \in \mathcal{P}} sum(XX, S_i) = sum(XX, S)$

Next, we calculate $A_S$, the adjusted sum of squares for the sequence $S$
$$
\begin{align}
A_S & = & \sum_{i=f_S}^{l_S} \left(x_i - \frac{sum(X, S)}{|S|} \right)^2 \\
& = & sum(XX, S) -  \frac{sum(X,S)^2}{|S|}
\end{align}
$$
The adjusted sum of squares for a partition is the sum of the adjusted sum of squares for each subsequence in the partition. Let $\mathcal{P}$ be a partition of $S$, and $A_{\mathcal{P}}$ be the adjusted sum of squares for the partition.
$$
A_{\mathcal{P}} = \sum_{S_i \in \mathcal{P}} A_{S_i}
$$

Now we consider the sum of adjusted sum of squares for a partition of $S$.
Consider
the partition $\mathcal{P} = \{S_1, S_2 \}$.
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

Our objective is to find the minimal partition of a subsequence $S$ that lies within $[1, \cdots, N]$. For a single split point, the algorithm proceeds as follows.

The inputs to the algorithm are:

* The data values, $x_i$.
* The prefix sums for each element in the sequence:
  * $y_n = \sum_{i=1}^n x_i$, where $y_0 = 0$

The algorithm proceeds as follows:

1. For $n \in [f_S+1, l_S]$:
   1. Calculate sum of first partition: $y'_{n-1} = y_{n-1} - y_{f_S-1}$
   2. Calculate sum of second partition: $z'_n = y_{l_S} - y_{n-1}$
   3. Calculate signal: $s_n = \frac{(y'_{n-1})^2}{n - f_S} + \frac{(z'_n)^2}{l_S - n + 1}$
2. Find the change point: $n^{\star} = \text{argmax}_n s_n$

### Multiple Change Points and Stopping Criteria

To find multiple change points, this process can be applied recursively (Binary Segmentation):

1. Find the optimal single split point $n^\star$ for the entire sequence $S$.
2. Split $S$ into $S_1 = [f_S, n^\star-1]$ and $S_2 = [n^\star, l_S]$.
3. Recursively apply the algorithm to $S_1$ and $S_2$.

**Stopping Criteria:**

* **Fixed number of points**: Stop when $K$ change points have been found.
* **Minimum Reduction**: A split is only accepted if the reduction in the adjusted sum of squares $A_S - (A_{S_1} + A_{S_2})$ exceeds a specified threshold $\epsilon \cdot A_{S_{total}}$, where $\epsilon$ is the minimum required relative reduction.
