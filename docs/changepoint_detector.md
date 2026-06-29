# Change Point Detector

## Background

Detecting change points is critical to adapting systems to nonstationarities. The detector provides an efficiet solution for a sequence of univariate data. Given a number of change points, the detector determines how to partition the sequence of the univariate data so as to minimize the adjusted sum of squares (subtracting the mean) of the partitions. Alternatively, the algorithm inputs the minimum change in required w.r.t. the total sum of squares, and then finds change points until this criteria is violated.

## Mathematic Background

We make the following notation.

* A **sequence** $S$ is a set of contiguous positive integers. The length of the sequence is denoted by $|S|$.
$f_S =1$ is the first element of the sequence and $l_S = f_S + |S| -1$ is the
last element of $S$.
* $U$ is a **subsequence** of the sequence $S$ is a sequence
that begins with $i \in [f_S, l_S]$ and ends if $j \in [i, l_S]$.
* $x_i$ be the $i$-th element in the univariate data.
* $sum(X, S) = \sum^{i=l_S}_{i=f_S}x_i$
* $sum(XX, S) = \sum^{i=l_S}_{i=f_S}x^2_i$
* $\cal{P}$ is a **partition** of the sequence $S$ iff $\cal{P} = \{ S_1, \cdots, S_K \}$ where: (a) $S_k$ is a subsequence of $S$; (b) $l_{S_k} + 1 = f_{S_{k+1}}$ for $k < K$; and (c) $l_{S_K} = |S|$.

First, note that
for the partition of $S$ that is partitioned by $\cal{P}$.

* $\sum_{S_i \in \cal{P}} sum(X, S_i) = sum(X, S)$
* $\sum_{S_i \in \cal{P}} sum(XX, S_i) = sum(XX, S)$

Next, we calculate $A_S$, the adjusted sum of squares for the sequence $S$
$$
\begin{align}
A_S & = & \sum_{i=f_S}^{l_S} \left(x_i - \frac{sum(X, S)}{|S|} \right)^2 \\
& = & sum(XX, S) -  \frac{sum(X,S)^2}{|S|}
\end{align}
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
