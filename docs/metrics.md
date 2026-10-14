# Metrics for Assessing Nonlinearity

This writeup describes a new metric for quantifying nonlinearity of a timecourse.

First, some definitions. Consider a timecourse in the interval $[0, T]$ with $N$ times that are simulated (points). Let $(c_1, \cdots, c_M)$ be
**changepoints** that delineate piecewise linear models. That is, we have the time segments $[c__{i}, c_{i+1})$ such that:
(a) $c_1 = 0$, (b) $c_M = T$, (c) the initial values for the segment are taken at $c_i$, and (d] predictions are available at
$(c_i, \cdots, c_i+1}]$.

The **segment curve** is a function $\psi$ of the segment sizes such that for a segment size of $s$, $\psi (s) = f_s$, the fraction of the timecourse that lies within a segment of size $s$. We normalize $s$ by $T$ and so $s \in [0, 1]$. Note that
$$\int_0^1 \psi (s)ds = 1$$.
The **cumulative segment curve** $\Psi(s) = \int_0^s \psi(s) ds$.

The **area under the cumulative segment curve (AUCSC)** is $\alpha = \int_0^1 \Psi(s) ds$.
