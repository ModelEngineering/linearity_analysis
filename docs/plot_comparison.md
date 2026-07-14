# ``plotComparison``

## Description

The method ``plotComparison`` compares the three ways that end time can be calculated: from ``endtime`` in the SEDML;
from steady state; and
from the maximum coefficient of variation.

## What it does

1. Construct three plots for the axes provided or generate a new figure with 3 axes if they are not present.

2. For each, a plot is constructed as follows.
    1. Calculate the end time for the appropriate plot using the methods in @characteristic_time_estimator. An optional timeout argument can be provided for steady state. Note that the SEDML method fails for any model that has no end time specified.
    2. Plots the normalized Time course over time from 0 to end time relative to the left vertical axis.
    3. Plot the *metric* normalized standard deviation between the species relative to the right vertical axis.
    4. There are no ticks or ticked labels on the x-axis except for the end time. There are no ticks or tick labels on the left vertical axis. The right vertical axis only has a tick and label for the maximum value of the metric.

3. A plot is constructed for each one of the three methods for obtaining end time. If that method produces no end time, "None" is written in the center of a blank plot.
