# Linearity Assessment

We assess the linearity of a model by how many changepoints are required to accurately predict the timecourse.

## Algorithm

1. Find the fewest number of evenly spaced changedpoints to achieve the desired accuracy.
   1. Selecting increasing number of changepoints by doubling
   2. Once a sufficient number is found, reduce the number of changepoints until the accuracy objective is violated.
2. Remove individual changepoints until accuracy is violated.

## Visualization

1. For each accuracy considered, plot a histogram of the number of changepoints required.

## Questions

1. What characterizes models that are more (less) linear?
