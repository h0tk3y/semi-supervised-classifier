Semi-supervised max-likelihood classifier
===

A small semi-supervised classifier with max-likelihood self-learning

See `src/main/kotlin/MaxLikelihoodSemiSupervisedClassifier`. 

Given a `MaxLikelihoodEstimator` and two lists of 
`data: List<List<Double>>` (list of vectors of features) and 
`labels: List<Int?>` (list of class labels for the `data` items, with `null`s 
for missing labels), it self-learns the missing labels and returns them.

You can therefore add the test data items to `data` without labels to classify
them according to the max-likelhood estimation used for self-learning, or 
use the restored labels in another classifying algorithm.

### Demo

The `Demo.kt` generates three uncorrelated gaussian clusters and erases 19 labels
out of 20 points. It then runs the algorithm to restore the labels and visualizes 
them.

TODO: add a picture

The points with known labels are filled. The shapes denote the original classes, 
and the color denotes the restored labels.