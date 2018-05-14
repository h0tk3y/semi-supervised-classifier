Semi-supervised max-likelihood classifier
===

A small semi-supervised classifier with max-likelihood self-learning

See [`src/main/kotlin/MaxLikelihoodSemiSupervisedClassifier`](https://github.com/h0tk3y/semi-supervised-classifier/blob/master/src/main/kotlin/MaxLikelihoodSemiSupervisedClassifier.kt). 

Given a `MaxLikelihoodEstimator` and two lists of 
`data: List<List<Double>>` (list of vectors of features) and 
`labels: List<Int?>` (list of class labels for the `data` items, with `null`s 
for missing labels), it self-learns the missing labels and returns them.

You can therefore add the test data items to `data` without labels to classify
them according to the max-likelhood estimation used for self-learning, or 
use the restored labels in another classifying algorithm.

### Demo

The [`Demo.kt`](https://github.com/h0tk3y/semi-supervised-classifier/blob/master/src/main/kotlin/Demo.kt) generates three uncorrelated gaussian clusters and erases 19 labels
out of 20 points. It then runs the algorithm to restore the labels and visualizes 
them.

The points with known labels are filled. The shapes denote the original classes, 
and the color denotes the restored labels.

![image](https://user-images.githubusercontent.com/1888526/39957966-6745d142-5604-11e8-85b7-2643663a8d0f.png)

Another run with only one point-per-class given with a known label (labeled on the chart as 1, 2, and 3):

![image](https://user-images.githubusercontent.com/1888526/39973241-4d19ac1c-5726-11e8-9196-6c97d1001b44.png)
