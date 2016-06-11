# Paper outline

## Abstract

* The problem is to cluster data where many values are missing.
* It is interesting because it is common, and ad-hoc solutions degrade accuracy.
* Our solution achieves better accuracy than inserting dummy data, and `O(nnz)` running time per iteration.
* This suggests that other simple algorithms could be sparsified and achieve similar results.

## Intro

## Models and Methods

* (Regular K means proble)
* The reformulated problem is: Given n d-dimensional observations, where each observation exists on some subset of [d]. Create k clusters, where the total euclidean distance in the defined subspaces of each observation to each mean is minimized.
* Assumes K is chosen correctly, and that the defined subspaces overlap "enough".
* Other approaches are cosine metric, or picking dimensions [10.1198/jasa.2010.tm09415]. Our method accounts for all dimensions that have data. It doesn't require filling missing observations with fake data.
* (algorithm description, running time analysis)

## Experiments and Results

* (user, film, rating) triples split up into 90% training, 10% validation data, and put into sparse matrices.
* For each K, the algorithm is run 4 times, and the outcome with the least validation error is selected. Then the optimal K is selected. 
* (figures)
* We compare against regular K-means, and SVD. Missing data is filled with the average. Regular K-means is run just like our K-means. The SVD is computed only once, and different ranks are tested. The scores of each algorithm is the rmse to the validation set.
