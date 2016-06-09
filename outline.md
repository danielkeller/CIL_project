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
* Other approaches are cosine metric, or picking dimensions [10.1198/jasa.2010.tm09415]
