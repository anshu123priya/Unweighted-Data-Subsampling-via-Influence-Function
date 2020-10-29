---
title: Mini Project Information
tags:
description: Unweighted Data Subsampling via Influence Function
---

# Mini Project Information
## Unweighted Data Subsampling via Influence Function

<!-- Put the link to this slide here so people can follow -->


Subsampling is a method that reduces data size by selecting a subset of the original data. The subset is specified by choosing a parameter n, specifying that every nth data point is to be extracted.


Weighted subsampling: trying to maintain the model performance when dropping several data.
 
our project work attempts to obtain a superior model by subsampling.

![](https://i.imgur.com/piLnPdb.png)

---

## Main Framework

The main process Implementation is as follows:

* Dataset Load 
* Training model on the full data set 
* compute the influence function (IF) for each sample in training set
* compute the sampling probability of each sample in training set
* Do subsampling
* train a subset-model on the reduced data set


![](https://i.imgur.com/8rbxsWf.png)

---
#### **Influence functions:** 
It give a measure of robustness of the statistics estimated from a sample against the sample data.
Influence function of an estimator is linked to the deviation occurring on the estimator because of an infinitely small data contamination. An unlimited influence function means that the deviation can be
infinite.

---

## Paper Implementation Results:

1. can be seen that for UIDS log loss is much less as compared to random subsampling and full datset trained log loss


![](https://i.imgur.com/5lhDF2m.png)

2. test accuracy improved and reach 98.4% on our UIDS technique while for random subsamping and full dataset it is below 95%
 
![](https://i.imgur.com/bZnz1jh.png)


---

## Improvement

basic steps :
1) training model ˆθ on the full Tr,
2) predicting on the Va, then computing the IF; 
3) getting sampling probability from the IF, doing sampling on Tr to get the subset, 
4) acquiring the subset-model ˜θ.

Improved Steps:
#### changed the step 2

* Instead for predicting on just one Va we do k fold cross validation to in our case with k=3
* Average out the IF for each 3 different training set 
* using this average out IF to calculate sampling probability


![](https://i.imgur.com/b6SkBJ0.png)



* We get Improvement in accuracy by 1% 



|          | Paper implementation | with k Fold implementation |     |
| -------- | -------------------- | -------------------------- | --- |
| Accuracy | 98.4281 %            | 99.1678 %    |

---

## Conlclusion:
1. All time biggest question in today's AI is can we achieve better model with less data? 
2. In our work, we implemented a novel Unweighted Influence Data Subsampling (UIDS) method, and prove that the subset-model acquired through our method can outperform the full-set-model and also random subsampling techniques.
3. We also showed that we can further improved the results of paper just by adding one addition step in implementation i.e k-cross validation and computeting Influence function using that.

---


