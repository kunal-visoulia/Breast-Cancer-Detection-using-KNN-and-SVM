# Breast-Cancer-Detection-using-KNN-and-SVM

# _RESULT_
I built a system for Benign or Malignant cancer classification based on various features like cell shape, cell size, mitoses rate, etc
On the given dataset, KNN performed better than SVM possibly signifying that the ***dataset is not linearly separable*** (there could be other reasons also, like, outliers in the data set). This conforms to what scatter plot of features from dataset showed.

<img src="https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/10.png"  />

### DATASET
Link for the dataset used
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

**_DATA PREPROESSING_**
I removed empty(?) values from dataset and replaced it with some very large negative value so it can be ignored by training algorithm. I also removed id column to as it does not provide any usefull information.

##  [K-nearest neighbors (KNN)](https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7)
A **supervised** learning algorithm can be used for both classification and regression predictive problems.

KNN is a [**non-parametric**](https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/), [**lazy learning**](https://sebastianraschka.com/faq/docs/lazy-knn.html) algorithm.
>When we say a technique is non-parametric , it means that it does not make any assumptions on the underlying data distribution. **Nonparametric methods are good when you have a lot of data and no prior knowledge, and when you don’t want to worry too much about choosing just the right features.**
>KNN is also a lazy algorithm (as opposed to an eager algorithm).It does not use the training data points to do any generalization, i.e., it doesn’t learn a discriminative function from the training data but “memorizes” the training dataset instead. 

KNN Algorithm is based on feature similarity: How closely out-of-sample features resemble our training set determines how we classify a given data point:

![](https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/1.png)

*Example of k-NN classification. The test sample (inside circle) should be classified either to the first class of blue squares or to the second class of red triangles. If k = 3 (outside circle) it is assigned to the second class because there are 2 triangles and only 1 square inside the inner circle. If, for example k = 5 it is assigned to the first class (3 squares vs. 2 triangles outside the outer circle).*

Some pros and cons of KNN
 
Pros:
- No assumptions about data — useful, for example, for nonlinear data
- the training phase is pretty fast ( no explicit training phase or it is very minimal)
- High accuracy (relatively) — it is pretty high but not competitive in comparison to better supervised learning models

Cons:
- Computationally expensive — because the algorithm stores all of the training data (Lack of generalization)
- High memory requirement
- Prediction stage might be slow (with big N)

## [Support Vector Machine (SVM)](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)
https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93
hyperplane — a generalization of the two-dimensional line and three-dimensional plane to an arbitrary number of dimensions.

Given labeled training data **(supervised learning)**, The objective of the support vector machine algorithm is to find the hyperplane that has the maximum margin in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

![](https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/2.png)

If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane.

Can you draw a separating line in this plane?

![](https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/3.png)

if we plot in z-axis, a clear separation is visible and a line can be drawn .

![](https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/4.png)

When we transform back this line to original plane, it maps to circular boundary as shown. These transformations are called [**kernels**](https://towardsdatascience.com/kernel-function-6f1d2be6091).

![](https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/5.png)

What in this case?

<img src="https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/6.png" width="440" />

<p float="left">
  <img src="https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/7.png" width="440" />
  <img src="https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/8.png" width="440" /> 
</p>

Left one has some misclassification due to lower regularization value(*C parameter*). Higher value leads to results like right one.

**_The Bias-Variance Trade-off_**:[High bias is underfitting and high variance is overfitting](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/). We need to find a golden mean between these two.

>Large C: Lower bias, high variance.

>Small C: Higher bias, low variance.

If a learning algorithm is **suffering from high bias**, getting more training data will not (by itself) help much. Adding features, Adding polynomial features, Increasing C: Fixes high bias.

If a learning algorithm is **suffering from high variance**, Getting more training data, Trying small set of features and Decreasing C: Fixes high Variance.

Compared to both [logistic regression](https://towardsdatascience.com/support-vector-machine-vs-logistic-regression-94cc2975433f) and neural networks, the SVM sometimes gives a cleaner, and sometimes more powerful way of learning complex non-linear functions.


## [k-fold Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

>there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “validation set”: training proceeds on the training set, after which evaluation is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.
>However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.
>A solution to this problem is a procedure called cross-validation (CV for short)

Cross-validation is a statistical method used to estimate the skill of machine learning models on limited data sample.
```
sklearn.model_selection.KFold(n_splits,random_state)
```

Split dataset into k consecutive folds (without shuffling by default).Each fold is then used once as a validation((i.e., it is used as a test set to compute a performance measure such as accuracy) while the k - 1 remaining folds form the training set.
The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
>A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV.
The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.

### Precision

![alt text](https://github.com/kunal-visoulia/Breast-Cancer-Detection-using-KNN-and-SVM/blob/master/images/9.png)

It is the Ratio of correctly predicted positive observations to the total predicted positive observations.
##### Precision = TP/TP+FP

>Of all passengers that labeled as survived, how many actually survived? 
>High precision relates to the low false positives (**Benign cells classified as malignant or vice-vers**).

### Recall
It is the Ratio of correctly predicted positive observations to the all observations in actual class-yes. 
##### Recall = TP/TP+FN

>Of all the passengers that truly survived, how many did we label? 
>  "a measure of false negatives (**cells getting through algorithm without being labelled**)"

### F1-Score
It is the Weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.
##### F1 Score = 2*(Recall * Precision) / (Recall + Precision)

