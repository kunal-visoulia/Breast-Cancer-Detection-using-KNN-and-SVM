# Breast-Cancer-Detection-using-KNN-and-SVM

### DATASET
Link for the dataset used
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

### [k-fold Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

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
![alt text](https://blog.exsilio.com/wp-content/uploads/2016/09/table-blog.png)
It is the Ratio of correctly predicted positive observations to the total predicted positive observations.
###### Precision = TP/TP+FP

>Of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate.

### Recall
It is the Ratio of correctly predicted positive observations to the all observations in actual class-yes. 
##### Recall = TP/TP+FN

>Of all the passengers that truly survived, how many did we label? "a measure of false negatives"

### F1-Score
It is the Weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.
##### F1 Score = 2*(Recall * Precision) / (Recall + Precision)

