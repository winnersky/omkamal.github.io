Practical Machine Learning Coursera Project
========================================================

Author: Omar Kamal

## Introduction:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. 
These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior.
The goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

## Steps:
### Reading, cleaning and prepocessing the data.
We start by loading the data from the training set.




```r
raw_training_data <- read.csv("./pml-training.csv",header=T)
```

Converting the data field from character into a Date field.



Dealing with ** DIV/0 ** feilds:
* Some columns have cells containg **DIV/0** cells, so a good strategy to replace those with **NaN**.
* Then looping over those columns that have such cells - and cast their type to numeric.



Then we need to drop un-nessary levels from the data frame.



Separting **factor** variables together and **numeric** variables together.



### Selecting features

We start by getting rid of features that have zero variance by getting their indexs.



Those are the columns that have zero variance.


```
## [1] "kurtosis_yaw_belt"     "skewness_yaw_belt"     "kurtosis_yaw_dumbbell"
## [4] "skewness_yaw_dumbbell" "kurtosis_yaw_forearm"  "skewness_yaw_forearm"
```

Then we create a new data frame with only the columns that have a **non zero variance**.



### Imputing missing data
As a strategy for getting good accuracy, we can use **knnImpute** method to fill-in missing or NaN value with appropriate values based on the K neighbours cases.




Similarly we elemintae columns that have zero variance after imputing the data.frame



Those are the columns to eleminate


```
## [1] "amplitude_yaw_dumbbell" "amplitude_yaw_forearm"
```

### Reducing the features using Principle Component Analysis

A good idea to reduce the number of features is to carry out Principle Component Analysis for the imputed data frame. Doing so, we reduce the number of features. We use a **threshold** of **90%** to determine the number of principle components.
> Note: we also did scale and center the data.



Combining the factors columns with the transformed PCA data frame will compose the new training data frame.



### Subseting the training sample
We will use the **createDataPartition** command in the **caret** package to divide our data frame into:
* A training sample (**75%**) of the data frame.
* A testing sample (**25%**) of the data frame.

The model will be trained on the the training subset and cross-validated on the testing sample.



### Building the model
We will build our model using **Random Forest** technique.


```r
### Fitting the data
rf_model1 <- train(classe ~ ., data=training[,-c(1,4)],method="rf")
```

* See the model


```
## Random Forest 
## 
## 14718 samples
##    32 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.9       0.9    0.004        0.005   
##   20    0.9       0.9    0.004        0.006   
##   30    0.9       0.9    0.005        0.007   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

#### In bound sample accuracy

Checking the accuracy of the **random forest** model.

* Predict the same training data using the model.
* Compute the **confusion matrix** to get the accuracy of the model (_**inbound sample error**_).


```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4138   23   10   10    1
##          B   18 2800   19    8    6
##          C   12   18 2519   30    8
##          D   12    4   13 2360    5
##          E    5    3    6    4 2686
## 
## Overall Statistics
##                                         
##                Accuracy : 0.985         
##                  95% CI : (0.983, 0.987)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.982         
##  Mcnemar's Test P-Value : 0.217         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.989    0.983    0.981    0.978    0.993
## Specificity             0.996    0.996    0.994    0.997    0.999
## Pos Pred Value          0.989    0.982    0.974    0.986    0.993
## Neg Pred Value          0.996    0.996    0.996    0.996    0.998
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.281    0.190    0.171    0.160    0.182
## Detection Prevalence    0.284    0.194    0.176    0.163    0.184
## Balanced Accuracy       0.992    0.989    0.988    0.988    0.996
```

* You can see it is **98.6 %** 

#### Out bound sample accuracy (Cross Validation)

Checking against the testing subset (**Cross Validation**):

* Again predict the test subset using the model.
* Compute the **confusion matrix** for the testing subset.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1384    6    4    3    0
##          B    4  930    4    2    2
##          C    4    7  837    6    1
##          D    3    3    8  791    3
##          E    0    3    2    2  895
## 
## Overall Statistics
##                                         
##                Accuracy : 0.986         
##                  95% CI : (0.983, 0.989)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.983         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.992    0.980    0.979    0.984    0.993
## Specificity             0.996    0.997    0.996    0.996    0.998
## Pos Pred Value          0.991    0.987    0.979    0.979    0.992
## Neg Pred Value          0.997    0.995    0.996    0.997    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.190    0.171    0.161    0.183
## Detection Prevalence    0.285    0.192    0.174    0.165    0.184
## Balanced Accuracy       0.994    0.988    0.987    0.990    0.996
```

* You can see it is **98.4%** 

#### You can see that the accuracy for the testing subset (**cross Validation**) returna very good accuracy **98.4%**.

-----

## Problem Sets
Now, we need to load the 20 cases that we need to predict and do the same exact process for the data including **cleaning** the data, **removing columns** that we did remove in the training set, dealing with DIV/0 in case found, **imputing missing data**, **scaling** and doing **PCA** the same way did in the training data.
> Note: detailed explaination for each step is ignored in this section as it almost identical to the section above.




### Predict the results for the problem set
Using the same Random Forest model we developed in the training round, we predict the outcome for the problem set.



### Results
Here is the results for the problem set.

```
##       problem_id prediction
##  [1,] "1"        "B"       
##  [2,] "2"        "A"       
##  [3,] "3"        "B"       
##  [4,] "4"        "A"       
##  [5,] "5"        "A"       
##  [6,] "6"        "E"       
##  [7,] "7"        "D"       
##  [8,] "8"        "B"       
##  [9,] "9"        "A"       
## [10,] "10"       "A"       
## [11,] "11"       "B"       
## [12,] "12"       "C"       
## [13,] "13"       "B"       
## [14,] "14"       "A"       
## [15,] "15"       "E"       
## [16,] "16"       "E"       
## [17,] "17"       "A"       
## [18,] "18"       "B"       
## [19,] "19"       "B"       
## [20,] "20"       "B"
```

