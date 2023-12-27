---
title: "Practical Machine Learning Course Project"
author: "Mitchell Clarke"
date: "2023-12-27"
output: 
  html_document:
    keep_md: true
---



# Introduction and Background
Introduction\
This is the final project for the Coursera Practical Machine Learning Course for the Data Science: Statistics and Machine Learning Specialization. 

Background\
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).Using a random forest model, we will predict the fashion in which a testing set of exercises was performed.

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: 
http://groupware.les.inf.puc-rio.br/har


# Load the Required Libraries and Read/Download Files

Packages

```r
library(caret)
```

```
## Loading required package: ggplot2
```

```
## Loading required package: lattice
```

```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(corrplot)
```

```
## corrplot 0.92 loaded
```
Read in Training and Testing Sets

```r
Training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))

Testing <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
```

# Data Cleaning and Preparation
First, look at the number of observations and variables of both datasets.

```r
dim(Training)
```

```
## [1] 19622   160
```

```r
dim(Testing)
```

```
## [1]  20 160
```

Remove the first 7 variables that are unnecessary to our model, and also any columns that contain NA values signifying missing data.

```r
TrainingClean <- Training %>% dplyr::select(-c(X:num_window)) %>% dplyr::select_if(~ !any(is.na(.)))
```

We also want to remove variables with near zero variance since these are not predictive of classe. We use the nearZeroVar function in the caret package to do this.

```r
RemTrain <- nearZeroVar(TrainingClean)

TrainingClean <- TrainingClean[,-RemTrain]
```

Now we get the same columns for the testing set using dplyr.

```r
TestingCleanNames <- intersect(names(TrainingClean), names(Testing))

TestingClean <- Testing %>% dplyr::select(all_of(TestingCleanNames))
```
We now have two datasets with the same variables (excluding the classe variable which is excluded from our quiz test set).

```r
dim(TrainingClean)
```

```
## [1] 19622    53
```

```r
dim(TestingClean)
```

```
## [1] 20 52
```

# Partition Data Into a Testing and Training Set
We will use 70% for the training set and 30% for the testing set. We will only use the true testing set (with 20 observations) for the quiz results, so we want another testing set (called validation) to test the accuracy and out of sample error of our model.

Set seed for reproducibility.

```r
set.seed(3421)
```

```r
inTraining <- createDataPartition(TrainingClean$classe, p=0.7, list = FALSE)

TrainingData <- TrainingClean[inTraining,]
ValidationData <- TrainingClean[-inTraining,]
```

# Fitting our Model
For simplicity, we will just use a random forest model with 5-fold cross validation. 

```r
ctrl <- trainControl(method = "cv", number = 5)

rfmodel <- train(classe ~ ., data=TrainingData, method="rf", trControl=ctrl, prox= TRUE)
rfmodel
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10988, 10990, 10991, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9904638  0.9879361
##   27    0.9906099  0.9881215
##   52    0.9842761  0.9801095
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

# Predictions on Our Test Set
Now we use the rfmodel to predict on our ValidationData dataset we partitioned before.

```r
rfprediction <- predict(rfmodel, ValidationData)
cfnmatrix <- confusionMatrix(as.factor(ValidationData$classe), rfprediction)
cfnmatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    1    0    0
##          B   13 1122    3    0    1
##          C    0    3 1020    3    0
##          D    0    0   12  951    1
##          E    0    1    2    6 1073
## 
## Overall Statistics
##                                           
##                Accuracy : 0.992           
##                  95% CI : (0.9894, 0.9941)
##     No Information Rate : 0.2863          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9899          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9923   0.9956   0.9827   0.9906   0.9981
## Specificity            0.9995   0.9964   0.9988   0.9974   0.9981
## Pos Pred Value         0.9988   0.9851   0.9942   0.9865   0.9917
## Neg Pred Value         0.9969   0.9989   0.9963   0.9982   0.9996
## Prevalence             0.2863   0.1915   0.1764   0.1631   0.1827
## Detection Rate         0.2841   0.1907   0.1733   0.1616   0.1823
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9959   0.9960   0.9907   0.9940   0.9981
```
# Accuracy and Out of Sample Error
Accuracy from our confusion matrix is 0.992.

```r
postResample(as.factor(ValidationData$classe), rfprediction)
```

```
##  Accuracy     Kappa 
## 0.9920136 0.9898962
```
The out of sample error is 0.0079.

```r
OutofSampleError <- 1 - cfnmatrix$overall[[1]]
OutofSampleError
```

```
## [1] 0.007986406
```

# Predictions on the Quiz Test Set

```r
testquizprediction <- predict(rfmodel, TestingClean)
testquizprediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
\newpage
# Appendix 
Figure 1 Correlation Plot

```r
Correlations <- cor(sapply(TrainingData[,-53], as.numeric))
corrplot(Correlations)
```

![](Practical-Machine-Learning-Course-Project_files/figure-html/unnamed-chunk-15-1.png)<!-- -->
\
Figure 2 Model Plot

```r
plot(rfmodel)
```

![](Practical-Machine-Learning-Course-Project_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

