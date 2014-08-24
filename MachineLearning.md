# Practical Machine Learning, Prediction Assignment
Solomon Davidsohn  
Sunday, August 24, 2014  

##Intro

Using data from accelerometers, 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. In this analysis, we attempt to predict in which way the participants perform the exercise using a data model. In this analysis, we split the data  and use it to create a model using the random forests method, and use an independent data set to check the validity of the model by cross-validation.  

More information: http://groupware.les.inf.puc-rio.br/har

##Analysis

###Download the files
We download the files for our training and testing set and parse them into data frames.


```r
urltrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urltest <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(urltrain,"pml-training.csv")
download.file(urltest,"pml-testing.csv")

traindata <- read.csv("pml-training.csv",stringsAsFactors= FALSE,na.strings=c("NA",""))
testdata <- read.csv("pml-testing.csv", stringsAsFactors=FALSE,na.strings=c("NA",""))
```

###Data Processing

Here we remove the variables with a large amount of missing data.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(ggplot2)

set.seed(300)

#this function finds variables with more than 90% NA's
significantNA <- function(x){
    natable <- table(is.na(x))
    action <- (natable["TRUE"] > natable["FALSE"]*10) #if over 90% of values are NA then return true
    if(is.na(action)) return(FALSE)
    else return(unname(action))
}

listOfSignificantNA <- sapply(traindata,significantNA)
traindataclean <- traindata[,!listOfSignificantNA]

traindataclean$classe <- as.factor(traindataclean$classe)
```

Then, we partition our training data into 80% and 20%. The 20% will be used for cross-validation to check the accuracy of our model.


```r
inTrain <- createDataPartition(y=traindataclean$classe,
                               p=0.80, list=FALSE)
train <- traindataclean[inTrain,]
validate <- traindataclean[-inTrain,] #validation
```

###Creating a model

Here, we create a model using random forests. The training set is subset from 8:60, using the outcome and including only measurements as the predictors.


```r
#create a model using random forests
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
forestfit <- randomForest(classe~.,data=train[,8:60]) #8:59 are the measurements, 60 is the classe
```

###Cross-validation

For our model, I expect high accuracy nearing 1 (100%) because of the large amount of data and variables to predict the outcome. Using the validation data saved for cross-validation, we predict the values on the model and create a confusion matrix to compare values and find the accuracy.


```r
cm1 <- predict(forestfit,validate)
confusionMatrix(validate$classe,cm1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    2    0    0    0
##          B    2  757    0    0    0
##          C    0    0  682    2    0
##          D    0    0    2  641    0
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.996, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.997    0.997    0.995    1.000
## Specificity             0.999    0.999    0.999    0.999    1.000
## Pos Pred Value          0.998    0.997    0.997    0.997    0.999
## Neg Pred Value          0.999    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.998    0.998    0.997    1.000
```
This model has a .998 accuracy on our validation set which is very good. I accept the model and use it to predict the test set.

##Results

Finally, we enter the test data into the model, and produce the predictions for that data.


```r
testpredictions <- predict(forestfit, testdata)
testpredictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
