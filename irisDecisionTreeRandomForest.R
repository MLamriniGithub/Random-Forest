# Title: Exploratory Data Analysis and Classification Models on Iris Dataset

# Description: This R code performs exploratory data analysis on the Iris dataset, including data dimensionality 
# inspection, summary statistics, visualizations, and correlation analysis. It then builds a decision 
# tree and a random forest classification model for species prediction, evaluates their performance on 
# training and testing datasets,and assesses model metrics such as precision, recall, accuracy, 
# and error rates

# free memory
rm(list = ls())
gc()

###################################################
dim(iris)
names(iris)
str(iris)
attributes(iris)

###################################################
iris[1:5,]
head(iris)
tail(iris)

###################################################
iris[1:10, "Sepal.Length"]
iris$Sepal.Length[1:10]

###################################################
summary(iris)

###################################################
quantile(iris$Sepal.Length)
quantile(iris$Sepal.Length, c(.1, .3, .65))

###################################################
var(iris$Sepal.Length)
hist(iris$Sepal.Length)

###################################################
plot(density(iris$Sepal.Length))

###################################################
table(iris$Species)
pie(table(iris$Species))

###################################################
barplot(table(iris$Species))

###################################################
cov(iris$Sepal.Length, iris$Petal.Length)
cov(iris[,1:4])
cor(iris$Sepal.Length, iris$Petal.Length)
cor(iris[,1:4])


###################################################
aggregate(Sepal.Length ~ Species, summary, data=iris)

###################################################
boxplot(Sepal.Length~Species, data=iris)

###################################################
with(iris, plot(Sepal.Length, Sepal.Width, col=Species, pch=as.numeric(Species)))



###################################################
pairs(iris)


###################################################
library(scatterplot3d)
scatterplot3d(iris$Petal.Width, iris$Sepal.Length, iris$Sepal.Width)




###################################################
### code chunk number 23: ch-exploration.rnw:257-259
###################################################
library(MASS)
parcoord(iris[1:4], col=iris$Species)


###################################################
library(lattice)
parallelplot(~iris[1:4] | Species, data=iris)


###################################################
library(ggplot2)
qplot(Sepal.Length, Sepal.Width, data=iris, facets=Species ~.)


###################################################
str(iris)
set.seed(1234) 
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

myFormula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
library(rpart)

iris_rpart <- rpart(myFormula, data=trainData, 
                    control = rpart.control(minsplit = 5))
attributes(iris_rpart)
print(iris_rpart)

###################################################
library(rpart.plot)
prp(iris_rpart,extra=1)
iris_rpart$control$cp
plotcp(iris_rpart) 

iris_rpart <- rpart(myFormula, data=trainData, 
                    control = rpart.control(minsplit = 5, cp=0.2))
print(iris_rpart)

library(rpart.plot)
prp(iris_rpart,extra=1)

###################################################
trainPred <- predict(iris_rpart, newdata = trainData, type="class")
table(trainPred, trainData$Species)

testPred <- predict(iris_rpart, newdata = testData, type="class")
table(testPred, testData$Species)
###################################################
evaluator <- function(model,test,n) {
  #predictions used for the confusion matrix
  predictions<-predict(model,test,type="class")
  #Confusion Matix
  cm<-table(predictions,test[,n])
  #Calculate Recall taking True Positives and
  #divised by True Positive and False Positive for each Trhee Class
  precision<-diag(cm)/rowSums(cm)
  #Calculate Recall taking True Positives and 
  #divised by True Positive and False Negatives for each Trhee Class
  recall <- (diag(cm) / colSums(cm))
  #Calculate Accuracy taking the total of the True Positives of every class and 
  #divised by the total of the training set
  accuracy<- sum(diag(cm)) / sum(cm)
  #Calculate the Error Rate taking the not correctly predicted instances 
  #divised by the total rows of the training set
  #errorRate<- sum(iris.test$Species != predict(model, iris.test[,1:4], type="raw")) / nrow(iris.test)
  errorRate<-1-accuracy
  print(cm)
  print("--------------Precision-------------------------")
  print(precision)
  print("--------------Recall-------------------------")
  print(recall)
  print("--------------Accuracy-------------------------")
  print(accuracy)
  print("--------------Error-------------------------")
  print(errorRate)
}

evaluator(iris_rpart,testData,5)
###################################################

library(randomForest)
rf <- randomForest(Species ~ ., data=trainData, ntree=100)
table(predict(rf), trainData$Species)
print(rf)
attributes(rf)
rf <- randomForest(Species ~ ., data=trainData, ntree=100, mtry=3)
###################################################
importance(rf)
varImpPlot(rf)

###################################################
irisPred <- predict(rf, newdata=testData)
table(irisPred, testData$Species)

evaluator(rf,testData,5)



