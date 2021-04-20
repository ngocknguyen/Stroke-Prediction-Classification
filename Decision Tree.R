##Decision Tree
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
#Symmetric 
stroke_rpart <- rpart(formula = stroke~., data=stroke_train1,method="class") 
stroke_rpart
fancyRpartPlot(stroke_rpart)
pred_tree <- predict(stroke_rpart,type="class")
table(stroke_train1$stroke, pred_tree, dnn = c("True", "Pred"))
#Asymmetric
stroke_rpart1 <- rpart(formula = stroke~., data=stroke_train1,method="class",parms = list(loss=matrix(c(0,5,1,0), nrow = 2)))
stroke_rpart1
fancyRpartPlot(stroke_rpart1)
pred_tree1 <- predict(stroke_rpart1,type="class")
table(stroke_train1$stroke, pred_tree1, dnn = c("True", "Pred"))
#Out-of-sample performance
pred_tree_test <- predict(stroke_rpart1,stroke_test1,type="class")
table(stroke_test1$stroke, pred_tree_test, dnn = c("True", "Pred"))
#Finding the cost
cost <- function(r, phat){
  weight1 <- 5
  weight0 <- 1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut)
  c0 <-(r==0)&(phat>pcut)
  return(mean(weight1*c1+weight0*c0))
}
cost(stroke_test1$stroke, predict(stroke_rpart1, stroke_test1, type="prob"))
#Finding the AUC on training data
library(ROCR)
stroke_train_prob_rpart = predict(stroke_rpart1, stroke_train1, type="prob")
pred_train_rpart = prediction(stroke_train_prob_rpart[,2], stroke_train1$stroke)
perf_train = performance(pred_train_rpart,"tpr","fpr")
plot(perf_train, colorize=TRUE)
slot(performance(pred_train_rpart, "auc"), "y.values")[[1]]
#Finding the AUC on testing data
stroke_test_prob_rpart = predict(stroke_rpart1, stroke_test1, type="prob")
pred_rpart = prediction(stroke_test_prob_rpart[,2], stroke_test1$stroke)
perf = performance(pred_rpart,"tpr","fpr")
plot(perf, colorize=TRUE)
slot(performance(pred_rpart, "auc"), "y.values")[[1]]