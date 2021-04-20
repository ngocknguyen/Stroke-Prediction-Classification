##load data
healthcare.dataset.stroke.data <- read.csv("~/Desktop/Data Science/R/Stroke prediction/healthcare-dataset-stroke-data.csv")
#EDA
dim(healthcare.dataset.stroke.data)
str(healthcare.dataset.stroke.data)
summary(healthcare.dataset.stroke.data)
healthcare.dataset.stroke.data$gender <- as.factor(healthcare.dataset.stroke.data$gender)
healthcare.dataset.stroke.data$ever_married <- as.factor(healthcare.dataset.stroke.data$ever_married)
healthcare.dataset.stroke.data$work_type <- as.factor(healthcare.dataset.stroke.data$work_type)
healthcare.dataset.stroke.data$Residence_type <- as.factor(healthcare.dataset.stroke.data$Residence_type)
healthcare.dataset.stroke.data$smoking_status <- as.factor(healthcare.dataset.stroke.data$smoking_status)
healthcare.dataset.stroke.data$hypertension <- as.factor(healthcare.dataset.stroke.data$hypertension)
healthcare.dataset.stroke.data$heart_disease <- as.factor(healthcare.dataset.stroke.data$heart_disease)
healthcare.dataset.stroke.data$stroke <- as.factor(healthcare.dataset.stroke.data$stroke)
healthcare.dataset.stroke.data$bmi <- as.factor(healthcare.dataset.stroke.data$bmi)
##Run summary again to check if the data is correct
summary(healthcare.dataset.stroke.data)
## Data Viz
boxplot(healthcare.dataset.stroke.data[,1:6], col = "rosybrown")
boxplot(healthcare.dataset.stroke.data[,7:12], col = "rosybrown")
barplot(table(healthcare.dataset.stroke.data$stroke), xlab = "No Stroke VS Stroke ", main= "Stroke frequency", col=c("pink", "skyblue"))
prop.table(table(healthcare.dataset.stroke.data$stroke))
## find missing values
summary(healthcare.dataset.stroke.data$bmi)
table(healthcare.dataset.stroke.data$bmi)
## split dataset 
subset <- sample(nrow(healthcare.dataset.stroke.data), nrow(healthcare.dataset.stroke.data)*0.8)
stroke_train = healthcare.dataset.stroke.data[subset,]
stroke_test = healthcare.dataset.stroke.data[-subset,]
## logistic regression
model_1 <- glm(stroke~.-bmi,family = binomial,data=stroke_train)
summary(model_1)
BIC(model_1)
model_2 <- glm(stroke~.,family = binomial,data=stroke_train)
summary(model_2)
BIC(model_2)
## variable selection to find the best model
#using backward BIC
stroke_backBIC <- step(model_1, k=log(nrow(stroke_train)))
summary(stroke_backBIC)
BIC(stroke_backBIC)
# this given the warning message fitted 0 or 1 probabilities occured
#backward selection
stroke_back <- step(model_1)
summary(stroke_back)
BIC(stroke_back)
## in sample prediction, ROC Curve, and Misclassification Rate
insample_pred <- predict(stroke_back, type ='response')
hist(insample_pred)
table(stroke_train$stroke, (insample_pred > 0.5)*1, dnn=c("Truth","Predicted"))
table(stroke_train$stroke, (insample_pred > 0.2)*1, dnn=c("Truth","Predicted"))
#ROC Curve
library(ROCR)
pred_train <- predict(stroke_back, type="response")
pred <- prediction(pred_train,stroke_train$stroke)
perf <- performance(pred,'tpr','fpr')
plot(perf, colorize=TRUE)
#AUC
unlist(slot(performance(pred, "auc"), "y.values"))
## out-of-sample performance
outsample_pred <- predict(stroke_back, newdata=stroke_test, type = "response")
hist(outsample_pred)
table(stroke_test$stroke, (outsample_pred > 0.5)*1, dnn=c("Truth","Predicted"))
table(stroke_test$stroke, (outsample_pred > 0.2)*1, dnn=c("Truth","Predicted"))
library(ROCR)
pred_test <- prediction(outsample_pred,stroke_test$stroke)
perf_test <- performance(pred_test,'tpr','fpr')
plot(perf_test, colorize=TRUE)
unlist(slot(performance(pred_test, "auc"), "y.values"))
## Precision Recall Curve
#in-sample
library(PRROC)
score1= insample_pred[stroke_train$stroke==1]
score0= insample_pred[stroke_train$stroke==0]
roc= roc.curve(score1, score0, curve = T)
roc$auc
pr= pr.curve(score1, score0, curve = T)
pr
plot(pr, main="In-sample PR curve")
#out-of-sample
score1.test= outsample_pred[stroke_test$stroke==1]
score0.test= outsample_pred[stroke_test$stroke==0]
roc.test= roc.curve(score1.test, score0.test, curve = T)
roc.test$auc
pr.test= pr.curve(score1.test, score0.test, curve = T)
pr.test
plot(pr.test, main="Out-of-sample PR curve")

### ROSE - balancing imbalance data on stroke
library(ROSE)
newdata <- ovun.sample(stroke~., data = healthcare.dataset.stroke.data,method="over",N=9722)$data
table(newdata$stroke)
summary(newdata)
barplot(table(newdata$stroke), xlab = "No Stroke VS Stroke ", main= "Stroke frequency", col=c("pink", "skyblue"))
subset <- sample(nrow(newdata), nrow(newdata)*0.8)
stroke_train1 = newdata[subset,]
stroke_test1 = newdata[-subset,]
#logistic model using balanced data
new_model <-  glm(stroke~.-bmi,family = binomial,data=stroke_train1)
summary(new_model)
#variable selection
newmodel_backward <- step(new_model)
summary(newmodel_backward)
newmodel_backwardBIC <- step(new_model, k=log(nrow(stroke_train1)))
summary(newmodel_backwardBIC)
AIC(newmodel_backward)
AIC(newmodel_backwardBIC)
#Prediction, PR Curve on training data
newmodel.insample.pred<- predict(newmodel_backward, type ='response')
hist(newmodel.insample.pred)
score1.resample= newmodel.insample.pred[stroke_train1$stroke==1]
score0.resample= newmodel.insample.pred[stroke_train1$stroke==0]
pr.newtrain= pr.curve(score1.resample, score0.resample, curve = T)
pr.newtrain
plot(pr.newtrain, main="In-sample PR curve")
table(stroke_train1$stroke, (newmodel.insample.pred> 0.2)*1, dnn=c("Truth","Predicted"))
#Prediction, PR Curve on testing data
newmodel.outsample.pred <- predict(newmodel_backward,newdata= stroke_test1,type ='response')
hist(newmodel.outsample.pred)
score1.resample.test= newmodel.outsample.pred[stroke_test1$stroke==1]
score0.resample.test= newmodel.outsample.pred[stroke_test1$stroke==0]
pr.newtest= pr.curve(score1.resample.test, score0.resample.test, curve = T)
pr.newtest
plot(pr.newtest, main="Out-of-sample PR curve")
table(stroke_test1$stroke, (newmodel.outsample.pred > 0.2)*1, dnn=c("Truth","Predicted"))
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