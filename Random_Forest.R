##Random Forest 
library(randomForest)
stroke_train1$bmi <- as.numeric(stroke_train1$bmi)
rf <- randomForest(stroke~., data=stroke_train1, ntree=500)
rf
rf_pred <- predict(rf, newdata = stroke_test1)
stroke_test1$bmi <- as.numeric(stroke_test1$bmi)
cm <- table(stroke_test1$stroke,rf_pred)
cm
library(caret)
mean(rf_pred == stroke_test1$stroke)
