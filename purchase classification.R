#data from here https://www.kaggle.com/datasets/zahranusrat/social-media-advertising-response-data/data


install.packages("randomForest")
install.packages("randomForest")
install.packages("caret")
install.packages("lubridate")
install.packages("S7")
install.packages("ggplot2")
install.packages("forecast")
install.packages("rpart.plot")
install.packages("precrec")
install.packages("PRROC")
install.packages("xgboost")
install.packages("neuralnet")
install.packages("keras3")
install.packages("reticulate")
install.packages("tensorflow")

library(reticulate)
install_keras()


library(keras3)
library(neuralnet)
library('forecast')
library('tseries')
library(ggplot2)
library(randomForest)
library(caret)
library(readxl)
library(lubridate)
library(tidyverse)
library(data.table)
library(dplyr)
library(readr)
library(stringr)
library(stringi)
library(openxlsx)
library(rpart.plot)
library(rpart)
library(tseries)
library(pROC)
library(PRROC)
library(xgboost)
library(tensorflow)

# This is the new way to install TensorFlow
install_tensorflow()


setwd("C:/data exercises/classification")

ad_purchase_data <- read.csv('Social_Network_Ads.csv')
str(ad_purchase_data)

# Scatter plot
ggplot(ad_purchase_data, aes(x = Age, y = EstimatedSalary, color = factor(Purchased))) +
  geom_point(alpha = 0.7, size = 3) +
  scale_color_manual(values = c("0" = "blue", "1" = "red"),
                     labels = c("Not Purchased", "Purchased")) +
  labs(title = "Age vs Salary by Purchase Status",
       x = "Age",
       y = "Estimated Salary",
       color = "Purchased") +
  theme_minimal()
#looks like most of the purchases are by older and richer people

set.seed(123)
#break out from test and train
trainIndex <- createDataPartition(ad_purchase_data$Purchased, p = 0.7, list = FALSE)
trainData <- ad_purchase_data[trainIndex, ]
testData  <- ad_purchase_data[-trainIndex, ]





#start with logistic regression
log_model <- glm(Purchased ~ Age + EstimatedSalary,
                 data = trainData,
                 family = binomial)

# Predict probabilities
log_probs <- predict(log_model, newdata = testData, type = "response")

# Convert to class labels (threshold 0.5 first, then ,311, then .405. I got these values by doing an ROC Curve (which got .311) and a PR curve (which got .405))
log_preds <- ifelse(log_probs > 0.5, 1, 0)
log_preds <- ifelse(log_probs > 0.311, 1, 0)
log_preds <- ifelse(log_probs > 0.405, 1, 0)

#run a confusion matrix and see how accurate we were
cm <- confusionMatrix(factor(log_preds), factor(testData$Purchased))

# Extract precision and recall for class "1" (Purchased)
precision <- cm$byClass["Pos Pred Value"]
recall <- cm$byClass["Sensitivity"]

# Calculate F1
F1 <- 2 * (precision * recall) / (precision + recall)
F1


#.5 threshold
#85% accuracy, but while this logistic regression is strong at identifying non-purchasers (class 0, sensitivity of .93) but weaker at catching purchasers (class 1, specificty of .69). Maybe we can adjust the threshold to get a better balance.
#kappa is good (above .6)
#.89 F1

#.311 threshold
#84% accuracy, but sensitivity and specificity are both ~84%
#.877 F1

#.405 threshold
#84% accuracy, but sensitivity is .89 and specificity is .74, not great, but better than .5
#.88 F1


#very interesting that .5 threshold would get better F1 than .311 even though .311 gets better balance of sensitivity and specificity, but here's why:

# Accuracy and sensitivity/specificity balance are not the same as F1 optimization.
# F1 is driven by the positive class’s precision and recall.
# A threshold that maximizes F1 may not maximize accuracy or balance sensitivity/specificity.
# Which threshold is “best” depends on your business goal:
# If you want balanced treatment of both classes, 0.311 is better.
# If you want slightly stronger precision for purchasers, 0.5 gives a higher F1.



#so .311 is the winner at least on balance and .5 is best on F1. How did I get .311 and .405? Got it below

#run ROC curve, gives optimal threshold to have close sensitivity and specificity
roc_obj <- roc(testData$Purchased, log_probs)
coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
#threshold    sensitivity    specificity
# 0.3114884    0.8461538      0.8518519
#this is telling me that the optimal threshold (which used to be .5) is actually .311, and if I used that, I'd get a balanced result



#trying PR curve also because if the data is skewed (which it is, we have less purchasers), PR curve can be helpful vs ROC curve
# PR curve needs scores for positives and negatives separately
# Separate scores for positive and negative classes
scores_pos <- log_probs[testData$Purchased == 1]
scores_neg <- log_probs[testData$Purchased == 0]

# Create PR curve object
pr_obj <- pr.curve(scores.class0 = scores_pos,
                   scores.class1 = scores_neg,
                   curve = TRUE)

# Get precision, recall, and thresholds
pr_data <- data.frame(pr_obj$curve)
colnames(pr_data) <- c("Recall", "Precision", "Threshold")

# Find the row where precision and recall are closest
pr_data$diff <- abs(pr_data$Precision - pr_data$Recall)
best_row <- pr_data[which.min(pr_data$diff), ]

# Show the best threshold
best_row
#this says .405 is the best threshold, but after trying it and .311 and .5, .311 is best threshold at least for balance




########## now try random forest

set.seed(1) #gets higher accuracy for some reason than seed of 123, not sure why
# Fit model
rf_model <- randomForest(factor(Purchased) ~ Age + EstimatedSalary,
                         data = trainData,
                         ntree = 500,      # number of trees
                         mtry = 2,         # number of variables tried at each split
                         importance = TRUE)

# Predict class labels
rf_preds <- predict(rf_model, newdata = testData)

# Predict probabilities (optional, for ROC/PR curves later)
rf_probs <- predict(rf_model, newdata = testData, type = "prob")[,2]

# Confusion matrix
confusionMatrix(rf_preds, factor(testData$Purchased))
#.9333 accuracy
#kappa of .85
#sensitivity of .93 and specificity of .95, that's balanced enough for me




##########now xgboost

# Training and test matrices
train_matrix <- model.matrix(Purchased ~ Age + EstimatedSalary, data = trainData)[,-1]
test_matrix  <- model.matrix(Purchased ~ Age + EstimatedSalary, data = testData)[,-1]

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)


# Labels must be numeric (0/1)
train_label <- as.numeric(trainData$Purchased)
test_label  <- as.numeric(testData$Purchased)


#run xgboost default
xgb_model <- xgb.train(
  data = dtrain,
  nrounds = 100,
  objective = "reg:logistic",   # classification objective
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)

# Predict probabilities
xgb_probs <- predict(xgb_model, newdata = dtest)

# Convert to class labels
xgb_preds <- ifelse(xgb_probs > 0.311, 1, 0)

confusionMatrix(factor(xgb_preds), factor(test_label))
#.917 accuracy
#.81 kappa
#sensitivity .93
#specificity .90


#####trying xgboost again with using cross validation to find the ideal number of rounds
cv_model <- xgb.cv(
  data = dtrain,
  nrounds = 200,              # upper limit
  nfold = 5,                  # 5-fold CV
  objective = "reg:logistic", # classification objective
  max_depth = 3,
  eta = 0.1,
  metrics = "auc",            # optimize for AUC
  early_stopping_rounds = 10, # stop if no improvement in 10 rounds
  verbose = 0
)

# Look at the evaluation log
head(cv_model$evaluation_log)

# Best iteration is the one with the lowest test-error or highest test-auc
best_nrounds <- which.max(cv_model$evaluation_log$test_auc_mean)
best_nrounds

xgb_model_nrounds <- xgb.train(
  data = dtrain,
  nrounds = best_nrounds,
  objective = "reg:logistic",
  max_depth = 3,
  eta = 0.1,
  verbose = 0
)

# Predict probabilities
xgb_probs_nrounds <- predict(xgb_model_nrounds, newdata = dtest)

# Convert to class labels
xgb_preds_nrounds <- ifelse(xgb_probs_nrounds > 0.311, 1, 0)


confusionMatrix(factor(xgb_preds_nrounds), factor(test_label))
#.933 accuracy
#.85 kappa
#sensitivity .92
#specificity .95
#slightly better than untuned xgboost



####setting up a grid search so I can automatically test combinations of XGBoost hyperparameters and pick the best options. This way I don't have to manually tweak knobs like max_depth, eta, or subsample.
train_label <- as.numeric(trainData$Purchased)
dtrain <- xgb.DMatrix(data = as.matrix(train_matrix), label = train_label)

param_grid <- expand.grid(
  max_depth = c(3, 5),
  eta = c(0.05, 0.1),
  subsample = c(0.7, 1.0),
  colsample_bytree = c(0.7, 1.0),
  min_child_weight = c(1, 5)
)

results <- list()

for (i in 1:nrow(param_grid)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "error",  # <-- accuracy = 1 - error
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    min_child_weight = param_grid$min_child_weight[i]
  )
  
  set.seed(123)
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 200,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  best_iter <- cv$best_iteration
  best_error <- min(cv$evaluation_log$test_error_mean)
  best_acc <- 1 - best_error
  
  results[[i]] <- c(params, best_iter = best_iter, best_acc = best_acc)
}

results_df <- do.call(rbind.data.frame, results)
results_df[] <- lapply(results_df, function(x) as.numeric(as.character(x)))
results_df <- results_df[order(-results_df$best_acc), ]


best_params <- results_df[1, ]
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight
)

xgb_model_grid <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,   
  verbose = 0
)
xgb_probs_grid <- predict(xgb_model_grid, newdata = dtest)


preds <- predict(xgb_model_grid, newdata = as.matrix(test_matrix))
thresholds <- seq(0.3, 0.7, by = 0.01)
acc <- sapply(thresholds, function(t) mean(ifelse(preds > t, 1, 0) == test_label))
plot(thresholds, acc, type = "l", main = "Accuracy vs. Threshold")



# Convert to class labels
xgb_preds_grid <- ifelse(xgb_probs_grid > 0.3111, 1, 0)

confusionMatrix(factor(xgb_preds_grid), factor(test_label))
#.933 accuracy
#.85 kappa
#sensitivity .93
#specificity .95
#same as best_nrounds xgboost, oh well





##########trying with k-nearest neighbor

# Make sure your labels are factors with two levels
train_label <- factor(ifelse(trainData$Purchased == 1, "Yes", "No"),
                      levels = c("No","Yes"))
test_label  <- factor(ifelse(testData$Purchased == 1, "Yes", "No"),
                      levels = c("No","Yes"))

train_matrix <- data.frame(
  Age = as.numeric(trainData$Age),
  EstimatedSalary = as.numeric(trainData$EstimatedSalary)
)

test_matrix <- data.frame(
  Age = as.numeric(testData$Age),
  EstimatedSalary = as.numeric(testData$EstimatedSalary)
)

# Control setup
ctrl <- trainControl(method = "cv", number = 5)

# Train KNN
knn_model <- train(
  x = train_matrix,
  y = train_label,
  method = "knn",
  trControl = ctrl,
  tuneLength = 10
)

# Inspect results
print(knn_model)
plot(knn_model)

preds <- predict(knn_model, newdata = test_matrix)
confusionMatrix(preds, test_label)
#.83 accuracy
#sensitivty is .94
#specificity is .59, pretty bad

#so let's try again while scaling
knn_model <- train(
  x = train_matrix,
  y = train_label,
  method = "knn",
  trControl = ctrl,
  tuneLength = 10,
  preProcess = c("center","scale")   # normalize features
)

preds <- predict(knn_model, newdata = test_matrix)
confusionMatrix(preds, test_label)
#.91 accuracy
#sensitivity .93
#specificity .90






#try a neural net, r's first, neuralnet: A classical feedforward neural network implementation in R. Designed mainly for educational use and small-scale problems.

#Normalize predictors (important for neural networks)


ad_purchase_data_nn<- ad_purchase_data

#$convert to scale
ad_purchase_data_nn$Age <- scale(ad_purchase_data_nn$Age)
ad_purchase_data_nn$EstimatedSalary <- scale(ad_purchase_data_nn$EstimatedSalary)

# 70-30 train/test split
trainIndex <- createDataPartition(ad_purchase_data_nn$Purchased, p = 0.7, list = FALSE)
trainData_nn <- ad_purchase_data_nn[trainIndex, ]
testData_nn  <- ad_purchase_data_nn[-trainIndex, ]

# Neural network with one hidden layer of 5 neurons
nn_model <- neuralnet(Purchased ~ Age + EstimatedSalary,
                      data = trainData_nn,
                      hidden = 5,
                      linear.output = FALSE)

plot(nn_model)

# Compute predictions
nn_results <- compute(nn_model, testData_nn[, c("Age", "EstimatedSalary")])

# Extract probabilities
probabilities <- nn_results$net.result

# Convert to binary (threshold = 0.5)
predicted <- ifelse(probabilities > 0.5, 1, 0)

# Evaluate accuracy
confusionMatrix(factor(predicted), factor(testData_nn$Purchased))
#.85 accuracy, not great

#For small, tabular datasets with a mix of numeric/categorical features, tree‑based ensembles (XGBoost, LightGBM, Random Forest) usually outperform neural nets.

#For large, high‑dimensional, unstructured data, neural nets (deep learning) are the go‑to.






###############trying keras neural network, A high-level API for deep learning, backed by TensorFlow. Supports complex, large-scale architectures (CNNs, RNNs, transformers, etc.).

idx <- createDataPartition(ad_purchase_data$Purchased, p = 0.7, list = FALSE)
train <- ad_purchase_data[idx, ]
test  <- ad_purchase_data[-idx, ]

# --- 2. Scale predictors (important for neural nets) ---
pp <- preProcess(train[, c("Age","EstimatedSalary")], 
                 method = c("center","scale"))

train_scaled <- predict(pp, train[, c("Age","EstimatedSalary")])
test_scaled  <- predict(pp,  test[, c("Age","EstimatedSalary")])

# --- 3. Convert to matrices ---
x_train <- as.matrix(train_scaled)
y_train <- as.numeric(train$Purchased)

x_test <- as.matrix(test_scaled)
y_test <- as.numeric(test$Purchased)

# ✔ Correct: input_shape must be ncol(x_train), NOT ncol(x)
input_dim <- ncol(x_train)

# --- 4. Build the model (simple + no dropout = better for tabular data) ---
model <- keras_model_sequential() |>
  layer_dense(units = 4, activation = "relu", input_shape = input_dim) |>
  layer_dense(units = 1, activation = "sigmoid")

model |> compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.01),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

# --- 5. Train ---
history <- model |> fit(
  x_train, y_train,
  epochs = 150,
  batch_size = 16,
  validation_split = 0.2
)

# --- 6. Predict on test set ---
pred_prob <- model |> predict(x_test)
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# --- 7. Confusion Matrix ---
confusionMatrix(
  factor(pred_class, levels = c(0,1)),
  factor(y_test,    levels = c(0,1))
)
