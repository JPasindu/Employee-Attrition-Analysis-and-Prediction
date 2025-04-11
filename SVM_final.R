#----------------- SVM Model ----------------------------------------------------------------
library(caret)
library(pROC)
library(ROSE) 
library(ggplot2)

df_for_SVM=read.csv("df.csv")

# Set categorical (nominal) variables as factors
nominal_vars <- c("BusinessTravel", "Department", "State",
                  "MaritalStatus", "Attrition", "OverTime",
                  "EducationField_Grouped", "Ethnicity_Grouped",
                  "Gender_Grouped", "JobRole_Grouped")

df_for_SVM[nominal_vars] <- lapply(df_for_SVM[nominal_vars], as.factor)

# Set ordinal variables as ordered factors or numeric (preserving ranking)
# Assuming higher numbers = higher satisfaction/education
ordinal_vars <- c("Education", "StockOptionLevel", "EnvironmentSatisfaction",
                  "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                  "ManagerRating")

df_for_SVM[ordinal_vars] <- lapply(df_for_SVM[ordinal_vars], function(x) as.numeric(as.character(x)))

# Ratio variables are already numeric, just ensure that
ratio_vars <- c("Age", "DistanceFromHome..KM.", "Salary",
                "YearsSinceLastPromotion",
                "TrainingOpportunitiesWithinYear", "TrainingOpportunitiesTaken")

df_for_SVM[ratio_vars] <- lapply(df_for_SVM[ratio_vars], as.numeric)

# Final structure check
str(df_for_SVM)

#---------------------
# Step 3: Separate target variable (e.g., Attrition)
target <- df_for_SVM$Attrition
df_for_SVM$Attrition <- NULL  # Remove temporarily

# Step 4: Dummy encoding
dummies <- dummyVars(" ~ .", data = df_for_SVM)
df_encoded <- as.data.frame(predict(dummies, newdata = df_for_SVM))

# Step 5: Add target back
df_encoded$Attrition <- target

# Step 6: Confirm structure
str(df_encoded)

# split
set.seed(123)

train_index <- createDataPartition(df_encoded$Attrition, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]


# Define training control with SMOTE and 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Define tuning grid for SVM with radial kernel
tune_grid <- expand.grid(
  C = seq(0,10,1),      # Regularization parameter
  sigma = c(0.01,0.1,0.5)  # Kernel width
)

# Train SVM model
# Set seed for reproducibility
set.seed(100)

svm_model <- train(
  Attrition ~ .,
  data = train_data,
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = tune_grid
)

#saved_svm=svm_model
# Predict classes
train_pred_class <- predict(svm_model, train_data, type = "raw")
test_pred_class <- predict(svm_model, test_data, type = "raw")

# Accuracy tables
tra_tab <- table(train_data$Attrition, train_pred_class)
test_tab <- table(test_data$Attrition, test_pred_class)
tra_accuracy <- sum(diag(tra_tab)) / sum(tra_tab)
test_accuracy <- sum(diag(test_tab)) / sum(test_tab)
tra_accuracy
test_accuracy
tra_accuracy-test_accuracy
# Predict probabilities
train_pred_prob <- predict(svm_model, train_data, type = "prob")
test_pred_prob <- predict(svm_model, test_data, type = "prob")

# Evaluate performance on training data
train_conf_mat <- confusionMatrix(train_pred_class, train_data$Attrition, positive = "Yes")
cat("Training Precision:", train_conf_mat$byClass["Precision"], "\n")
cat("Training Recall:", train_conf_mat$byClass["Recall"], "\n")
cat("Training F1 Score:", train_conf_mat$byClass["F1"], "\n\n")

# Evaluate performance on test data
test_conf_mat <- confusionMatrix(test_pred_class, test_data$Attrition, positive = "Yes")
cat("Test Precision:", test_conf_mat$byClass["Precision"], "\n")
cat("Test Recall:", test_conf_mat$byClass["Recall"], "\n")
cat("Test F1 Score:", test_conf_mat$byClass["F1"], "\n")


# Calculate AUC
train_roc <- roc(train_data$Attrition, train_pred_prob$Yes)
test_roc <- roc(test_data$Attrition, test_pred_prob$Yes)

cat("Training AUC:", auc(train_roc), "\n")
cat(" Test AUC:", auc(test_roc), "\n")
