#----------------- XGBoost Model ----------------------------------------------------------------
# Load necessary libraries
library(caret)
library(pROC)
library(ROSE)   # for SMOTE
library(ggplot2)

train_data=read.csv("train_data.csv")
train_data=train_data[,-1]
test_data=read.csv("test_data.csv")
test_data=test_data[,-1]
head(test_data)

nominal_vars_keep <- c("BusinessTravel", "Department", "State",
                       "MaritalStatus", "Attrition", "OverTime",
                       "EducationField_Grouped", "Ethnicity_Grouped",
                       "Gender_Grouped", "JobRole_Grouped")
ordinal_vars_keep <- c("Education", "StockOptionLevel", "EnvironmentSatisfaction",
                       "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                       "ManagerRating")
ratio_vars_keep <- c("Age", "DistanceFromHome..KM.", "Salary",
                     "YearsSinceLastPromotion",
                     "TrainingOpportunitiesWithinYear","TrainingOpportunitiesTaken")

vars_keep <- c(nominal_vars_keep, ordinal_vars_keep, ratio_vars_keep)

# Convert nominal variables to factor
train_data[nominal_vars_keep] <- lapply(train_data[nominal_vars_keep], factor)

# Convert ordinal variables to ordered factors
train_data[ordinal_vars_keep] <- lapply(train_data[ordinal_vars_keep], function(x) {
  ordered(x, levels = sort(unique(x)))
})

# Ensure ratio variables are numeric
train_data[ratio_vars_keep] <- lapply(train_data[ratio_vars_keep], as.numeric)

# Convert nominal variables to factor
test_data[nominal_vars_keep] <- lapply(test_data[nominal_vars_keep], factor)

# Convert ordinal variables to ordered factors
test_data[ordinal_vars_keep] <- lapply(test_data[ordinal_vars_keep], function(x) {
  ordered(x, levels = sort(unique(x)))
})

# Ensure ratio variables are numeric
test_data[ratio_vars_keep] <- lapply(test_data[ratio_vars_keep], as.numeric)
str(train_data)

# Define training control with SMOTE and 10-fold cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

tune_grid <- expand.grid(
  nrounds = 200,                  # Number of boosting iterations
  max_depth = seq(1,10),      # Tree depth
  eta = c(0.01, 0.05, 0.1, 0.15, 0.2),       # Learning rate
  gamma = 0,                      # Minimum loss reduction
  colsample_bytree = 0.8,         # Subsample ratio of columns
  min_child_weight = 1,           # Minimum sum of instance weight in a child
  subsample = 0.8                 # Subsample ratio of the training instance
)

tune_grid <- expand.grid(
  nrounds = c(50, 75, 100, 200),
  max_depth = seq(1,10),
  eta = c(0.01, 0.05),               # Smaller learning rate
  gamma = c(0, 1, 2),                   # Adds regularization
  colsample_bytree = 0.7,
  min_child_weight = c(1, 2, 3, 4,  5),        # Forces larger leaf nodes
  subsample = 0.7
)

# final
tune_grid <- expand.grid(
  nrounds = c(250),
  max_depth = c(6),
  eta = c(0.0005),
  gamma = c(0),
  colsample_bytree = c(0.6),
  min_child_weight = c(10),
  subsample = c(0.7)
)

# Train XGBoost model
# Set seed for reproducibility
set.seed(100)
xgb_model <- suppressWarnings(train(
  Attrition ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = tune_grid
))
#saved_xgb_model_best_final=xgb_model
#xgb_model=saved_xgb_model_04
# Predict classes
train_in <- as.numeric(rownames(xgb_model$trainingData))
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]
#----------------------------------------------------------------------
train_pred_class <- predict(xgb_model, train_data, type = "raw")
test_pred_class <- predict(xgb_model, test_data, type = "raw")

# Accuracy tables
tra_tab <- table(train_data$Attrition, train_pred_class)
test_tab <- table(test_data$Attrition, test_pred_class)
tra_accuracy <- sum(diag(tra_tab)) / sum(tra_tab)
test_accuracy <- sum(diag(test_tab)) / sum(test_tab)
tra_accuracy
test_accuracy
tra_accuracy-test_accuracy

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

# Predict probabilities
train_pred_prob <- predict(xgb_model, train_data, type = "prob")
test_pred_prob <- predict(xgb_model, test_data, type = "prob")

# Calculate AUC
train_roc <- roc(train_data$Attrition, train_pred_prob$Yes)
test_roc <- roc(test_data$Attrition, test_pred_prob$Yes)

cat("Training AUC:", auc(train_roc), "\n")
cat(" Test AUC:", auc(test_roc), "\n")

#----------------- Variable Importance Plot ---------------------
# Get variable importance from the trained model
var_imp <- varImp(xgb_model)

# Print importance scores
print(var_imp)

# Plot top variables
plot(var_imp, top = 15, main = "Top 15 Important Features - XGBoost")


# ROC Curve comparison
performance_df <- data.frame(
  Dataset = rep(c("Train", "Test"), each = 100),
  Sensitivity = c(train_roc$sensitivities, test_roc$sensitivities),
  Specificity = c(train_roc$specificities, test_roc$specificities)
)

ggplot(performance_df, aes(x = 1 - Specificity, y = Sensitivity, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  ggtitle("ROC Curve Comparison - XGBoost") +
  theme_minimal()

# Final evaluation
print(xgb_model)
plot(xgb_model)


#---------------partial dependency plots
library(pdp)

# PDP: Years Since Last Promotion
pdp_promo <- partial(xgb_model, pred.var = "YearsSinceLastPromotion", prob = TRUE, which.class = "Yes")
plot(pdp_promo,
     main = "PDP: Years Since Last Promotion",
     xlab = "Years Since Last Promotion",
     ylab = "Predicted Probability of Attrition",
     type = "l",
     col = "steelblue",
     lwd = 2)
grid()

# PDP: Salary
pdp_salary <- partial(xgb_model, pred.var = "Salary", prob = TRUE, which.class = "Yes")
plot(pdp_salary,
     main = "PDP: Salary",
     xlab = "Salary",
     ylab = "Predicted Probability of Attrition",
     type = "l",
     col = "darkred",
     lwd = 2)
grid()

#--------ICE
# Load the pdp package if not already
# install.packages("pdp")  # Uncomment if not installed
library(pdp)


# PDP: OverTime
pdp_overtime <- partial(xgb_model, pred.var = "OverTime", prob = TRUE, which.class = "Yes")
pdp_overtime_df <- as.data.frame(pdp_overtime)

ggplot(pdp_overtime_df, aes(x = OverTime, y = yhat)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "PDP: OverTime",
       x = "OverTime",
       y = "Predicted Probability of Attrition") +
  theme_minimal(base_size = 14)


# PDP: BusinessTravel
pdp_travel <- partial(xgb_model, pred.var = "BusinessTravel", prob = TRUE, which.class = "Yes")
pdp_travel_df <- as.data.frame(pdp_travel)

ggplot(pdp_travel_df, aes(x = BusinessTravel, y = yhat)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "PDP: Business Travel",
       x = "Business Travel Frequency",
       y = "Predicted Probability of Attrition") +
  theme_minimal(base_size = 14)

# Generate Partial Dependence for 'StockOptionLevel' (a factor variable)
# PDP for StockOptionLevel (numeric)
pdp_stock <- partial(
  xgb_model,
  pred.var = "StockOptionLevel",
  prob = TRUE,
  which.class = "Yes"
)

# Bar plot from Partial Dependence Data
barplot(height = pdp_stock$yhat, 
        names.arg = pdp_stock$StockOptionLevel,
        main = "Partial Dependence: Stock Option Level",
        xlab = "Stock Option Level (0 = None, 3 = Highest)",
        ylab = "Predicted Probability of Attrition",
        col = "skyblue",
        border = "white")

# Add gridlines manually
grid(nx = NA, ny = NULL)
