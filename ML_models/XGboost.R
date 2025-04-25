#----------------------------- XGBoost Model with SMOTE and Interpretation -----------------------------

# Load required libraries
library(caret)
library(pROC)
library(ROSE)     
library(ggplot2)
library(pdp)

#----------------------------- Load Data -----------------------------
train_data <- readRDS("train_data.rds")
test_data <- readRDS("test_data.rds")

#----------------------------- Train Control Setup -----------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

#----------------------------- Final Tuning Grid -----------------------------
tune_grid <- expand.grid(
  nrounds = 250,
  max_depth = 6,
  eta = 0.0005,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 10,
  subsample = 0.7
)

#----------------------------- Train XGBoost Model -----------------------------
set.seed(100)
xgb_model <- suppressWarnings(train(
  Attrition ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  tuneGrid = tune_grid,
  trControl = ctrl
))

#----------------------------- Class Predictions -----------------------------
train_pred_class <- predict(xgb_model, train_data)
test_pred_class <- predict(xgb_model, test_data)

#----------------------------- Accuracy -----------------------------
train_acc <- mean(train_pred_class == train_data$Attrition)
test_acc <- mean(test_pred_class == test_data$Attrition)
cat("Train Accuracy:", round(train_acc, 3), "\n")
cat("Test Accuracy:", round(test_acc, 3), "\n")
cat("Accuracy Gap:", round(train_acc - test_acc, 3), "\n\n")

#----------------------------- Confusion Matrix & Metrics -----------------------------
train_conf <- confusionMatrix(train_pred_class, train_data$Attrition, positive = "Yes")
test_conf <- confusionMatrix(test_pred_class, test_data$Attrition, positive = "Yes")

cat("Train Precision:", round(train_conf$byClass["Precision"], 3), "\n")
cat("Train Recall:", round(train_conf$byClass["Recall"], 3), "\n")
cat("Train F1 Score:", round(train_conf$byClass["F1"], 3), "\n\n")

cat("Test Precision:", round(test_conf$byClass["Precision"], 3), "\n")
cat("Test Recall:", round(test_conf$byClass["Recall"], 3), "\n")
cat("Test F1 Score:", round(test_conf$byClass["F1"], 3), "\n\n")

#----------------------------- AUC Evaluation -----------------------------
train_prob <- predict(xgb_model, train_data, type = "prob")[, "Yes"]
test_prob <- predict(xgb_model, test_data, type = "prob")[, "Yes"]

train_roc <- roc(train_data$Attrition, train_prob)
test_roc <- roc(test_data$Attrition, test_prob)

cat("Training AUC:", round(auc(train_roc), 3), "\n")
cat("Test AUC:", round(auc(test_roc), 3), "\n")

#----------------------------- Variable Importance -----------------------------
var_imp <- varImp(xgb_model)
plot(var_imp, top = 15, main = "Top 15 Important Features - XGBoost")

#----------------------------- Partial Dependence Plots (PDPs) -----------------------------
# Numeric PDPs
plot(partial(xgb_model, pred.var = "YearsSinceLastPromotion", prob = TRUE, which.class = "Yes"),
     main = "PDP: Years Since Last Promotion", xlab = "Years Since Last Promotion",
     ylab = "Probability of Attrition", type = "l", col = "steelblue", lwd = 2)
grid()

plot(partial(xgb_model, pred.var = "Salary", prob = TRUE, which.class = "Yes"),
     main = "PDP: Salary", xlab = "Salary", ylab = "Probability of Attrition",
     type = "l", col = "darkred", lwd = 2)
grid()

# Categorical PDPs (Bar Charts)
# OverTime
pdp_overtime <- as.data.frame(partial(xgb_model, pred.var = "OverTime", prob = TRUE, which.class = "Yes"))
ggplot(pdp_overtime, aes(x = OverTime, y = yhat)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "PDP: OverTime", x = "OverTime", y = "Probability of Attrition") +
  theme_minimal()

# BusinessTravel
pdp_travel <- as.data.frame(partial(xgb_model, pred.var = "BusinessTravel", prob = TRUE, which.class = "Yes"))
ggplot(pdp_travel, aes(x = BusinessTravel, y = yhat)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "PDP: Business Travel", x = "Business Travel Frequency", y = "Probability of Attrition") +
  theme_minimal()

# StockOptionLevel (numeric encoded factor)
pdp_stock <- partial(xgb_model, pred.var = "StockOptionLevel", prob = TRUE, which.class = "Yes")
barplot(height = pdp_stock$yhat,
        names.arg = pdp_stock$StockOptionLevel,
        main = "PDP: Stock Option Level",
        xlab = "Stock Option Level (0 = None, 3 = Highest)",
        ylab = "Probability of Attrition",
        col = "skyblue", border = "white")
grid(nx = NA, ny = NULL)


