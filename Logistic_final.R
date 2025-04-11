#----------------- Logistic Regression Pipeline -----------------------------
library(ROSE)
library(pROC)
library(ggplot2)
library(dplyr)
library(caret)

# Copy original
df_for_logit <- read.csv("df.csv")

# Convert nominal vars to factors
nominal_vars <- c("BusinessTravel", "Department", "State",
                  "MaritalStatus", "Attrition", "OverTime",
                  "EducationField_Grouped", "Ethnicity_Grouped",
                  "Gender_Grouped", "JobRole_Grouped")
df_for_logit[nominal_vars] <- lapply(df_for_logit[nominal_vars], as.factor)

# Convert ordinal vars to numeric
ordinal_vars <- c("Education", "StockOptionLevel", "EnvironmentSatisfaction",
                  "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                  "ManagerRating")
df_for_logit[ordinal_vars] <- lapply(df_for_logit[ordinal_vars], function(x) as.numeric(as.character(x)))

# Ratio variables
ratio_vars <- c("Age", "DistanceFromHome..KM.", "Salary", "YearsSinceLastPromotion",
                "TrainingOpportunitiesWithinYear", "TrainingOpportunitiesTaken")
df_for_logit[ratio_vars] <- lapply(df_for_logit[ratio_vars], as.numeric)

# Separate target
target <- df_for_logit$Attrition
df_for_logit$Attrition <- NULL

# Dummy encoding
dummies <- dummyVars(" ~ .", data = df_for_logit)

# Apply to full dataset to ensure all levels are present
df_encoded <- as.data.frame(predict(dummies, newdata = df_for_logit))
df_encoded$Attrition <- target

# Split
set.seed(123)
train_index <- createDataPartition(df_encoded$Attrition, p = 0.8, list = FALSE)
train_data <- df_encoded[train_index, ]
test_data  <- df_encoded[-train_index, ]

# Ensure column names match
train_data <- train_data %>% mutate(across(everything(), identity))
test_data  <- test_data %>% mutate(across(everything(), identity))
colnames(train_data) <- make.names(colnames(train_data))
colnames(test_data)  <- make.names(colnames(test_data))

# Apply SMOTE to train_data only
train_data_lg<- ROSE::ovun.sample(Attrition ~ ., data = train_data, method = "both", N = nrow(train_data))$data

# Fit logistic model
logit_model <- glm(Attrition ~ ., data = train_data_lg, family = binomial)
# saved_logit_model_final=logit_model
# Predict probabilities
train_pred_prob <- predict(logit_model, newdata = train_data, type = "response")
test_pred_prob  <- predict(logit_model, newdata = test_data, type = "response")

# Predict classes
train_pred_class <- factor(ifelse(train_pred_prob > 0.5, "Yes", "No"), levels = c("No", "Yes"))
test_pred_class  <- factor(ifelse(test_pred_prob > 0.5, "Yes", "No"), levels = c("No", "Yes"))


# Accuracy
train_tab <- table(train_data$Attrition, train_pred_class)
test_tab  <- table(test_data$Attrition, test_pred_class)

train_acc <- sum(diag(train_tab)) / sum(train_tab)
test_acc  <- sum(diag(test_tab)) / sum(test_tab)

cat("Train Accuracy:", train_acc, "\n")
cat("Test Accuracy :", test_acc, "\n")

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

# ROC / AUC
train_roc <- roc(train_data$Attrition, train_pred_prob)
test_roc  <- roc(test_data$Attrition, test_pred_prob)

cat("Train AUC:", auc(train_roc), "\n")
cat("Test AUC :", auc(test_roc), "\n")
