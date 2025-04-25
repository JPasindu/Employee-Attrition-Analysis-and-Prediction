#-------------------------#
# 1. Load Libraries & Data
#-------------------------#
library(skimr)
library(caret)

# Read CSVs
df_perf <- read.csv("PerformanceRating.csv", stringsAsFactors = FALSE)
df_emp <- read.csv("Employee.csv", stringsAsFactors = FALSE)

# Explore data
skim(df_perf) 
head(df_emp)
nrow(df_emp)

#---------------------------------#
# 2. Data Cleaning & Corrections
#---------------------------------#

# Fix inconsistent spacing in 'EducationField'
df_emp$EducationField[df_emp$EducationField == "Marketing "] <- "Marketing"

#----------------------------------#
# 3. Feature Engineering
#----------------------------------#

## 3.1 JobRole Grouping
df_emp$JobRole_Grouped <- df_emp$JobRole
df_emp$JobRole_Grouped[df_emp$JobRole %in% c("Software Engineer", "Senior Software Engineer")] <- "Software Engineer"
df_emp$JobRole_Grouped[df_emp$JobRole %in% c("Machine Learning Engineer", "Data Scientist")] <- "AI/ML"
df_emp$JobRole_Grouped[df_emp$JobRole %in% c("Engineering Manager", "Analytics Manager", "Manager")] <- "Management"
df_emp$JobRole_Grouped[df_emp$JobRole %in% c("HR Manager", "HR Executive", "HR Business Partner", "Recruiter")] <- "HR"
df_emp$JobRole_Grouped[df_emp$JobRole %in% c("Sales Executive", "Sales Representative")] <- "Sales"
table(df_emp$JobRole, df_emp$JobRole_Grouped)

## 3.2 EducationField Grouping
df_emp$EducationField_Grouped <- df_emp$EducationField
df_emp$EducationField_Grouped[df_emp$EducationField %in% c("Computer Science", "Information Systems", "Technical Degree")] <- "STEM"
df_emp$EducationField_Grouped[df_emp$EducationField %in% c("Economics", "Business Studies", "Marketing")] <- "Business"
df_emp$EducationField_Grouped[df_emp$EducationField == "Human Resources"] <- "HR"
df_emp$EducationField_Grouped[df_emp$EducationField == "Other"] <- "Other"
table(df_emp$EducationField, df_emp$EducationField_Grouped)

## 3.3 Ethnicity Grouping
df_emp$Ethnicity_Grouped <- df_emp$Ethnicity
df_emp$Ethnicity_Grouped[df_emp$Ethnicity == "Asian or Asian American"] <- "Asian"
df_emp$Ethnicity_Grouped[df_emp$Ethnicity == "Black or African American"] <- "Black"
df_emp$Ethnicity_Grouped[df_emp$Ethnicity == "White"] <- "White"
df_emp$Ethnicity_Grouped[df_emp$Ethnicity %in% c("American Indian or Alaska Native", "Native Hawaiian ", "Mixed or multiple ethnic groups", "Other ")] <- "Mixed/Other"
table(df_emp$Ethnicity, df_emp$Ethnicity_Grouped)

## 3.4 Gender Grouping
df_emp$Gender_Grouped <- df_emp$Gender
df_emp$Gender_Grouped[df_emp$Gender %in% c("Non-Binary", "Prefer Not To Say")] <- "Other"
table(df_emp$Gender, df_emp$Gender_Grouped)

#----------------------------------------#
# 5. Prepare Performance Ratings
#----------------------------------------#

# Convert and sort ReviewDate
df_perf$ReviewDate <- as.Date(df_perf$ReviewDate, format="%m/%d/%Y")
df_perf <- df_perf[order(df_perf$EmployeeID, -as.numeric(df_perf$ReviewDate)), ]

# Keep latest review per employee
latest_perf <- df_perf[!duplicated(df_perf$EmployeeID), ]

# Merge performance ratings into employee data
merged_df <- merge(df_emp, latest_perf, by = "EmployeeID", all = FALSE)
skim(merged_df)
print(dim(merged_df))

#-------------------------------------------#
# 6. Variable Type Conversion
#-------------------------------------------#

# Nominal to factor
nominal_vars <- c("EmployeeID", "FirstName", "LastName", "Gender", "BusinessTravel",
                  "Department", "State", "Ethnicity", "EducationField", "JobRole",
                  "MaritalStatus", "PerformanceID", "Attrition", "OverTime",
                  "EducationField_Grouped", "Ethnicity_Grouped", "Gender_Grouped",
                  "JobRole_Grouped")
merged_df[nominal_vars] <- lapply(merged_df[nominal_vars], factor)

# Ordinal to ordered factor
ordinal_vars <- c("Education", "StockOptionLevel", "EnvironmentSatisfaction",
                  "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                  "SelfRating", "ManagerRating")
merged_df[ordinal_vars] <- lapply(merged_df[ordinal_vars], function(x) ordered(x, levels = sort(unique(x))))

# Dates and numeric conversion
merged_df$HireDate <- as.Date(merged_df$HireDate)
ratio_vars <- c("Age", "DistanceFromHome..KM.", "Salary", "YearsAtCompany",
                "YearsInMostRecentRole", "YearsSinceLastPromotion",
                "YearsWithCurrManager", "TrainingOpportunitiesWithinYear",
                "TrainingOpportunitiesTaken")
merged_df[ratio_vars] <- lapply(merged_df[ratio_vars], as.numeric)

str(merged_df)

#-------------------------------------------#
# 7. Prepare Data for EDA
#-------------------------------------------#
nominal_vars_keep <- c("BusinessTravel", "Department", "State", "MaritalStatus", "Attrition",
                       "OverTime", "EducationField_Grouped", "Ethnicity_Grouped",
                       "Gender_Grouped", "JobRole_Grouped")

ordinal_vars_keep <- c("Education", "StockOptionLevel", "EnvironmentSatisfaction",
                       "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                       "SelfRating", "ManagerRating")

ratio_vars_keep <- c("Age", "DistanceFromHome..KM.", "Salary", "YearsAtCompany",
                     "YearsInMostRecentRole", "YearsSinceLastPromotion",
                     "YearsWithCurrManager", "TrainingOpportunitiesWithinYear",
                     "TrainingOpportunitiesTaken")

vars_keep <- c(nominal_vars_keep, ordinal_vars_keep, ratio_vars_keep)
merged_df=merged_df[,vars_keep]
# Train-test split
set.seed(123)
train_index <- createDataPartition(df$Attrition, p = 0.8, list = FALSE)

train_data_EDA=merged_df[train_index,]

saveRDS(train_data_EDA, "train_data_EDA.rds")

#-------------------------------------------#
# 8. Final Feature Selection for Modeling
#-------------------------------------------#

nominal_vars_keep <- c("BusinessTravel", "Department", "State", "MaritalStatus", "Attrition",
                       "OverTime", "EducationField_Grouped", "Ethnicity_Grouped",
                       "Gender_Grouped", "JobRole_Grouped")

ordinal_vars_keep <- c("Education", "StockOptionLevel", "EnvironmentSatisfaction",
                       "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                       "ManagerRating")

ratio_vars_keep <- c("Age", "DistanceFromHome..KM.", "Salary",
                     "YearsSinceLastPromotion", "TrainingOpportunitiesTaken")

vars_keep <- c(nominal_vars_keep, ordinal_vars_keep, ratio_vars_keep)

df <- merged_df[, vars_keep]
str(df)
head(df)

#-------------------------------------------#
# 9. Prepare Data for Modeling
#-------------------------------------------#

train_data <- df[train_index, ]
test_data <- df[-train_index, ]
#saveRDS(train_data, "train_data.rds")
#saveRDS(test_data, "test_data.rds")
