# 🧠 Employee Attrition Analysis and Prediction

## 📚 Overview

This project explores, predicts, and simulates employee attrition at Atlas Labs, based on workforce data.  
Starting from **exploratory data analysis (EDA)**, progressing through **machine learning models**, we finally built a **functional admin panel simulation** to predict individual employee attrition risk and suggest interventions.

**The main goal:**
👉 Understand why employees leave  
👉 Predict attrition early  
👉 Simulate real-world HR decision support using data science

---

## 🚀 Project Phases

### 1. Exploratory Data Analysis (EDA)
- Identified key attrition patterns: young, single, recently promoted employees leaving early
- Salary gaps, overtime, and business travel highlighted as major risk factors
- Techniques: Statistical tests (Chi-Square, Mann-Whitney U), correlation analysis, feature engineering

### 2. Predictive Modeling
- Built multiple machine learning models:
  - XGBoost (best performing)
  - Random Forest
  - Support Vector Machines (SVM)
  - Decision Trees
- **87% Test Accuracy** achieved with XGBoost
- **Feature Importance**:
  - Overtime
  - Business Travel
  - Years Since Last Promotion
  - Salary
  - Stock Options
  - Travelling
  - Marital Status

### 3. Attrition Risk Simulation Product
- Developed a **simulated Admin Dashboard**
- **Key Features**:
  - Add new employee data and instantly predict attrition risk
  - Filter attrition risk across departments, states, and job roles
  - Personalized suggestions for retaining high-risk employees
  - Simulate the impact of changes (salary increases, travel reductions, overtime adjustments)

---

## 🛠️ Tech Stack

- **Languages:** Python, R, PHP, HTML/CSS/JavaScript
- **Libraries:** Pandas, Seaborn, Matplotlib, XGBoost, scikit-learn, SMOTE
- **Statistical Tools:** Chi-Square Tests, Mann-Whitney U Test, Correlation Analysis
- **Web:** Flask (Python Backend), Basic PHP Integration for Admin Simulation
- **Visualization:** Seaborn, Matplotlib

---

## 📂 Repository Structure


---

## 📈 Key Outcomes

- Discovered hidden patterns influencing early attrition
- Built a predictive model to forecast employee risk at onboarding
- Created a simple dashboard that could inspire real HR decision support tools
- Highlighted the value of data-driven understanding rather than assumptions

---
