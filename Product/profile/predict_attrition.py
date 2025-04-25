import sys
import json
import pandas as pd
import joblib

def clean_categorical_levels(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.title()
    )

def preprocess_features(df):
    try:
        features = ["BusinessTravel", "MaritalStatus", "StockOptionLevel", "OverTime", "Age", "Salary", "YearsSinceLastPromotion"]

        df["BusinessTravel"] = clean_categorical_levels(df["BusinessTravel"])
        df["MaritalStatus"] = clean_categorical_levels(df["MaritalStatus"])
        df["OverTime"] = clean_categorical_levels(df["OverTime"])

        business_travel_mapping = {
            'No Travel': 0,
            'Some Travel': 1,
            'Frequent Traveller': 2
        }

        marital_status_mapping = {
            'Divorced': 0,
            'Single': 1,
            'Married': 2
        }

        overtime_mapping = {
            'No': 0,
            'Yes': 1
        }

        df['BusinessTravel'] = df['BusinessTravel'].map(business_travel_mapping)
        df['MaritalStatus'] = df['MaritalStatus'].map(marital_status_mapping)
        df['OverTime'] = df['OverTime'].map(overtime_mapping)

        df = df.astype(int)
        return df[features]
    
    except Exception as e:
        raise ValueError(f"Feature preprocessing failed: {str(e)}")

def main():
    try:
        raw_input = sys.stdin.read()
        data_dict = json.loads(raw_input)
        new_data = pd.DataFrame([data_dict])

        processed_data = preprocess_features(new_data)

        try:
            pipeline = joblib.load("../model/xgb_smote_pipeline.pkl")
        except FileNotFoundError:
            print("Error: Model file 'xgb_smote_pipeline.pkl' not found")
            sys.exit(1)
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")

        try:
            predictions = pipeline.predict(processed_data)
            print(predictions[0])
            #print(json.dumps(predictions[0]))
        except Exception as e:
            error_type = type(e).__name__
            if "unknown categories" in str(e):
                col_idx = int(str(e).split("column ")[1].split()[0])
                feature_names = pipeline.named_steps['preprocessor'].transformers_[0][2]
                problem_feature = feature_names[col_idx]
                problem_value = processed_data[problem_feature].iloc[0]
                allowed_categories = pipeline.named_steps['preprocessor'].transformers_[0][1].categories_[col_idx]
                raise ValueError(
                    f"Unknown category '{problem_value}' found in feature '{problem_feature}'. "
                    f"Allowed categories: {allowed_categories}"
                )
            else:
                raise ValueError(f"Prediction failed: {error_type}: {str(e)}")

    except json.JSONDecodeError:
        print("Error: Invalid JSON input")
        sys.exit(1)
    except ValueError as ve:
        print(f"Error: {str(ve)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
