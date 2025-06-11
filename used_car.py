import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import sys

def load_data(path):
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df

def preprocess(df):
    
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    
    df = df.dropna()

    
    df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['mileage'] = df['mileage'].astype(str).str.replace(r'[^\d.]', '', regex=True)

   
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df = df.dropna(subset=['price', 'mileage'])

    
    df = df[(df['price'] > 100) & (df['mileage'] > 0)]

    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    print("✅ Preprocessing complete!")
    return df

def split_data(df):
    X = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Data split: {x_train.shape[0]} train samples, {x_test.shape[0]} test samples")
    return x_train, x_test, y_train, y_test

def train_decision_tree(x_train, y_train):
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(x_train, y_train)
    print("✅ Decision Tree trained")
    return dt

def train_random_forest(x_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train, y_train)
    print("✅ Random Forest trained")
    return rf

def evaluate_model(model, x_test, y_test, model_name="Model"):
    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = model.score(x_test, y_test)

    print(f"\n🔍 {model_name} Evaluation:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")
    return preds

def main(args):
    df = load_data(args.data)
    df = preprocess(df)
    x_train, x_test, y_train, y_test = split_data(df)

    dt_model = train_decision_tree(x_train, y_train)
    rf_model = train_random_forest(x_train, y_train)

    evaluate_model(dt_model, x_test, y_test, "Decision Tree")
    rf_preds = evaluate_model(rf_model, x_test, y_test, "Random Forest")

    if args.plot:
        import warnings
        warnings.filterwarnings("ignore")
        sns.kdeplot(y_test, color='r', label='Actual')
        sns.kdeplot(rf_preds, color='b', label='Predicted')
        plt.title('Price Distribution: Actual vs Predicted')
        plt.xlabel('Price')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Used Car Price Prediction")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV file")
    parser.add_argument('--plot', action='store_true', help="Plot price distribution")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
