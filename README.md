# 🚗 Used Car Price Prediction

A machine learning project to predict used car prices based on attributes like year, mileage, fuel type, transmission, and engine size. This project uses a clean, tabular dataset and supports both Decision Tree and Random Forest regressors.

---

## 📁 Dataset

The dataset should be a CSV file containing columns such as:

-   `price` (target)
-   `year`
-   `mileage`
-   `fuel_type`
-   `transmission`
-   `engine_size`

Example Data Sources are provided in the data folder

---

## ⚙️ Features

-   Handles preprocessing (missing values, numeric conversion, categorical encoding)
-   Trains two models: Decision Tree and Random Forest
-   Evaluates models using RMSE, MAE, and R² Score
-   (Optional) Plots actual vs predicted price distributions

---

## 🧪 Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/used-car-price-predictor.git
cd used-car-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python predictor.py --data path/to/bmw.csv --plot
```

    --data: Path to the CSV file

    --plot: (Optional) Show KDE plot of actual vs predicted prices

### 🧠 Models Used

    DecisionTreeRegressor: Simple, interpretable baseline

    RandomForestRegressor: More robust, typically better performance

### 📊 Example Output

Decision Tree Regressor trained.
Random Forest Regressor trained.

```Decision Tree Evaluation:
RMSE: 2940.67
MAE: 2190.54
R^2 Score: 0.78

Random Forest Evaluation:
RMSE: 2153.39
MAE: 1710.45
R^2 Score: 0.87
```
