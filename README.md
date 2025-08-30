# Sales-Forecasting

# 📈 Sales Forecasting (Walmart Dataset)

## 📌 Objective
The goal of this project is to **predict future sales based on historical Walmart sales data**. By creating time-based features and applying regression models, the project demonstrates how machine learning can be used to generate sales forecasts and guide business decisions.

## 📂 Dataset
- **Walmart Sales Forecast Dataset (Kaggle)**  
  Contains historical sales data including:
  - Store ID  
  - Department ID  
  - Date  
  - Weekly Sales (Target variable)  
  - Additional attributes (holiday, temperature, fuel price, etc.)

## ⚙️ Technologies & Tools Used
- **Python 3.x**
- **Pandas** – data preprocessing, feature engineering  
- **NumPy** – numerical computations  
- **Matplotlib / Seaborn** – time-series visualization  
- **Scikit-learn** – regression models & evaluation metrics  
  - Linear Regression  
  - Decision Tree / Random Forest Regressors (optional)  
- **Statsmodels** (optional) – for time series methods like ARIMA  
- **Jupyter Notebook / PyCharm** – development environment  

## 🧠 Methods & Approach
1. **Data Preprocessing**
   - Handled missing values and formatted dates  
   - Generated time-based features: **day, month, year, lag values**  

2. **Exploratory Data Analysis (EDA)**
   - Analyzed sales trends over time  
   - Visualized sales by store, department, and seasonality  

3. **Feature Engineering**
   - Created lag features for capturing past sales influence  
   - Encoded categorical variables where needed  

4. **Model Building**
   - Applied regression models to forecast next period’s sales  
   - Compared performance of models (Linear Regression, Decision Tree, Random Forest, etc.)  

5. **Model Evaluation**
   - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score  
   - Plotted **Actual vs Predicted sales** over time  

## 🎯 Learning Outcomes
- Understanding time-series forecasting and regression approaches  
- Performing feature engineering with temporal data  
- Evaluating regression models on forecasting tasks  
- Visualizing sales trends and prediction performance  

