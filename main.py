import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
feat = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')

train.head(3)
stores.head(3)
test.head(3)
feat.head()

feat.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis =1 , inplace = True )

merged_df = feat.merge(stores, on='Store', how='left')
merged_df.head(3)

train_merged = train.merge(merged_df, on=['Store', 'Date', 'IsHoliday'], how='left')
test_merged = test.merge(merged_df, on=['Store', 'Date', 'IsHoliday'], how='left')

data=train_merged
data['IsHoliday'] = data['IsHoliday'].map({True: 1, False: 0})
data['Type'].value_counts()
data['Type'] = data['Type'].map({'A': 1, 'B': 2,'C' :3})

data.head(3)

data['Date']=pd.to_datetime(data['Date'])
data['day']=data.Date.dt.day
data['month']=data.Date.dt.month
data['year']=data.Date.dt.year
data['lag_1'] = data['Weekly_Sales'].shift(1)
data['lag_7'] = data['Weekly_Sales'].shift(7)  # Weekly lag
data.set_index(data['Date'],inplace=True)
data.drop('Date',axis=1 , inplace =True)
data['rolling_avg_7'] = data['Weekly_Sales'].rolling(window=7).mean()

data.duplicated().sum()
data.isna().sum()

data.shape
data.dropna(axis=0,inplace=True)
data.isna().sum()

result = seasonal_decompose(data['Weekly_Sales'], model='additive', period=30)
result.plot()
plt.xticks(rotation=45)
plt.show()

X = data.drop('Weekly_Sales',axis=1)
y = data['Weekly_Sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(),
    'LBGM': lgb.LGBMRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f'{name} - MSE: {mse:.2f} - R2: {r2}')

model = lgb.LGBMRegressor()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [-1, 3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=3,
                           verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best score: {-grid_search.best_score_}')

models = {
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(learning_rate= 0.2, max_depth= 7, n_estimators= 200),
    'LBGM': lgb.LGBMRegressor(learning_rate= 0.2, max_depth= -1, n_estimators= 200)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f'{name} - MSE: {mse:.2f} - R2: {r2}')

scaler_X = MinMaxScaler()
X_train_scaled= scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.to_numpy().reshape(-1, 1))

models = {
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(),
    'LBGM': lgb.LGBMRegressor()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred)
    r2 = r2_score(y_test_scaled,y_pred)
    print(f'{name} - MSE: {mse:.2f} - R2: {r2}')

models = {
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(learning_rate= 0.2, max_depth= 7, n_estimators= 200),
    'LBGM': lgb.LGBMRegressor(learning_rate= 0.2, max_depth= -1, n_estimators= 200)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred)
    r2 = r2_score(y_test_scaled,y_pred)
    print(f'{name} - MSE: {mse:.2f} - R2: {r2}')

y_pred_scaled = model.predict(X_test_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_actual = scaler_y.inverse_transform(y_test_scaled).ravel()

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:300], label='True Values')
plt.plot(y_pred[:300], label='Predicted Values', alpha=0.7)
plt.title('True vs Predicted Sales (first 300 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()

performance = pd.DataFrame({
    'Models' : ['Linear Regression', 'XGBoost','LGBM','Linear Regression with Scaled data',
                'XBGoost with Scaled Data','LGBM with Scaled Data','XGBoost Hyperparameter Tuned',
                'LBGM Hyperparameter Tuned','XGBoost Hyperparameter Tuned and Data Scaled',
                'LBGM Hyperparameter Tuned and Data Scaled'
                ],
    'Mean Squared Error': [37871290.84, 11324349.62,14459745.28,0,0,0,10539729.03,10717770.84,0,0],
    'R2 Score': [0.92616,0.97792,0.97180,0.92616,0.97767,0.97268,0.97945,0.97910,0.97914,0.97916]
})
performance

best_model = performance.loc[performance['R2 Score'].idxmax()]

print(f'Best Model: {best_model["Models"]}')
print(f'Best R2: {best_model["R2 Score"]}')
print(f'MSE: {best_model["Mean Squared Error"]}')





