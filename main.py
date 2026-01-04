import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

file = 'Walmart_Sales.csv'
df = pd.read_csv(file)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')
    
all_sales = df.groupby('Date').agg({
    'Weekly_Sales': 'sum',
    'CPI': 'mean'
}).reset_index()
all_sales.columns = ['Date', 'Total_Sales', 'Avg_CPI']

x = all_sales['Date']
y1 = all_sales['Total_Sales']
y2 = all_sales['Avg_CPI']
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Sales', color='g')
ax2.set_ylabel('Total CPI', color='b')
plt.title('Total Sales and CPI Over Time')


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter

# Define features and target
feature_cols = ['Store', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                'Holiday_Flag', 'Year', 'Month', 'Week', 'Quarter']
X = df[feature_cols]
y = df['Weekly_Sales']

# Time-based split (80-20)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Initialize and train
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

rf_model.fit(X_train, y_train)

# Make predictions
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Train RMSE: {train_rmse:,.2f}")
print(f"Test RMSE: {test_rmse:,.2f}")
print(f"Test MAE: {test_mae:,.2f}")
print(f"Test R²: {test_r2:.4f}")

# Get feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

print(f"Train period: {df['Date'].iloc[0]} to {df['Date'].iloc[split_idx-1]}")
print(f"Test period: {df['Date'].iloc[split_idx]} to {df['Date'].iloc[-1]}")

plt.figure()
importance_sorted = importance_df.sort_values('Importance', ascending=True)
plt.barh(importance_sorted['Feature'], importance_sorted['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()

# Get test dates and CPI values
test_dates = df['Date'][split_idx:].values
test_cpi = df['CPI'][split_idx:].values

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot sales on primary axis
ax1.plot(test_dates, y_test.values, 
         label='Actual Sales', color='blue', linewidth=2, marker='o', markersize=4)
ax1.plot(test_dates, test_pred, 
         label='Predicted Sales', color='red', linewidth=2, marker='x', markersize=4)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Weekly Sales', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot CPI on secondary axis
ax2 = ax1.twinx()
ax2.plot(test_dates, test_cpi, 
         label='CPI', color='green', linewidth=1.5, linestyle='--', alpha=0.7)
ax2.set_ylabel('CPI', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

plt.title('Sales Prediction with CPI Context', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import joblib
joblib.dump(rf_model, 'rf_sales_model.pkl')