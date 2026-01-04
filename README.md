# Walmart Sales Forecasting

A beginner-level data science project that analyzes Walmart sales data and builds a Random Forest machine learning model to forecast weekly sales using temporal patterns and economic indicators.

## Project Overview

This project performs:
- **Exploratory Data Analysis (EDA)** on historical Walmart sales data
- **Feature engineering** to extract temporal patterns (year, month, week, quarter)
- **Sales forecasting** using Random Forest Regressor
- **Model evaluation** with multiple metrics (RMSE, MAE, R²)
- **Visualization** of predictions vs actual sales with economic context (CPI)

### Why Sales Forecasting?

Sales forecasting was chosen over stock prediction because it offers clearer patterns, more reliable data, and better learning outcomes for beginners. Unlike stock markets with unpredictable external influences, retail sales exhibit identifiable seasonal patterns and trends that reward proper modeling techniques.

## Dataset

**File**: `Walmart_Sales.csv`

**Structure**:
- **6,435 records** across 45 stores with weekly granularity
- **Date range**: Multiple years of historical data
- **Target variable**: `Weekly_Sales` (sales amount per store per week)
- **Features**:
  - `Store`: Store number (1-45)
  - `Date`: Week of sales
  - `Holiday_Flag`: Binary indicator for holiday weeks
  - `Temperature`: Average temperature during the week
  - `Fuel_Price`: Regional fuel cost
  - `CPI`: Consumer Price Index (inflation indicator)
  - `Unemployment`: Regional unemployment rate


## Setup

### Requirements
- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn, joblib

### Installation

**Using virtual environment (recommended)**:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
