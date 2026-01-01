# Walmart Sales Analysis (EDA)

Exploratory data analysis and visualizations on the `Walmart_Sales.csv` dataset to understand weekly sales trends, holiday effects, and relationships between sales and external factors (temperature, fuel price, CPI, unemployment). [web:66]

## Project Overview

This project performs exploratory data analysis on Walmart sales data and examines:
- Weekly sales trends and seasonality patterns
- Impact of holidays on sales performance  
- Relationships between sales and economic factors (CPI, fuel prices, unemployment, temperature)

**Future extension**: Building forecasting models to predict future Weekly_Sales using economic indicators as exogenous variables.

## Dataset

File: `Walmart_Sales.csv`

Typical columns in this dataset:
- `Store`: Store number. [web:66]
- `Date`: Week of sales (date). [web:66]
- `Weekly_Sales`: Sales for the given store in that week (target). [web:66]
- `Holiday_Flag`: Whether the week is a holiday week. [web:66]
- `Temperature`: Temperature on the day/week. [web:66]
- `Fuel_Price`: Fuel cost in the region. [web:66]
- `CPI`: Consumer Price Index. [web:66]
- `Unemployment`: Unemployment rate. [web:66]

## Repository Structure

- `main.py` — runs the analysis and generates plots.
- `Walmart_Sales.csv` — dataset (not included if this is a public repo; consider adding it to `.gitignore`).

## Setup

### Requirements
- Python 3.8+
- Dependencies are listed in `requirements.txt`

