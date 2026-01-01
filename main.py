import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'Walmart_Sales.csv'
data = pd.read_csv(file)
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')
    
all_sales = []
for i in pd.unique(df['Date']):
    sales_on_date = df[df['Date'] == pd.to_datetime(i)]['Weekly_Sales'].values
    if len(sales_on_date) > 0:
        total_sales = np.sum(sales_on_date)
    else:
        total_sales = 0
    
    cpi_on_date = df[df['Date'] == pd.to_datetime(i)]['CPI'].values
    if len(cpi_on_date) > 0:
        total_cpi = np.sum(cpi_on_date)
    else:
        total_cpi = 0
    all_sales.append((i, total_sales, total_cpi))
all_sales = pd.DataFrame(all_sales, columns=['Date', 'Total_Sales', 'Total_CPI'])

x = all_sales['Date']
y1 = all_sales['Total_Sales']
y2 = all_sales['Total_CPI']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Sales', color='g')
ax2.set_ylabel('Total CPI', color='b')
plt.title('Total Sales and CPI Over Time')
plt.show()