# Ex.No: 08     MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING

## AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
## ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
## PROGRAM:
### Name : Dharshni V M
### Register Number : 212223240029
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
data = pd.read_csv('gold.csv')

rate_data = data[['USD (AM)']]

print("Shape of the dataset:", rate_data.shape)
print("First 10 rows of the dataset:")
print(rate_data.head(10))

plt.figure(figsize=(12, 6))
plt.plot(rate_data['USD (AM)'], label='Original USD data')
plt.title('Original USD Data')
plt.xlabel('Rate')
plt.ylabel('USD')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = rate_data['USD (AM)'].rolling(window=5).mean()
rolling_mean_10 = rate_data['USD (AM)'].rolling(window=10).mean()
rolling_mean_5.head(10)
rolling_mean_10.head(20)

plt.figure(figsize=(12, 6))
plt.plot(rate_data['USD (AM)'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of USD')
plt.xlabel('Rate')
plt.ylabel('USD')
plt.legend()
plt.grid()
plt.show()

data.head()

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_monthly = data.resample('MS').mean()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_usd = pd.Series(
    scaler.fit_transform(data_monthly[['USD (AM)']]).flatten(),
    index=data_monthly.index,
    name='USD_scaled'
)

scaled_usd=scaled_usd+1  
x=int(len(scaled_usd)*0.8)
train_data = scaled_usd[:x]
test_data = scaled_usd[x:]
from sklearn.metrics import mean_squared_error

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_usd.var()),scaled_usd.mean()

usd_series = data_monthly['USD (AM)']

model = ExponentialSmoothing(usd_series, trend='add', seasonal='mul', seasonal_periods=12).fit()

predictions = model.forecast(steps=int(len(usd_series)/4))

ax = usd_series.plot(figsize=(10, 5))
predictions.plot(ax=ax)
ax.legend(["USD monthly", "USD forecast"])
ax.set_xlabel('Date')
ax.set_ylabel('USD Rate')
ax.set_title('PREDICTION')
plt.show()
```

### OUTPUT:
### Original Data
![original](https://github.com/user-attachments/assets/be39e106-d6a3-4264-949c-d4baa30353bc)
![original usd](https://github.com/user-attachments/assets/a6472027-5617-4491-b4f2-89f788185677)
### Moving Average
![rolling](https://github.com/user-attachments/assets/2ebb775a-dbe4-499c-88ac-2df53bc7e4c8)
![Moving](https://github.com/user-attachments/assets/c25a9226-bf88-4f8f-b7d7-fbae8dd3edd9)
### Exponential Smoothing (Test)
![visual](https://github.com/user-attachments/assets/4442eec4-82d4-4975-8ab5-ab96541f4403)
### Performance (MSE)
![mse](https://github.com/user-attachments/assets/417cc5e7-d0d6-46ca-ac53-4b5c2c5a9afb)
### Prediction
![prediction](https://github.com/user-attachments/assets/ba7cc192-2174-4da9-a9ef-5123feb85c31)

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
