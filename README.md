import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

data = pd.read_csv('gold_prices.csv', parse_dates=['Date'])


data = data[['Date', 'Price']]


data['Price'] = data['Price'].str.replace(',', '').astype(float)
data.set_index('Date', inplace=True)


data.index = pd.to_datetime(data.index, format='%d-%m-%Y') 

data['Day'] = data.index.day  
data['Month'] = data.index.month
data['Year'] = data.index.year
data['Lag1'] = data['Price'].shift(1)
data['Lag2'] = data['Price'].shift(2)
data.dropna(inplace=True)


X = data[['Day', 'Month', 'Year', 'Lag1', 'Lag2']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

future_dates = pd.date_range(start='2025-02-26', end='2025-03-26', freq='D')
future_data = pd.DataFrame(future_dates, columns=['Date'])
future_data['Day'] = future_data['Date'].dt.day
future_data['Month'] = future_data['Date'].dt.month
future_data['Year'] = future_data['Date'].dt.year

last_known_price = data['Price'].iloc[-1]
future_data['Lag1'] = last_known_price
future_data['Lag2'] = last_known_price

X_future = future_data[['Day', 'Month', 'Year', 'Lag1', 'Lag2']]

future_predictions = model.predict(X_future)

future_data['Predicted_Price'] = future_predictions * 7

print(future_data[['Date', 'Predicted_Price']])

plt.figure(figsize=(10, 5))
plt.plot(future_data['Date'], future_data['Predicted_Price'], label='Predicted Price', color='orange')
plt.title('Gold Price Prediction (feb 2025 - mar 2025)')
plt.xlabel('Date')
plt.ylabel('Price in INR')
plt.legend()
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

data = pd.read_csv('gold_prices.csv', parse_dates=['Date'])
data = data[['Date', 'Price']]
data['Price'] = data['Price'].str.replace(',', '').astype(float)
data.set_index('Date', inplace=True)
data.index = pd.to_datetime(data.index, format='%d-%m-%Y') 

data['Day'] = data.index.day  
data['Month'] = data.index.month
data['Year'] = data.index.year
data['Lag1'] = data['Price'].shift(1)
data['Lag2'] = data['Price'].shift(2)
data.dropna(inplace=True)

X = data[['Day', 'Month', 'Year', 'Lag1', 'Lag2']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

future_dates = pd.date_range(start='2025-02-26', end='2025-03-26', freq='D')
future_data = pd.DataFrame(future_dates, columns=['Date'])
future_data['Day'] = future_data['Date'].dt.day
future_data['Month'] = future_data['Date'].dt.month
future_data['Year'] = future_data['Date'].dt.year

last_known_price = data['Price'].iloc[-1]
future_data['Lag1'] = last_known_price
future_data['Lag2'] = last_known_price

X_future = future_data[['Day', 'Month', 'Year', 'Lag1', 'Lag2']]

future_predictions = model.predict(X_future)

# Scale the predicted prices by 5.6 to get INR equivalent (assuming original data is USD)
future_data['Predicted_Price_INR'] = future_predictions   # This line was changed

print(future_data[['Date', 'Predicted_Price_INR ']])

plt.figure(figsize=(10, 5))
plt.plot(future_data['Date'], future_data['Predicted_Price_INR'], label='Predicted Price (INR)', color='orange')
plt.title('Gold Price Prediction (feb 2025 - mar 2025)')
plt.xlabel('Date')
plt.ylabel('Price in INR')
plt.legend
