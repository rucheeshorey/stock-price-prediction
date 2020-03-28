import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'AAPL_data.csv'
dataset = pd.read_csv(path)
training_set = dataset.iloc[:1007,4:5].values


#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60,len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping
#x_train is now the first 60 items in training_set_scaled
#y_train is now the rest of the items in training_set_scaled
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


#Initialising the RNN
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences = True, input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output later
regressor.add(Dense(units = 1))


#Compiling the RNN
#Adam - stochastic process optimizer
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training set
#batch size - number of samples per gradient update
#epochs - number of iterations over the entire x and y provided
regressor.fit(x_train, y_train, epochs=100, batch_size = 32)

# Making the predictions and visualizing the results
test_set = dataset.iloc[1007:,4:5].values
real_stock_price = test_set

# Getting the predicted stock price
closing_total = dataset.iloc[:,4:5]
inputs = closing_total[len(closing_total)-len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60,310):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Apple Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
