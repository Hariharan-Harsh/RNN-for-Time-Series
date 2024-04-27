#!/usr/bin/env python
# coding: utf-8

# 
# 
# # RNN for Time Series
# 
# RNNs are used for sequence modeling. This tutorial will look at a time series data to be modeled and predicted using RNNs. 

# In[1]:


#
# Import Libraries
#
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data
# 
# We will use retail data for time-series modeling. 
# 
# Link to the dataset:
#  https://fred.stlouisfed.org/series/MRTSSM448USN
# 
# Information about the Advance Monthly Retail Sales Survey can be found on the Census website at:
# https://www.census.gov/retail/marts/about_the_surveys.html
# 
# Release: Advance Monthly Sales for Retail and Food Services  
# Units:  Millions of Dollars, Not Seasonally Adjusted
# Frequency:  Monthly
# 
# Suggested Citation:
# U.S. Census Bureau, Advance Retail Sales: Clothing and Clothing Accessory Stores [RSCCASN], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/RSCCASN, November 16, 2019.
# 
# https://fred.stlouisfed.org/series/RSCCASN

# ### Read data first -  Use index_col = 'DATE' and 'parse_dates = True' as a parameter.

# In[2]:


# Your code to read data
df = pd.read_csv("RSCCASN.csv", index_col='DATE', parse_dates=True)
df.head()
# Print first few rows of data


# Does the sales column has any name?
# 
# If no, set the name of the colum as 'Sales'.

# In[3]:


# Set name of column as 'Sales'. Use - df.columns 
df.columns = ['sales']


# Plot your data - Year vs Sales

# In[4]:


# Your code to plot Year vs Sales. Use either matplot library of pandas dataframe.
df.head()


# In[5]:


df.plot(figsize=(12,8))


# ### Next we will do Train Test Split. 
# 
# We will use last 1.5 year (18 month) samples for testing. Rest is for training.

# In[6]:


test_size = 18


# Now, we will find the indexes of the test data. Remember, these are the last 18 indexes in the pandas dataframe.

# In[7]:


#Assign the start of test index in data frame to variable test_index.  Remember, it is equal to the length of dataframe - test size
data_length = len(df)
data_length


# Next, we will separate train and test datasets.

# In[8]:


len(df)- 18


# In[9]:


train_size = data_length- test_size
train_size


# In[10]:


test_index =  train_size


# In[11]:


train = df.iloc[:test_index]
test = df.iloc[test_index:]


# In[12]:


print(train.size)
print(test.size)


# In[13]:


train


# In[14]:


test


# ### In Neural Networks, we need to Scale Data between 0-1

# In[15]:


from sklearn.preprocessing import MinMaxScaler


# In[16]:


scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[17]:


#
# Check if the data has been scaled properly
#
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[18]:


print(scaled_train.max())
print(scaled_test.max())
print(scaled_train.min())
print(scaled_test.min())


# # Time Series Generator
# 
# This class takes in a sequence of data-points gathered at
# equal intervals, along with time series parameters such as
# stride, length of history, etc., to produce batches for
# training/validation.
# 
# #### Arguments
#     data: Indexable generator (such as list or Numpy array)
#         containing consecutive data points (timesteps).
#         The data should be at 2D, and axis 0 is expected
#         to be the time dimension.
#     targets: Targets corresponding to timesteps in `data`.
#         It should have same length as `data`.
#     length: Length of the output sequences (in number of timesteps).
#     sampling_rate: Period between successive individual timesteps
#         within sequences. For rate `r`, timesteps
#         `data[i]`, `data[i-r]`, ... `data[i - length]`
#         are used for create a sample sequence.
#     stride: Period between successive output sequences.
#         For stride `s`, consecutive output samples would
#         be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
#     start_index: Data points earlier than `start_index` will not be used
#         in the output sequences. This is useful to reserve part of the
#         data for test or validation.
#     end_index: Data points later than `end_index` will not be used
#         in the output sequences. This is useful to reserve part of the
#         data for test or validation.
#     shuffle: Whether to shuffle output samples,
#         or instead draw them in chronological order.
#     reverse: Boolean: if `true`, timesteps in each output sample will be
#         in reverse chronological order.
#     batch_size: Number of timeseries samples in each batch
#         (except maybe the last one).

# # We will use 12 months as input and then predict the next month out
# 

# In[19]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# In[20]:


X, y = generator[0]

print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# ### Create the Model

# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU

import tensorflow as tf


# In[22]:


# We're only using one feature in our time series
n_features = 1


# In[23]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))


# In[24]:


model.summary()


# In[25]:


model.compile(optimizer='adam', loss='mse')


# # Define your own models. 
# 
# Use 1. SimpleRNN, LSTM, or GRU neural network.
# 
# APIs:
# https://keras.io/api/layers/recurrent_layers/

# In[ ]:





# In[ ]:





# In[ ]:





# ### EarlyStopping and creating a Validation Generator
# 
# NOTE: The scaled_test dataset size MUST be greater than your length chosen for your batches. Review video for more info on this.

# In[26]:


validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)


# In[27]:


from tensorflow.keras.callbacks import EarlyStopping
get_ipython().run_line_magic('pinfo', 'EarlyStopping')
# Your code to create an object early-stop.


# Now, fit your model.

# In[28]:


# Your code to fit your model.
early_stop = EarlyStopping(monitor='val_loss',patience=5)


# In[29]:


# Get Losses from dataframe (hint - model.history.history)- See previous week tutorial.
# Plot losses in the dataframe.
history =  model.fit(generator,  epochs=30,
                    validation_data=validation_generator,
                    callbacks=[early_stop])


# In[30]:


losses = pd.DataFrame(model.history.history)


# In[31]:


losses.plot();


# In[32]:


# printing Loss for the neural network training process

history_dict = history.history
plt.style.use('seaborn-darkgrid')

acc_values = history_dict['loss']
val_acc_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)

plt.figure(num=1, figsize=(15,7))
plt.plot(epochs, acc_values, 'bo', label='Training loss')
plt.plot(epochs, val_acc_values, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (mse)')
plt.legend()

plt.show()


# ## Evaluate on Test Data

# In[33]:


first_eval_batch = scaled_train[-length:]


# In[34]:


n_input = 12
first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
model.predict(first_eval_batch)


# In[35]:


# compare with the true result:
scaled_test[0]


# #### Try predicting the series!

# In[36]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

current_batch


# In[37]:


for i in range(len(test)):
    
    print(i)
    
    # get prediction 1 time stamp ahead ([0] is for 
    # grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    print(current_pred)
    
    # store prediction
    test_predictions.append(current_pred) 
    print(test_predictions)
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    print(current_batch)


# In[38]:


test_predictions


# ## Inverse Transformations and Compare

# In[39]:


get_ipython().run_line_magic('pinfo', 'scaler.inverse_transform')


# In[40]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[41]:


# IGNORE WARNINGS
test['Predictions'] = true_predictions


# # Check and plot predictions

# In[42]:


# Print the test variable.
test


# In[43]:


# Your code to plot actual sales and predictions.
test.plot(figsize=(12,8))


# # Retrain and Forecasting

# In[44]:


full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)


# In[45]:


length = 12 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_full_data, 
                                scaled_full_data, length=length, batch_size=1)


# In[46]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))


# In[47]:


# compile the model
model.summary()


# In[48]:


# use early_stop
model.compile(optimizer='adam', loss='mse')


# In[49]:


# fit the model
early_stop = EarlyStopping(monitor='loss',patience=5)


# In[50]:


history_2 =  model.fit(generator, epochs=30, callbacks=[early_stop])


# In[51]:


history_dict = history_2.history
plt.style.use('seaborn-darkgrid')

acc_values = history_dict['loss']
#val_acc_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)

plt.figure(num=1, figsize=(15,7))
plt.plot(epochs, acc_values, 'bo', label='Training loss')
#plt.plot(epochs, val_acc_values, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (mse)')
plt.legend()

plt.show()


# In[52]:


forecast = []
# Replace periods with whatever forecast length you want
periods = 12

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(periods):
    
    # get prediction 1 time stamp ahead ([0] is for 
    # grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    forecast.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[53]:


forecast = scaler.inverse_transform(forecast)


# In[54]:


forecast


# ### Creating new timestamp index with pandas.

# In[ ]:





# In[55]:


df


# In[56]:


forecast_index = pd.date_range(start='2024-01-01',periods=periods,freq='MS')


# In[57]:


forecast_df = pd.DataFrame(data=forecast,index=forecast_index,
                           columns=['Forecast'])


# In[58]:


forecast_df


# In[59]:


# Plot sales - Values in dataframe
df.plot();


# In[60]:


# Plot forecast - Values in forecast_df
forecast_df.plot()


# ### Joining pandas plots
# 
# https://stackoverflow.com/questions/13872533/plot-different-dataframes-in-the-same-figure

# In[61]:


ax = df.plot()
forecast_df.plot(ax=ax)


# In[62]:


ax = df.plot()
forecast_df.plot(ax=ax)
plt.xlim('2022-01-01','2025-01-01')


# In[ ]:





# # Try the same example with a LSTM and GRU! 
# Hint: Use LSTM instead of SimpleRNN!
