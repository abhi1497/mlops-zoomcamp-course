#!/usr/bin/env python
# coding: utf-8

# ## Q1. Notebook
# 
# We'll start with the same notebook we ended up with in homework 1.
# We cleaned it a little bit and kept only the scoring part. You can find the initial notebook [here](homework/starter.ipynb).
# 
# Run this notebook for the March 2023 data.
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# * 1.24
# * 6.24
# * 12.28
# * 18.28

# In[ ]:


import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[ ]:


print(f"✅ Standard deviation of predictions: {np.std(y_pred)}")
# print(f"✅ Standard deviation of predictions: {round(np.std(y_pred), 2)}")
# print(f"✅ Mean predicted duration: {round(np.mean(y_pred), 2)}")
# print(f"✅ RMSE: {root_mean_squared_error(df['duration'], y_pred)}")


# ## Q2. Preparing the output
# 
# Like in the course videos, we want to prepare the dataframe with the output. 
# 
# First, let's create an artificial `ride_id` column:
# 
# ```python
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# ```
# 
# Next, write the ride id and the predictions to a dataframe with results. 
# 
# Save it as parquet:
# 
# ```python
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# ```
# 
# What's the size of the output file?
# 
# * 36M
# * 46M
# * 56M
# * 66M
# 
# __Note:__ Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the
# dtypes of the columns and use `pyarrow`, not `fastparquet`. 
# 

# In[ ]:


year = 2023
month = 3

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = df[['ride_id']].copy()
df_result['predicted_duration'] = y_pred

output_file = 'output/predictions.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[10]:


print(f'✅ Output file size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB')


# ## Q3. Creating the scoring script
# 
# Now let's turn the notebook into a script. 
# 
# Which command you need to execute for that?
# 

# In[ ]:


get_ipython().system('jupyter nbconvert --to script starter.ipynb')

