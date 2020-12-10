import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
import math

taxi_train = pd.read_csv("train.csv")
taxi_test = pd.read_csv("test.csv")
#weather = pd.read_csv("ny_2016_weather.csv")

### Column Info ###
# * id - a unique identifier for each trip
# * vendor_id - a code indicating the provider associated with the trip record
# * pickup_datetime - date and time when the meter was engaged
# * dropoff_datetime - date and time when the meter was disengaged
# * passenger_count - the number of passengers in the vehicle (driver entered
#   value)
# * ickup_longitude - the longitude where the meter was engaged
# * pickup_latitude - the latitude where the meter was engaged
# * dropoff_longitude - the longitude where the meter was disengaged
# * dropoff_latitude - the latitude where the meter was disengaged
# * store_and_fwd_flag - This flag indicates whether the trip record was held
#   in vehicle memory before sending to the vendor because the vehicle did not
#   have a connection to the server - Y=store and forward; N=not a store and
#   forward trip
# * trip_duration - duration of the trip in seconds

#pick_lon = taxi_train['pickup_longitude']
#pick_lat = taxi_train['pickup_latitude']
#
#drop_lon = taxi_train['dropoff_longitude']
#drop_lat = taxi_train['dropoff_longitude']

#plot = plt.plot(pick_lon, pick_lat)

#pd.cut(taxi.train['pickup_longitude'], bins = 50,


#Feature Engineering

#Fix coordinates

taxi_train["pickup_longitude"] = np.radians(taxi_train["pickup_longitude"])
taxi_train["pickup_latitude"] = np.radians(taxi_train["pickup_latitude"])
taxi_train["dropoff_longitude"] = np.radians(taxi_train["dropoff_longitude"])
taxi_train["dropoff_latitude"] = np.radians(taxi_train["dropoff_latitude"])

taxi_test["pickup_longitude"] = np.radians(taxi_test["pickup_longitude"])
taxi_test["pickup_latitude"] = np.radians(taxi_test["pickup_latitude"])
taxi_test["dropoff_longitude"] = np.radians(taxi_test["dropoff_longitude"])
taxi_test["dropoff_latitude"] = np.radians(taxi_test["dropoff_latitude"])

#Fix store_and_fwd_flag

taxi_train['store_and_fwd_flag'] = taxi_train['store_and_fwd_flag'].eq('Y').mul(1)
taxi_test['store_and_fwd_flag'] = taxi_test['store_and_fwd_flag'].eq('Y').mul(1)

# Is the Pickup During Rush Hour?
morn_rush_start = datetime.datetime.strptime('07:30:00', '%H:%M:%S')
morn_rush_end = datetime.datetime.strptime('09:00:00', '%H:%M:%S')
eve_rush_start = datetime.datetime.strptime('16:30:00', '%H:%M:%S')
eve_rush_end = datetime.datetime.strptime('19:00:00', '%H:%M:%S')

def during_rush(x):
    if ((x.time() >= morn_rush_start.time() and x.time() <= morn_rush_end.time())
            or (x.time() >= eve_rush_start.time() and x.time() <= eve_rush_end.time())):
        return 1
    else:
        return 0

taxi_train['pickup_datetime'] = taxi_train['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d  %H:%M:%S'))
taxi_test['pickup_datetime'] = taxi_test['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d  %H:%M:%S'))

taxi_train['during_rush'] = taxi_train['pickup_datetime'].apply(during_rush)
taxi_test['during_rush'] = taxi_test['pickup_datetime'].apply(during_rush)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

feature_names = (['vendor_id', 'passenger_count',
                    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                    'dropoff_latitude', 'store_and_fwd_flag', 'during_rush'])

X = taxi_train[feature_names]
y = taxi_train['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lr = LinearRegression(fit_intercept=True, normalize=False)

lr.fit(X_train, y_train)
yhat = (lr.predict(X_test))


new_test = taxi_test.drop(['id','pickup_datetime'], axis=1)

submit = lr.predict(new_test)

submit = pd.Series(submit, name = 'trip_duration')

submission = pd.concat([taxi_test.id, submit], axis=1)
submission = submission.set_index('id')
submission.to_csv('taxi_test.csv')


