#!/usr/bin/python3
from data_helper import *
from feature_selection import *
from sklearn.preprocessing import RobustScaler

dh = DataHolder()
X, y = dh.get('gastrointestinal') # not yet 我通靈不到他怎麼算的
# X, y = dh.get('nursery') # done
# X, y = dh.get("australian")  # done -> positive count 307 vs. 383
# X, y = dh.get('careval') # done
# X, y = dh.get('wisconsin') # done -> positive count 212 vs. 357
# X, y = dh.get('votes') # done -> positive count 168 vs. 267

# min max
#     scaler = MinMaxScaler()
scaler = RobustScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])

dh.show_details(X, y)
#     X = X.replace(0, -1)
y = y.replace(0, -1)

# res = feature_selection(X, y, lambda_=[0.5, 0.5], radial_kernel=True)
res = feature_selection(X, y, lambda_=[0.5, 0.5])
