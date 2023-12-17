#!/usr/bin/python3
from data_helper import *
from feature_selection import *
from sklearn.preprocessing import RobustScaler
from os.path import join

# for family in ["wisconsin", "votes", "nursery", "australian", "careval"]:
for family in ["votes", "nursery", "australian", "careval"]:
    dh = DataHolder()
    X, y = dh.get("gastrointestinal")  # not yet 我通靈不到他怎麼算的
    scaler = RobustScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    dh.show_details(X, y)
    # X = X.replace(0, -1)
    y = y.replace(0, -1)
    res = feature_selection(X, y, lambda_=[0.85, 0.5])
    with open(join("log", family+".result"), "w") as f:
        print(res, file=f)
