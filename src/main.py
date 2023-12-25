# !/usr/bin/python3
from data_helper import *
from feature_selection import *
from sklearn.preprocessing import RobustScaler
from os.path import join
from result import prettifier
import pickle

path = os.path.realpath(__file__)

for family in ["australian", "wisconsin", "votes", "nursery", "careval"]:
    # get data
    dh = DataHolder()
    X, y = dh.get(family)
    scaler = RobustScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])
    dh.show_details(X, y)

    # feature_selection
    lambda_list = [
        # [lambda(-1), lambda(1)]
        [0.75, 0.85], [0.8, 0.85], [0.65, 0.9]
    ]
    for lambda_ in lambda_list:
        res = feature_selection(X, y, lambda_=lambda_)
        
        # output
        prettifier(res)

        # write file
        with open(f"{path[:-12]}/results/{family}_{lambda_[0]}_{lambda_[1]}.txt", mode="w") as f:
            print(res, file=f)
