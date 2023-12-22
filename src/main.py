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
    lambda_ = [0.55, 0.85]
    res = feature_selection(X, y, lambda_=lambda_)
    
    # output
    prettifier(res)

    # write file
    with open(f"{path[:-12]}/results/{family}_{lambda_[0]}_{lambda_[1]}.txt", mode="w") as f:
        print(res, file=f)


# dataset = "australian"
# lambda_ = [0.6, 0.9] # [\lambda_{-1}, \lambda_1]
# print(f"Using lambda = {lambda_}")
# dh = DataHolder()
# X, y = dh.get(dataset)  # not yet 我通靈不到他怎麼算的
# scaler = RobustScaler()
# X[X.columns] = scaler.fit_transform(X[X.columns])

# dh.show_details(X, y)
# y = y.replace(0, -1)
# res = feature_selection(X, y, lambda_=lambda_)



# with open(f"../results/{dataset}_{lambda_[0]},{lambda_[1]}.pickle","wb") as f:
#     pickle.dump(res, f)
