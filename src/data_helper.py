import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


class DataHolder:
    def __init__(self) -> None:
        self.files_id = {
            "careval": 19,
            "wisconsin": 17,
            "votes": 105,
        }

    def get(self, name):
        if name == "nursery":
            df = self.fetch_dataset(name)
            #             df = df.sample(frac=1)
            X = df.loc[:, df.columns[:-1]]
            X = self.encode(X)
            y = df[df.columns[-1]].to_frame("class")
            y = self.multiclass_preprocessing(y)

        elif name == "australian":
            df = self.fetch_dataset(name)
            X = df.loc[:, df.columns[:-1]]
            X["A1"] = X["A1"].replace({0: "a", 1: "b"})
            X["A4"] = X["A4"].replace({1: "p", 2: "g", 3: "gg"})
            X["A5"] = X["A5"].replace(
                {
                    1: "ff",
                    2: "d",
                    3: "i",
                    4: "k",
                    5: "j",
                    6: "aa",
                    7: "m",
                    8: "c",
                    9: "w",
                    10: "e",
                    11: "q",
                    12: "r",
                    13: "cc",
                    14: "x",
                }
            )
            X["A6"] = X["A6"].replace(
                {
                    1: "ff",
                    2: "dd",
                    3: "j",
                    4: "bb",
                    5: "v",
                    6: "n",
                    7: "o",
                    8: "h",
                    9: "z",
                }
            )
            X["A8"] = X["A8"].replace({0: "f", 1: "t"})
            X["A9"] = X["A9"].replace({0: "f", 1: "t"})
            X["A11"] = X["A11"].replace({0: "0", 1: "t"})
            X["A12"] = X["A12"].replace({1: "s", 2: "g", 3: "p"})
            X = self.encode(X)
            y = df[df.columns[-1]].to_frame("class")

        elif name == "gastrointestinal":
            df = self.fetch_dataset(name)
            X = df.iloc[:, 3:]
            t1 = df.iloc[:, 0].to_frame("class")
            t2 = df.iloc[:, 1].to_frame("class")
            t3 = df.iloc[:, 2].to_frame("class")
            indices_with_WL = t3[t3["class"] == "1"].index
            indices_with_NBI = t3[t3["class"] == "2"].index
            result_t2 = t2.loc[indices_with_WL]
            X = X.loc[indices_with_WL]

            y = self.multiclass_preprocessing(result_t2)
            y = self.encode(y)

        elif name == "votes":
            df = self.fetch_dataset(name)
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]
            X = self.encode(X)
            y = self.encode(y)

        else:
            df = self.fetch_dataset(name)

            X = df.data.features
            y = df.data.targets

            df = pd.concat([X, y], axis=1)
            df = df.sample(frac=1)  # shuffle

            if name == "careval":
                X = self.encode(X)
                y = self.multiclass_preprocessing(y)

            y = self.encode(y)

        return X, y

    def fetch_dataset(self, name):
        prefix = "data" + os.sep

        if name == "nursery":
            file_path = prefix + name + os.sep + name + ".data"
            df = pd.read_csv(file_path, header=None)
            attr = [
                "parents",
                "has_nurs",
                "form",
                "children",
                "housing",
                "finance",
                "social",
                "health",
                "class",
            ]
            df.columns = attr

        elif name == "votes":
            file_path = prefix + name + os.sep + "house-votes-84.data"
            df = pd.read_csv(file_path, delimiter=",", header=None)
            class_name = ["class"]
            features_name = [f"A{i+1}" for i in range(len(df.columns) - 1)]
            column_names = class_name + features_name
            df.columns = column_names

        elif name == "australian":
            file_path = prefix + name + os.sep + "australian.dat"
            df = pd.read_csv(file_path, delimiter=" ", header=None)
            attr = [f"A{i+1}" for i in range(15)]
            df.columns = attr

        elif name == "gastrointestinal":
            file_path = prefix + name + os.sep + "data.txt"
            df = pd.read_csv(file_path, header=None)
            list_of_rows = df.values.tolist()
            raw_features = list_of_rows[3:]
            labels_name = [f"label{i+1}" for i in range(3)]
            features_name = [f"A{i+1}" for i in range(698)]
            attr = labels_name + features_name
            df = pd.DataFrame(list_of_rows).transpose()
            df.columns = attr

        else:
            df = fetch_ucirepo(id=self.files_id[name])

        return df

    def impute(self, df):
        columns_with_missing_values = df.data.features.columns[
            df.data.features.isna().any()
        ].tolist()

        # imputation with mode
        df.data.features.loc[:, columns_with_missing_values] = df.data.features[
            columns_with_missing_values
        ].fillna(df.data.features[columns_with_missing_values].mode().iloc[0])
        return df

    def encode(self, df):
        df = pd.get_dummies(df, drop_first=True)
        df = df.astype(int)
        return df

    def multiclass_preprocessing(self, y):
        majority_class = y["class"].mode().iloc[0]
        y_copy = y.copy()
        y_copy["binary_label"] = y_copy["class"].apply(
            lambda x: 1 if x == majority_class else 0
        )
        y_copy = y_copy.drop("class", axis=1)
        return y_copy

    def show_details(self, X, y):
        nums_of_row = X.shape[0]
        nums_of_feature = X.shape[1]
        positive_count = y.sum()
        print(f"nums_of_row:{nums_of_row}")
        print(f"nums_of_feature:{nums_of_feature}")
        if positive_count.values < (nums_of_row - positive_count.values):
            positive_count == nums_of_row - positive_count.values
        print(f"positive_count:{positive_count.values}")
