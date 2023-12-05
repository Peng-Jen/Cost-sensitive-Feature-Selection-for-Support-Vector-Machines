import os
import pandas as pd
from ucimlrepo import fetch_ucirepo 

class DataHolder():
    def __init__(self) -> None:

        self.files_id = {
            "careval": 19,
            "wisconsin": 17,
            "votes": 105,
        }

    def get(self, name):
        if name == "nursery":
            df = self.fetch_dataset(name)
            X = df.loc[:, df.columns[:-1]]
            y = df[df.columns[-1]]

        elif name == "statlog":
            df = self.fetch_dataset(name)
            X = df.loc[:, df.columns[:-1]]
            y = df[df.columns[-1]]
            
        elif name == "gastrointestinal":
            df = self.fetch_dataset(name)
            X = df.iloc[:, 3:]
            y = df.iloc[:, :3]

        else:
                    
            df = self.fetch_dataset(name)
            X = df.data.features 
            y = df.data.targets 
            
        return X, y
        
    def fetch_dataset(self, name):

        prefix = 'data'+os.sep

        if name == 'nursery':

            file_path = prefix+name+os.sep+name+'.data'
            df = pd.read_csv(file_path, header=None) 
            attr = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
            df.columns = attr

        elif name == 'statlog':

            file_path = prefix+name+os.sep+'australian.dat'
            df = pd.read_csv(file_path, delimiter=' ', header=None) 
            attr = [f'A{i+1}' for i in range(15)]
            df.columns = attr

        elif name == 'gastrointestinal':

            file_path = prefix+name+os.sep+'data.txt'
            df = pd.read_csv(file_path, header=None)
            list_of_rows = df.values.tolist()
            raw_features = list_of_rows[3:]
            labels_name = [f'label{i+1}' for i in range(3)]
            features_name = [f'A{i+1}' for i in range(698)]
            attr = labels_name + features_name
            df = pd.DataFrame(list_of_rows).transpose()
            df.columns = attr

        elif name == 'votes':

            df = fetch_ucirepo(id=self.files_id[name]) 
            df = self.impute(df)
            
        else:
            df = fetch_ucirepo(id=self.files_id[name]) 

        return df
    
    def impute(self, df):
        columns_with_missing_values = df.data.features.columns[df.data.features.isna().any()].tolist()
        df.data.features.loc[:, columns_with_missing_values] = \
            df.data.features[columns_with_missing_values].fillna(df.data.features[columns_with_missing_values].mode().iloc[0])
        return df

if __name__ == '__main__':
    
    dh = DataHolder()
    X, y = dh.get('gastrointestinal')
    X, y = dh.get('nursery')
    X, y = dh.get('statlog')
    X, y = dh.get('careval')
    X, y = dh.get('wisconsin')
    X, y = dh.get('votes')
   
    
    
    