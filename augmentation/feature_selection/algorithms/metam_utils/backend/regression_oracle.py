import pandas as pd
import copy 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class Oracle:
    def __init__(self,name):
        self.name=name
    def train_classifier(self,data,target_col):
        dataset=copy.deepcopy(data)
        columns=list(dataset.columns)

        columns.remove(target_col)


        for col in columns:
            dataset[col] = dataset[col].fillna(0)
            dataset[col] = dataset[col].replace(np.nan, 0)
            #print (col,dataset[col])
            if dataset.dtypes[col]=='object':
                dataset[col] = dataset[col].astype('category')
                dataset[col] = dataset[col].cat.codes

        #corrMatrix = dataset.corr()['class']
        #print (corrMatrix)
        mae=0
        for seed in [4]:#[0,1,2,3,4,5]:
            X=dataset[columns]
            y=dataset[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            clf = RandomForestRegressor(random_state=0)
            clf.fit(X_train, y_train)
                    
            y_pred=clf.predict(X_test)
            mae += (mean_absolute_error(y_test,y_pred))

        mae=mae
        return (100-mae)/100
if __name__ == '__main__':
    query_data='/home/cc/open_data_usa/base_taxi.csv'
    class_attr='n. collisions' 
    base_df=pd.read_csv(query_data)
    oracle=Oracle("random forest")

    orig_metric=oracle.train_classifier(base_df,class_attr)
    print(orig_metric)


