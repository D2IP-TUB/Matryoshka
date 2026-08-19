from causallearn.search.FCMBased.ANM.ANM import ANM
import sys
import pandas as pd
import copy 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import VarianceThreshold


class Oracle:
    def __init__(self,name):
        self.name=name

    def causal_inference(self,source, target):
        anm = ANM()
        n = len(source)
      
        source = source.to_numpy().reshape(n,1)
        target = target.to_numpy().reshape(n,1)
       
        p_value_foward, p_value_backward = anm.cause_or_effect(source, target)
        return p_value_foward, p_value_backward


    def analyze_causal_inference(self,source_table, source_col):
        #source_table[source_col] = source_table[source_col].apply(lambda x: float(x))
        count=0
        for target in source_table.columns:
            print (target)
            if source_table.dtypes[target]=='object':
                continue

            to_analyze = source_table[[source_col, target]].dropna()
            p_forward, p_backward = self.causal_inference(to_analyze[source], to_analyze[target])
            print(target, p_forward, p_backward)
            if p_forward<0.05:
                count+=1
        return count
    

if __name__ == '__main__':
    path='/Users/sainyam/Documents/MetamDemo/sigmod23/open_data_usa/f9bf-2cp4.csv'#/Users/sainyam/Documents/MetamDemo/sigmod23/open_data_usa/' # path to csv files
   

    base_df=pd.read_csv(path)
    oracle=Oracle("random forest")

    orig_metric=oracle.analyze_causal_inference(base_df,'SAT Critical Reading Avg. Score')
    
    
    print(orig_metric)
