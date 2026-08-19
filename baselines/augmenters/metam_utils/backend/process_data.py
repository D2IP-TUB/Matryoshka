'''
This file can be used to run for schools data with classification task.

For other tasks, please change line 19 to import respective Oracle, e.g. regression_oracle for Figure 3b
causal_oracle for Figure 3c.

'''


from distutils.ccompiler import new_compiler
import profile_weights
import os,copy
import profile
from sklearn.feature_selection import mutual_info_classif
from os import listdir
from os.path import isfile, join
import pandas as pd
from dataset import Dataset
from classifier_oracle import Oracle
import math
import pandas as pd
from join_path import JoinKey, JoinPath
from join_column import  JoinColumn
import sys
import pickle
import join_path
import operator,random
from sklearn import datasets, linear_model
import group_helper,querying
random.seed(0)

path='/home/cc/opendata_cleaned/'#open_data_usa/' # path to csv files
query_data='base_school.csv'
class_attr='class'
query_path=path+"/"+query_data
model_path = '../../opendata_graph/'#'/Users/sainyam/Documents/MetamDemo/models/'#  # path to the graph model
uninfo=0
epsilon=0.05

theta = 0.90
filepath='/home/cc/network_opendata_06.csv'




def get_size_dic(path):
    size_dic={}
    f=open(path+'size_lst.txt')
    for line in f:
        line=line.strip()
        line=line.split()
        size_dic[line[1]]=int(line[0])
    return size_dic

def get_ignore_lst():
    ignore_lst=[]
    f=open('ignore.txt','r')
    for line in f:
        line=line.strip()
        ignore_lst.append(line)
    f.close()
    return ignore_lst





options = join_path.get_join_paths_from_file(query_data,filepath)


files = [f for f in listdir(path) if isfile(join(path, f))]


dataset_lst=[]
data_dic={}
iter=0




base_df=pd.read_csv(query_path)


joinable_lst=options


oracle=Oracle("random forest")
orig_metric=oracle.train_classifier(base_df,'class')


print ("original metric is ",orig_metric)

size_dic=get_size_dic(path)
ignore_lst=get_ignore_lst()

i=0
new_col_lst=[]


(new_col_lst,skip_count) = join_path.get_column_lst(joinable_lst)

print ("skip count",skip_count)
print ("length is ",len(new_col_lst))


(centers,assignment,clusters)=join_path.cluster_join_paths(new_col_lst,100,epsilon)
print (centers)



tau = len(centers)
weights={}
weights=profile_weights.initialize_weights(new_col_lst[0],weights)

metric=orig_metric
initial_df=copy.deepcopy(base_df)
candidates=centers


if tau==1:
    candidates=[i for i in range(len(new_col_lst))]


augmented_df= querying.run_metam(tau,oracle,candidates,theta,metric,initial_df,new_col_lst,weights,class_attr,clusters,assignment,uninfo,epsilon)    
augmented_df.to_csv('augmented_data.csv')

