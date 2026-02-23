import math
import copy 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from feature_engine.imputation import MeanMedianImputer


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as pipe
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from feature_engine.discretisation import DecisionTreeDiscretiser
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)

class Oracle:
	def __init__(self,name):
		self.name=name
	def train_classifier(self,data,target_col):

		dataset=copy.deepcopy(data)
		columns=list(dataset.columns)
		columns.remove(target_col)

		#print(dataset.dtypes)
		#TODO: string columns we need to ignore
		for col in columns:
			
			dataset[col] = dataset[col].fillna(0)
			dataset[col] = dataset[col].replace(np.nan, 0)
			#print (col,dataset[col])
			if dataset.dtypes[col]=='object':
				dataset[col] = dataset[col].astype('category')
				dataset[col] = dataset[col].cat.codes




		#columns.remove(target_col)
		dataset["class"] = dataset["class"].astype(int)
		#print (dataset[dataset['School Type']==2])
		corrMatrix = dataset.corr()['class']
		#print (corrMatrix)

		X=dataset[columns]
		y=dataset[target_col]

		#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

		print(y_train.value_counts())
		# print(X_train['class'].value_counts())
		
		train_data = pd.concat([X_train, y_train], axis=1)
		df_majority = train_data[train_data['class']==0]
		df_minority = train_data[train_data['class']==1]

		# df_minority_upsampled = resample(df_minority, 
        #                          replace=True,     # sample with replacement
        #                          n_samples=700,    # to match majority class
        #                          random_state=1) # reproducible results
	
		df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=488,     # to match minority class
                                 random_state=123) # reproducible results
		
		train_data = pd.concat([df_minority, df_majority_downsampled])

		print(train_data['class'].value_counts())

		X_train = train_data[columns]
		y_train = train_data[target_col]

		fscore=0
		seed_lst=[0]#,1,2,3,4,5]
		for seed in seed_lst:
			
			'''
			var_thr = VarianceThreshold(threshold = 0.1)
			var_thr.fit(X_train)

			#print (var_thr.get_support())
			concol = [column for column in X_train.columns if column not in X_train.columns[var_thr.get_support()]]
			X_train=X_train.drop(concol,axis=1)
			X_test=X_test.drop(concol,axis=1)
			'''
			# selector = SelectKBest(chi2, k=6).fit(X_train, y_train)
			# concol = [column for column in X_train.columns if column not in X_train.columns[selector.get_support()]]
			# X_train=X_train.drop(concol,axis=1)
			# X_test = X_test.drop(concol, axis=1)
			# scaler = StandardScaler()
			# num_d = X_train[['Total Parent Response Rate (%)', 'Total Teacher Response Rate (%)', 'Total Student Response Rate (%)']]
			# print(num_d)
			# X_train[num_d.columns] = scaler.fit_transform(num_d)
			# X_scaled = scaler.fit_transform(X_train)
	
			# X_train = pd.DataFrame(X_scaled, columns = X_train.columns)
			clf = AdaBoostClassifier(random_state=seed)
			clf.fit(X_train, y_train)
			#print(X_test)
			y_pred=clf.predict(X_test)
			(tn, fp, fn, tp) = confusion_matrix(y_test, y_pred).ravel()
			X_test['true_label'] = y_test
			X_test['pred_label'] = y_pred
			
			recall=tp*1.0/(tp+fn)

			precision=tp*1.0/(tp+fp)
			fscore += 2*precision*recall*1.0/(precision+recall)

			accuracy = (tp+tn)*1.0/(tp+fp+tn+fn)
			#print ("current fscore",2*precision*recall*1.0/(precision+recall),(tp+tn)*1.0/(tp+fp+tn+fn))

		count=0
		total=0
		rule=0
		notrule=0
		'''
		for iter,row in X_test.iterrows():
			if row['School Type']==2 and row['true_label'] != row['pred_label']:
				count+=1
				print (row)
			elif row['School Type']==2:
				total+=1
			if row['School Type'] ==2 and row['Total Student Response Rate (%)'] ==0:
				if row['pred_label']==0:
					rule+=1
				else:
					notrule+=1

		print(count,count+total,rule,notrule)
		print (dataset.corr())
		'''

		#print ("Precision:",precision, "\nRecall:",recall)
		#print ("F-score:",fscore*1.0/len(seed_lst), "\nAccuracy:",accuracy)
		
		#print (clf.feature_importances_)
		#return scores.mean()
		return fscore/len(seed_lst)

	#def oracle_score(self,join_paths):
		


if __name__ == '__main__':
	path='/home/cc/opendata_cleaned/'#/Users/yuegong/Documents/metam'#/Users/sainyam/Documents/MetamDemo/sigmod23/open_data_usa/' # path to csv files
	query_data='base_school.csv'
	query_path=path+"/"+query_data

	base_df=pd.read_csv(query_path)
	oracle=Oracle("random forest")

	orig_metric=oracle.train_classifier(base_df,'class')

	print (orig_metric)

	


