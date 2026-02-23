import copy 
import pandas as pd
import numpy as np

import autosklearn.classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import VarianceThreshold


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

		
		dataset["class"] = dataset["class"].astype(int)

		corrMatrix = dataset.corr()['class']
		#print (corrMatrix)

		X=dataset[columns]
		y=dataset[target_col]

		#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
		fscore=0
		seed_lst=[0]#,1,2,3,4,5]
		for seed in seed_lst:
			

			var_thr = VarianceThreshold(threshold = 0.1)
			var_thr.fit(X_train)

			#print (var_thr.get_support())
			concol = [column for column in X_train.columns if column not in X_train.columns[var_thr.get_support()]]
			X_train=X_train.drop(concol,axis=1)
			X_test=X_test.drop(concol,axis=1)



			clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder="/tmp/autosklearn_parallel_1_example_tmp",
        n_jobs=4,
        # Each one of the 4 jobs is allocated 3GB
        memory_limit=30072,
        seed=5)#,random_state=seed)
			clf.fit(X_train, y_train)

			y_pred=clf.predict(X_test)
			(tn, fp, fn, tp) = confusion_matrix(y_test, y_pred).ravel()

			recall=tp*1.0/(tp+fn)

			precision=tp*1.0/(tp+fp)
			fscore += 2*precision*recall*1.0/(precision+recall)

			accuracy = (tp+tn)*1.0/(tp+fp+tn+fn)
			#print ("current fscore",2*precision*recall*1.0/(precision+recall),(tp+tn)*1.0/(tp+fp+tn+fn))
		#print ("Precision:",precision, "\nRecall:",recall)
		#print ("F-score:",fscore*1.0/len(seed_lst), "\nAccuracy:",accuracy)
		
		#print (clf.feature_importances_)
		#return scores.mean()
		return fscore/len(seed_lst)

	#def oracle_score(self,join_paths):
		


if __name__ == '__main__':
	path='/home/cc/open_data_usa/'#'/Users/sainyam/Documents/MetamDemo/sigmod23/open_data_usa/' # path to csv files
	query_data='augmented_data.csv'#base_school.csv'
	query_path="./"+query_data
	#query_path=path+"/"+query_data

	base_df=pd.read_csv(query_path)
	oracle=Oracle("random forest")

	orig_metric=oracle.train_classifier(base_df,'class')
	print (orig_metric)
