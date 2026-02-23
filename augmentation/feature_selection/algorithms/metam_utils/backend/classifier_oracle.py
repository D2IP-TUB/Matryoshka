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

		
		dataset["class"] = dataset["class"].astype(int)

		corrMatrix = dataset.corr()['class']
		#print (corrMatrix)

		X=dataset[columns]
		y=dataset[target_col]

		#print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
		fscore=0
		seed_lst=[2]
		for seed in seed_lst:
			

			var_thr = VarianceThreshold(threshold = 0.1)
			var_thr.fit(X_train)

			#print (var_thr.get_support())
			concol = [column for column in X_train.columns if column not in X_train.columns[var_thr.get_support()]]
			X_train=X_train.drop(concol,axis=1)
			X_test=X_test.drop(concol,axis=1)



			clf = RandomForestClassifier(random_state=seed)
			clf.fit(X_train, y_train)
			#print(X_test)
			y_pred=clf.predict(X_test)
			(tn, fp, fn, tp) = confusion_matrix(y_test, y_pred).ravel()
			X_test['true_label'] = y_test
			X_test['pred_label'] = y_pred
			#print(X_test)

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
	path='/home/cc/open_data_usa/'#/Users/sainyam/Documents/MetamDemo/sigmod23/open_data_usa/' # path to csv files
	query_data='base_school.csv'
	query_path=path+"/"+query_data

	base_df=pd.read_csv(query_path)
	oracle=Oracle("random forest")

	orig_metric=oracle.train_classifier(base_df,'class')
	new_df=pd.read_csv(path+'2010_Gen_Ed_Survey_Data.csv')#test1.csv')
	new_df2=pd.read_csv(path+'test2.csv')

	#merged_df['newcol']=new_col_lst[candidate_id].merged_df[new_col_lst[candidate_id].column]
        

	merged_df=pd.merge(base_df,new_df,left_on='DBN',right_on='dbn',how="left")
	merged_df2=pd.merge(base_df,new_df2,left_on='School Name',right_on='SCHOOL',how="left")
	#self.merged_df=self.merged_df[collst]	

	tmp_metric=max(oracle.train_classifier(merged_df,'class'),orig_metric)
	print(tmp_metric)
	tmp_metric=max(oracle.train_classifier(merged_df2,'class'),orig_metric)
	print(merged_df2.columns)
	print (orig_metric,tmp_metric)
