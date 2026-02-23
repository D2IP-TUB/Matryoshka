from audioop import cross
import math,copy
import pandas as pd
import numpy as np
from scipy.stats import chisquare,chi2_contingency
from difflib import SequenceMatcher
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import random
class JoinColumn:
    def __init__(self, join_path,df,column,base_df,class_attr,array_loc,uninformative=0):
        #print ("initiating")
        self.join_path = join_path
        self.loc=array_loc
        #self.dataset=dataset
        self.column=column
        self.orig_name=column
        self.base_copy=copy.deepcopy(base_df)
        self.df=df.drop_duplicates(subset=self.join_path.join_path[1].col, keep="first")
        self.profiles={'sim':self.syntactic,'corr':self.corr, 'mutual':self.syntactic}
        self.class_attr=class_attr

        self.key_r=self.join_path.join_path[1].col
        

        if column in base_df.columns:
            new_col_lst=list(self.df.columns)
            new_col_lst.append(column+"_new")
            new_col_lst.remove(column)
            new_name=column+"_new"
            self.df[new_name] = self.df[column]
            self.df=self.df[new_col_lst]
            self.column=new_name
            

        if self.key_r in base_df.columns:
            new_col_lst=list(self.df.columns)
            new_col_lst.append(self.key_r+"_new")
            new_col_lst.remove(self.key_r)
            new_name=self.key_r+"_new"
            self.df[new_name] = self.df[self.key_r]
            self.df=self.df[new_col_lst]
            self.key_r = new_name

        '''
        collst=list(base_df.columns)
        collst.append(self.column)
        self.merged_df=pd.merge(self.base_copy,self.df[[self.key_r,self.column]],left_on=self.join_path.join_path[0].col,right_on=self.key_r,how="left")
        self.merged_df=self.merged_df[collst]
        '''
        try:
            #print (self.join_path.join_path[1].col,self.join_path.join_path[0].col)
            #print (self.column)
            collst=list(base_df.columns)
            collst.append(self.column)
            self.merged_df=pd.merge(self.base_copy,self.df[[self.key_r,self.column]],left_on=self.join_path.join_path[0].col,right_on=self.key_r,how="left")
            self.merged_df=self.merged_df[collst]

        except:
            self.merged_df=copy.deepcopy(self.base_copy)
            self.merged_df[self.column]=0
            #Add 0 as the profile
        
        #self.merged_df[self.column]=self.merged_df[self.column].fillna(0)
        #print ("merged",self.merged_df)

        self.copied_df=copy.deepcopy(self.merged_df)

        for c in self.copied_df.columns:
            if self.copied_df.dtypes[c]=='object':
                self.copied_df[c]=self.copied_df[c].astype('category')
                self.copied_df[c]=self.copied_df[c].cat.codes
        #print (self.copied_df.dtypes)

        self.profile_values={}

        self.profile_values['mutual']={}

        self.copied_df[self.column]=self.copied_df[self.column].fillna(-1)

        self.profile_values['corr']=  {}#(self.copied_df.corr()[self.column])

        corr_values=self.copied_df.corr()[self.column]
        for c in self.copied_df.columns:
            if c==self.column:
                continue
            self.copied_df[c]=self.copied_df[c].fillna(-1)
            #print (self.copied_df[c])
            if np.isnan(corr_values[c]):#.isnan():# is NaN:
                self.profile_values['corr'][c] = 0
            else:
                self.profile_values['corr'][c] = corr_values[c]
            self.profile_values['mutual'][c] = adjusted_mutual_info_score(self.copied_df[c][:100],self.copied_df[self.column][:100])


        
        
        #self.profile_values['corr'].
        self.profile_values['nan'] = {'all':1-self.merged_df[self.column].isna().sum()*1.0/self.merged_df.shape[0]}
        
        for prof in self.profiles.keys():
            self.profile_values[prof]={}
            #print ("prof",self.profiles,prof,(self.merged_df.columns),self.column)
            for col in self.merged_df.columns:
                #print (col)
                if col==self.column:
                    continue
                #get profile for each col
                self.profile_values[prof][col]=self.profiles[prof](col)

        
        i=0
        self.profile_values['uninfo']={}
        while i<uninformative:
            self.profile_values['uninfo'][str(i)]=random.random()

            i+=1

        '''
        try:
            merged_df=pd.merge(base_copy,self.df[[self.join_path.join_path[1].col,column]],left_on=self.join_path.join_path[0].col,right_on=self.join_path.join_path[1].col,how="left")
            self.corr=merged_df[self.column].corr(merged_df['n. collisions'])
            #self.corr=dataset.df[column].corr(base_df['class'])
            if math.isnan(self.corr):
                self.corr=0
        except:
            self.corr=0
        '''

    
    def syntactic(self,col):
        
        return SequenceMatcher(None, col, self.orig_name).ratio()
    def corr(self,col):

        #print (self.merged_df[[col,self.column]][:10])
        cross_tab=pd.crosstab(self.merged_df[col][:4],self.merged_df[self.column][:4])
        #print (cross_tab)
        
        try:
            chi2, p, dof, ex=chi2_contingency(cross_tab)
        except:
            return 0
        #print (chi2,p)
        #print (self.merged_df.corr(method='pearson', min_periods=1))

        if p<0.1:
            print (cross_tab)
            print(chi2,p)
            print (self.merged_df[[col,self.column]][:10])
            return chi2
        else:
            return 0

        
    def get_distance (self,jc2):
        dist=0
        #print (self.profile_values)
        #print (jc2.profile_values)
        #print (self.base_copy)
        for prof in self.profile_values.keys():
            prof_dic = self.profile_values[prof]
            for col in prof_dic.keys():#self.base_copy.columns:
                curr= abs(abs(self.profile_values[prof][col]) - abs(jc2.profile_values[prof][col] ))
                if curr>dist:
                    dist=curr
        return dist

        #Option 1: correlation between columns
        try:
            corr = self.dataset.df[self.column].corr(jc2.dataset.df[jc2.column])
        except:
            corr=0
        return 1-abs(corr)
