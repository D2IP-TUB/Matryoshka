import copy
import pandas as pd
import subprocess
import warnings
from .metam_utils.backend.classifier_oracle import Oracle as ClfOracle
from .metam_utils.backend.join_column import JoinColumn
from .metam_utils.backend.join_path import get_join_paths_from_file, cluster_join_paths
from .metam_utils.backend.profile_weights import initialize_weights
from .metam_utils.backend.querying import run_metam
from .metam_utils.backend.regression_oracle import Oracle as RegOracle
from ...utils.common import process_key


class MetamAugmenter:
    def run(self, task: str, path: str, query_data: str, filepath: str, class_attr: str, epsilon: float, theta: float, uninfo: int, orig_metric: float, output_path: str, sep_lake: str, sep_query: str):
        '''
        Run the data augmentation process using Metam system.

        Parameters:
        - task (str): The type of machine learning task ('classification' or 'regression').
        - path (str): The directory path where the datasets from a certain lake are located.
        - query_data (str): The name of the query dataset to be augmented.
        - filepath (str): The path to the file containing join paths extracted from Aurum index.
        - class_attr (str): The name of the target attribute in the query dataset.
        - epsilon (float): Metam parameter.
        - theta (float): Required utility.
        - uninfo (int): Number of uninformative profiles to be added on top of default set of profiles.
        - orig_metric (float): Original performance metric score of the model on the query dataset.
        - output_path (str): The path to save the output augmented dataset.
        - sep_lake (str): The separator used in the lake CSV files.
        - sep_query (str): The separator used in the query table CSV file.
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            query_path=path+"/"+query_data
            filepath_dir = '/'.join(filepath.split('/')[:-2])
            query_data_path = filepath_dir + '/' + query_data
            subprocess.run(['cp', query_data_path, query_path])
            options = get_join_paths_from_file(query_data,filepath)[:1]
            data_dic={}
            join_paths_df = pd.read_csv(filepath).head()
            key_cols = join_paths_df['col1'].unique().tolist()
            base_df=pd.read_csv(query_path)
            for col in key_cols:
                base_df[col] = base_df[col].apply(process_key)
            joinable_lst=options

            match task:
                case "classification":
                    oracle=ClfOracle("random forest")
                case "regression":
                    oracle=RegOracle("random forest")

            i=0
            new_col_lst=[]

            while i<len(joinable_lst):
                print (i,len(new_col_lst))
                jp=joinable_lst[i]
                print (jp.join_path[0].tbl,jp.join_path[0].col,jp.join_path[1].tbl,jp.join_path[1].col)            

                if jp.join_path[0].tbl not in data_dic.keys():
                    df_l=pd.read_csv(path+"/"+jp.join_path[0].tbl,low_memory=False,on_bad_lines='skip', sep=sep_query)
                    data_dic[jp.join_path[0].tbl]=df_l
                else:
                    df_l=data_dic[jp.join_path[0].tbl]
                if jp.join_path[1].tbl not in data_dic.keys():
                    df_r=pd.read_csv(path+"/"+jp.join_path[1].tbl,low_memory=False,on_bad_lines='skip', sep=sep_lake)
                    join_col = jp.join_path[1].col
                    df_r[join_col] = df_r[join_col].apply(process_key)
                    data_dic[jp.join_path[1].tbl]=df_r
                else:
                    df_r=data_dic[jp.join_path[1].tbl]
                collst=list(df_r.columns)

                if jp.join_path[1].col not in df_r.columns or jp.join_path[0].col not in df_l.columns:
                    i+=1
                    continue
                
                for col in collst:
                    jc=JoinColumn(jp,df_r,col,base_df,class_attr,len(new_col_lst),uninfo)
                    new_col_lst.append(jc)

                i+=1

            (centers,assignment,clusters)=cluster_join_paths(new_col_lst,10,epsilon)

            tau = len(centers)

            weights={}
            weights=initialize_weights(new_col_lst[0],weights)

            metric=orig_metric
            initial_df=copy.deepcopy(base_df)
            candidates=centers

            if tau==1:
                candidates=[i for i in range(len(new_col_lst))]
            print('Initialized, starting augmentation with Metam for candidates = ', tau)
            augmented_df = run_metam(tau,oracle,candidates,theta,metric,initial_df,new_col_lst,weights,class_attr,clusters,assignment,uninfo,epsilon)    
            augmented_df.to_csv(output_path, index=False)
            subprocess.run(['rm', query_path])