import copy
import math
from .profile_weights import sort_candidates, get_weights
import random
from .group_helper import identify_group_query

def run_metam(tau,oracle,candidates,theta,metric,initial_df,new_col_lst,weights,class_attr,clusters,assignment,uninfo,epsilon):


    likelihood_num=[]
    likelihood_den=[]
    for c in clusters:
        likelihood_num.append(1)
        likelihood_den.append(1)

    cluster_size={}
    for k in assignment.keys():
        if assignment[k] in cluster_size.keys():
            cluster_size[assignment[k]]+=1
        else:
            cluster_size[assignment[k]]=1

    stopping_criterion=1000#10000
    base_df=copy.deepcopy(initial_df)
    orig_metric = oracle.train_classifier(base_df,class_attr)
    fout=open("output.txt","w")
    total_queries=0
    iter=0
    grp_size=1
    grp_queried_cand={}
    run_seq=True
    overall_queried={}
    while True:
        print("Starting iteration ", iter, " current metric ", metric, " total queries so far ", total_queries)
        if metric >=theta:
            break

        i=0

        max_file=''
        curr_max = metric
        max_candidate=initial_df
        max_jc =''
        if tau>1:
            queried_cand = {}

        if tau==1 and iter==0:
            queried_cand={}
        curr_max_grp = metric
        max_grp_file=''
        max_candidate_grp=base_df
        while i<tau or curr_max <= metric:
            print("Current group max ", curr_max_grp, " overall max ", curr_max)

            #Choose the candidate with maximum score
            if iter==0 or i==0:
                sorted_cand = sort_candidates(new_col_lst,candidates,weights,overall_queried)
            #print (sorted_cand)
            j=0
            while j<len(sorted_cand):
                print("Current candidate ", sorted_cand[j])
                print("Queried candidates so far ", (queried_cand))
                if sorted_cand[j][0] in queried_cand.keys():
                    j+=1
                    continue
                else:
                    break
                j+=1
            
            if j==len(sorted_cand):
                break

            #Query the candidate 
            print ("Chosen candidate in iteration ", i,len(queried_cand))
            print ("Candidate id and score ", sorted_cand[j])

            if j>0 and sorted_cand[j][1]==0:
                candidate_id = sorted_cand[j][0]
                #fout.write("zero score"+new_col_lst[candidate_id].join_path.join_path[1].tbl+";"+new_col_lst[candidate_id].join_path.join_path[1].col+";"+new_col_lst[candidate_id].column+'\n')
                break#continue
            elif sorted_cand[j][1]==0:
                run_seq=False
            else:
                run_seq=True

            if total_queries > stopping_criterion:
                break
            if run_seq:
                merged_df=copy.deepcopy(initial_df)
                candidate_id = sorted_cand[j][0]
                merged_df[new_col_lst[candidate_id].column]=new_col_lst[candidate_id].merged_df[new_col_lst[candidate_id].column]
                tmp_metric=max(oracle.train_classifier(merged_df,class_attr),metric)
                print("iteration",tmp_metric,new_col_lst[candidate_id].join_path.join_path[1].tbl+";"+new_col_lst[candidate_id].join_path.join_path[1].col)
                #fout.write(new_col_lst[candidate_id].join_path.join_path[1].tbl+";"+new_col_lst[candidate_id].join_path.join_path[1].col+";"+new_col_lst[candidate_id].column+'\n')

                queried_cand[candidate_id] = tmp_metric-metric
                total_queries+=1
                fout.write(str(max(curr_max,curr_max_grp))+" "+str(total_queries)+"\n")
                if tmp_metric > curr_max:
                    curr_max=tmp_metric
                    max_candidate = merged_df
                    max_file=new_col_lst[candidate_id].join_path.join_path[1].tbl+";"+new_col_lst[candidate_id].join_path.join_path[1].col
                    print ("metric",tmp_metric,new_col_lst[candidate_id].join_path.join_path[1].tbl)#,corr)
                    print(new_col_lst[candidate_id].profile_values)
                    print (merged_df)
                    max_jc = candidate_id

                #4a
                #break

            

            
            if len(list(grp_queried_cand.keys())) == len(new_col_lst):
                grp_size*=2

            (jc_lst,jc_representation)=identify_group_query(new_col_lst,clusters,grp_size,likelihood_num,likelihood_den,grp_queried_cand)
            grp_merged_df=copy.deepcopy(base_df)
            for jc in jc_lst:
                grp_merged_df[jc.column]=jc.merged_df[jc.column]
            #fout.write(jc_lst[0].join_path.join_path[1].tbl+";"+jc_lst[0].join_path.join_path[1].col+";"+jc_lst[0].column+'\n')
            tmp_metric=max(oracle.train_classifier(grp_merged_df,class_attr),orig_metric)
            if tmp_metric > orig_metric and len(jc_lst)==1:
                candidates.append(jc_lst[0].loc)
            if len(jc_lst)==1:
                queried_cand[jc_lst[0].loc] = tmp_metric-orig_metric
                #if tmp_metric-orig_metric > 0:
                #    candidates.append(jc_lst[0].loc)        
            grp_queried_cand[jc_representation] = tmp_metric
            total_queries+=1
            fout.write(str(max(curr_max,curr_max_grp))+" "+str(total_queries)+"\n")
            if tmp_metric > curr_max_grp:
                curr_max_grp=tmp_metric
                max_candidate_grp = grp_merged_df
                print ("metric",tmp_metric,new_col_lst[candidate_id].join_path.join_path[1].tbl)#,corr)
                max_grp_file=new_col_lst[candidate_id].join_path.join_path[1].tbl+";"+new_col_lst[candidate_id].join_path.join_path[1].col
                print(new_col_lst[candidate_id].profile_values)
                print (grp_merged_df)
            #Likelihood update
            for jc in jc_lst:
                clust_id=assignment[jc]
                if tmp_metric>orig_metric:
                    likelihood_num[clust_id]+=1
                likelihood_den[clust_id]+=1

            if iter==0:
                weights=get_weights(new_col_lst,base_df,queried_cand,weights,uninfo)
                #if len(list(queried_cand.keys())) > 10:
                #    print (weights)
            i+=1


        weights=get_weights(new_col_lst,base_df,queried_cand,weights,uninfo)
        if total_queries>stopping_criterion:
            break


        if iter==0 and tau>1:
            cl_iter=0
            k=len(clusters)
            while cl_iter < k:
                lst=clusters[cl_iter]
                if len(lst)==0:
                    cl_iter+=1
                    continue
                #Get a sample of lst to check if homogenous
                samp_iter=0
                irregularity_count=0

                count=math.ceil(math.log(len(lst))/math.log(2))
                print (len(lst),count)
                if count <=1:
                    cl_iter+=1
                    continue

                #Utility of center
                queried_lst=[]

                while samp_iter<count:
                    curr_samp=lst[random.randint(0,len(lst)-1)]
                    merged_df=copy.deepcopy(initial_df)

                    merged_df[curr_samp.column]=curr_samp.merged_df[curr_samp.column]
                    if not curr_samp in queried_cand.keys():
                        total_queries+=1
                    tmp_metric=max(oracle.train_classifier(merged_df,class_attr),metric)
                    queried_cand[curr_samp] = tmp_metric-metric
                    queried_lst.append(tmp_metric)

                    if tmp_metric > curr_max:
                        curr_max=tmp_metric
                        max_candidate = merged_df
                        max_file=curr_samp.join_path.join_path[1].tbl+";"+curr_samp.join_path.join_path[1].col
                        print ("metric",tmp_metric,curr_samp.join_path.join_path[1].tbl)#,corr)
                        print (merged_df)
                        max_jc = curr_samp.loc
                    #    irregularity_count+=1
                    fout.write(str(max(curr_max,curr_max_grp))+" "+str(total_queries)+"\n")
                    samp_iter+=1
                
                #Check mean (1+epsilon)
                print (queried_lst)
                mean_sc=sum(queried_lst)* 1.0/len(queried_lst)
                count=0
                for v in queried_lst:
                    if v > mean_sc*(1+epsilon) or v < mean_sc*1.0/(1+epsilon):
                        count+=1
                if count > len(queried_lst)/2:
                    print ("not a homogenous cluster", mean_sc, count, len(lst))
                    for c in lst:
                        candidates.append(c.loc)
                
                fout1=open('log.txt','a')
                fout1.write(str(cl_iter)+" cluster count "+str(count)+" "+str(len(queried_lst))+"\n")
                fout1.close()
                cl_iter+=1
            #Check clusters and update candidates
        print ("length of candidates",len(candidates))
        print ("Chosen first augmentation", curr_max,metric)
        print (max_jc)
        #print ("Max table and column are ", new_col_lst[max_jc].join_path.join_path[1].tbl,new_col_lst[max_jc].join_path.join_path[1].col)
        for c in queried_cand.keys():
            overall_queried[c]=queried_cand[c]
        if curr_max_grp > curr_max:
            metric = curr_max_grp
            initial_df = max_candidate_grp
            fout1=open('log.txt','a')
            fout1.write(max_grp_file+str(metric)+"\n")
            fout1.close()
        elif curr_max==metric:
            iter+=1
            continue
        else:
            metric = curr_max
            fout1=open('log.txt','a')
            fout1.write(max_file+str(metric)+"\n")
            fout1.close()
            initial_df = max_candidate


       
        iter+=1


    return (initial_df)
