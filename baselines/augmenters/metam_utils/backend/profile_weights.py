import operator
from sklearn import datasets, linear_model

def initialize_weights(jc,weights):
    proflst= jc.profile_values.keys()
    for prof in proflst:
        collst=jc.profile_values[prof].keys()
        for col in collst:
            weights [(prof,col)]=1
    return weights


def sort_candidates_irrespective (new_col_lst,candidates,weights,queried_cand):
    score_dic = {}
    for c in candidates:
        jc=new_col_lst[c]
        #print (c,jc.profile_values)
        sc = 0
        for w in weights.keys():
            sc+= abs(float(jc.profile_values[w[0]][w[1]]) * weights[w])
        score_dic[c] = sc
    sorted_sc = sorted(score_dic.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_sc

def sort_candidates (new_col_lst,candidates,weights,queried_cand):
    score_dic = {}
    for c in candidates:
        if c in queried_cand.keys():
            score_dic[c]=queried_cand[c]
            #print ("queried score is ",score_dic[c])
            continue
        jc=new_col_lst[c]
        #print (c,jc.profile_values)
        sc = 0
        for w in weights.keys():
            sc+= abs(float(jc.profile_values[w[0]][w[1]]) * weights[w])
        score_dic[c] = sc
    sorted_sc = sorted(score_dic.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_sc

def get_features(profile_dic,prof_lst,col_lst,uninfo):
    lst=[]
    for prof in prof_lst:
        dic=profile_dic[prof]
        if prof=='nan':
            lst.append(dic['all'])
        elif prof=='uninfo':
            i=0
            while i<uninfo:
                lst.append(dic[str(i)])
                i+=1
        else:
            for c in col_lst:
                lst.append(dic[c])

    return lst

def get_weights(new_col_lst,base_df,queried_cand,weights,uninfo):
    X=[]
    Y=[]
    prof_lst=new_col_lst[0].profile_values.keys()
    collst=list(base_df.columns)
    for jc_id in queried_cand.keys():
        X.append(get_features(new_col_lst[jc_id].profile_values,prof_lst,collst,uninfo))
        Y.append(queried_cand[jc_id])
    #print (X,Y)


    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    wt_iter=0
    coef_lst=regr.coef_
    for prof in prof_lst:
        if prof=='nan':
            print (prof,"all",coef_lst[wt_iter])
            weights[(prof,'all')]=coef_lst[wt_iter]
            wt_iter+=1
            continue
        if prof=='uninfo':
            i=0
            while i<uninfo:
                weights[(prof,str(i))]=coef_lst[wt_iter]
                i+=1 
                wt_iter+=1
            continue
        for c in collst:
            print (prof,c,coef_lst[wt_iter],weights[(prof,c)])
            weights[(prof,c)]=coef_lst[wt_iter]
            wt_iter+=1
    return weights

