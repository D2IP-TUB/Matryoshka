import random
def identify_group_query(new_col_lst,clusters,grp_size,likelihood_num,likelihood_den,queried_cand):
    print('Clusters:',clusters)
    likelihood=[]
    i=0
    while i<len(likelihood_num):
        likelihood.append(likelihood_num[i]*1.0/likelihood_den[i])
        i+=1
    
    i=0
    tot=sum(likelihood)
    normalized=[]
    while i<len(likelihood):
        normalized.append(likelihood[i]*1.0/tot)
        i+=1
    

    group=[]
    grp_representation = ''
    while len(group)<grp_size:
        coint_toss=random.random()
        l_iter=0
        while coint_toss > normalized[l_iter]:
            coint_toss-=normalized[l_iter]
            l_iter+=1

        lst=clusters[l_iter]
        if len(lst)==0:
            continue
        clust_elem = lst[random.randint(0,len(lst)-1)]
        group.append(clust_elem)
        grp_representation+= str(clust_elem.loc)+"|"
    if grp_representation in queried_cand.keys():
        return identify_group_query(new_col_lst,clusters,grp_size,likelihood_num,likelihood_den,queried_cand)
    return (group,grp_representation)

