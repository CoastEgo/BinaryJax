import numpy as np
def search(m,n,roots,parity,curve,counts,theta,theta_map):
    if m==np.shape(roots)[0]-1:
        m=m-np.shape(roots)[0]
    elif m==-np.shape(roots)[0]:
        m=m+np.shape(roots)[0]
    if counts==1:
        curve+=[roots[m,n]]
        theta_map+=[theta[m]]
        return np.array(curve),np.array(theta_map)
    #如果下一个是nan并且当前还有别的列，就转换列
    elif ((np.isnan(roots[m-int(parity[m,n]),n])) & (len(np.where(~np.isnan(roots[m,:]))[0])>1)):
        if (~np.isnan(roots[m,:])).all():
            transit=np.where((parity[m,:]==-parity[m,n])&(np.isnan(roots[m-int(parity[m,n]),:])))[0][0]
        else:
            transit=np.where((parity[m,:]==-parity[m,n]))[0][0]
        curve+=[roots[m,n]]
        theta_map+=[theta[m]]
        roots[m,n]=np.nan
        parity[m,n]=2
        n=transit
        counts-=1
        result,theta_map_result=search(m,n,roots,parity,curve,counts,theta,theta_map)
        return result,theta_map_result
    elif (~np.isnan(roots[m-int(parity[m,n]),n])): #如果下一个不是nan，继续往下走
        curve+=[roots[m,n]]
        theta_map+=[theta[m]]
        roots[m,n]=np.nan
        counts-=1
        par=parity[m,n]
        parity[m,n]=2
        m-=int(par)
        result,theta_map_result=search(m,n,roots,parity,curve,counts,theta,theta_map)
        return result,theta_map_result
    #如果当前有一个不是nan，下一个个是nan，并且下一行也只有一个不是nan
    elif (len(np.where(~np.isnan(roots[m,:]))[0])==1)&(len(np.where(~np.isnan(roots[m-int(parity[m,n]),:]))[0])==1)&(np.isnan(roots[m-int(parity[m,n]),n])):
        curve+=[roots[m,n]]
        theta_map+=[theta[m]]
        roots[m,n]=np.nan
        par=parity[m,n]
        parity[m,n]=2
        m-=int(par)
        transit=np.where((~np.isnan(roots[m,:])) & (parity[m,:]==-1))[0][0]
        n=transit
        counts-=1
        result,theta_map_result=search(m,n,roots,parity,curve,counts,theta,theta_map)
        return result,theta_map_result
def search_first_postion(temp_roots,temp_parity,temp_idx):
    sample_n=np.shape(temp_roots)[0]
    if len(temp_idx)==3:#如果有三列待链接
        temp_cond=np.where((~np.isnan(temp_roots.real[0,:])) & (temp_parity[0,:]==-1))[0]#第一行存在的parity=-1的根
        initk=0
        if len(temp_cond)!=1:#如果有两个这样的根
            for k in range(sample_n):
                if (np.isnan(temp_roots[k,temp_cond[0]])) & (~np.isnan(temp_roots[k,temp_cond[1]])):
                    break#如果第一列先变为nan
                if (~np.isnan(temp_roots[k,temp_cond[0]])) & (np.isnan(temp_roots[k,temp_cond[1]])):
                    temp_cond[0]=temp_cond[1]#如果第二列先变为nan
                    break
    else:#如果有两列待链接
        for i in range(sample_n):
            temp_cond=np.where(temp_parity[i,:]==-1)[0]
            if len(temp_cond)!=0:
                initk=np.where(~np.isnan(temp_roots[:,temp_cond[0]]))[0][0]
                break
    counts=np.count_nonzero(~np.isnan(temp_roots.real))
    return initk,temp_cond[0],counts