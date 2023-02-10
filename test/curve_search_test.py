import numpy as np
temp_roots=np.array([[np.nan,np.nan],[np.nan,np.nan],[8,9]])
print(temp_roots)
temp_parity=np.array([[-1,1]*7],dtype=int).reshape(7,2)
if len(temp_roots[0,:])==3:
    temp_cond=np.where((~np.isnan(temp_roots.real[0,:])) & (temp_parity[0,:]==-1))[0]
    initk=0
    if len(temp_cond)!=1:
        for k in range(4):
            if (np.isnan(temp_roots[k,temp_cond[0]])) & (~np.isnan(temp_roots[k,temp_cond[1]])):
                break
            if (~np.isnan(temp_roots[k,temp_cond[0]])) & (np.isnan(temp_roots[k,temp_cond[1]])):
                temp_cond[0]=temp_cond[1]
                break
else:
    temp_cond=np.where(temp_parity[0,:]==-1)[0]
    initk=np.where(~np.isnan(temp_roots[:,temp_cond[0]]))[0][0]
curve=[]
counts=np.count_nonzero(~np.isnan(temp_roots))
def search(m,n,roots,parity,curve,counts):
    if m==np.shape(roots)[0]-1:
        m=m-np.shape(roots)[0]
    elif m==-np.shape(roots)[0]:
        m=m+np.shape(roots)[0]
    if counts==1:
        curve+=[roots[m,n]]
        return np.array(curve)
    #如果下一个是nan并且当前还有别的列，就转换列
    elif ((np.isnan(roots[m-parity[m,n],n])) & (len(np.where(~np.isnan(roots[m,:]))[0])>1)):
        if (~np.isnan(roots[m,:])).all():
            transit=np.where((parity[m,:]==-parity[m,n])&(np.isnan(roots[m-parity[m,n],:])))[0][0]
        else:
            transit=np.where((parity[m,:]==-parity[m,n]))[0][0]
        curve+=[roots[m,n]]
        roots[m,n]=np.nan
        parity[m,n]=2
        n=transit
        counts-=1
        result=search(m,n,roots,parity,curve,counts)
        return result
    elif (~np.isnan(roots[m-parity[m,n],n])): #如果下一个不是nan，继续往下走
        curve+=[roots[m,n]]
        roots[m,n]=np.nan
        counts-=1
        par=parity[m,n]
        parity[m,n]=2
        m-=par
        result=search(m,n,roots,parity,curve,counts)
        return result
    #如果当前有一个不是nan，下一个个是nan，并且下一行也只有一个不是nan
    elif (len(np.where(~np.isnan(roots[m,:]))[0])==1)&(len(np.where(~np.isnan(roots[m-parity[m,n],:]))[0])==1)&(np.isnan(roots[m-parity[m,n],n])):
        curve+=[roots[m,n]]
        roots[m,n]=np.nan
        par=parity[m,n]
        parity[m,n]=2
        m-=par
        transit=np.where((~np.isnan(roots[m,:])) & (parity[m,:]==-1))[0][0]
        n=transit
        counts-=1
        result=search(m,n,roots,parity,curve,counts)
        return result
cur=search(initk,temp_cond[0],temp_roots,temp_parity,[],counts)
print(cur)