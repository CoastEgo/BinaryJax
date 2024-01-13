import numpy as np
from scipy.optimize import linear_sum_assignment
idx_all=np.linspace(0,4,5,dtype=int)
def fz0(z,m1,m2,s):
    return -m1/(z-s)-m2/z
def fz1(z,m1,m2,s):
    return m1/(z-s)**2+m2/z**2
def fz2(z,m1,m2,s):
    return -2*m1/(z-s)**3-2*m2/z**3
def fz3(z,m1,m2,s):
    return 6*m1/(z-s)**4+6*m2/z**4
def J(z,m1,m2,s):
    return 1-fz1(z,m1,m2,s)*np.conj(fz1(z,m1,m2,s))
def Quadrupole_test(rho,s,q,zeta,z,zG,tol):
    m1=1/(1+q)
    m2=q/(1+q)
    cQ=6;cG=2;cP=2
    ####Quadrupole test
    miu_Q=np.abs(-2*np.real(3*np.conj(fz1(z,m1,m2,s))**3*fz2(z,m1,m2,s)**2-(3-3*J(z,m1,m2,s)+J(z,m1,m2,s)**2/2)*np.abs(fz2(z,m1,m2,s))**2+J(z,m1,m2,s)*np.conj(fz1(z,m1,m2,s))**2*fz3(z,m1,m2,s))/(J(z,m1,m2,s)**5))
    miu_C=np.abs(6*np.imag(3*np.conj(fz1(z,m1,m2,s))**3*fz2(z,m1,m2,s)**2)/(J(z,m1,m2,s)**5))
    cond1=np.nansum(miu_Q+miu_C,axis=1)*cQ*(rho**2+1e-4*tol)<tol
    ####ghost image test
    zwave=np.conj(zeta[:,np.newaxis])-fz0(zG,m1,m2,s)
    J_wave=1-fz1(zG,m1,m2,s)*fz1(zwave,m1,m2,s)
    miu_G=1/2*np.abs(J(zG,m1,m2,s)*J_wave**2/(J_wave*fz2(np.conj(zG),m1,m2,s)*fz1(zG,m1,m2,s)-np.conj(J_wave)*fz2(zG,m1,m2,s)*fz1(np.conj(zG),m1,m2,s)*fz1(zwave,m1,m2,s)))
    cond2=~((cG*(rho+1e-3)>miu_G).any(axis=1))#any更加宽松，因为ghost roots应该是同时消失的，理论上是没问题的
    #####planet test
    cond3=(q>1e-2)|(np.abs(zeta+1/s)**2>cP*(rho**2+9*q/s**2))|(rho*rho*s*s<q)
    return cond1&cond2&cond3,np.nansum(np.abs(1/J(z,m1,m2,s)),axis=1)
def get_poly_coff(zeta_l,s,m2):
    zeta_conj=np.conj(zeta_l)
    c0=s**2*zeta_l*m2**2
    c1=-s*m2*(2*zeta_l+s*(-1+s*zeta_l-2*zeta_l*zeta_conj+m2))
    c2=zeta_l-s**3*zeta_l*zeta_conj+s*(-1+m2-2*zeta_conj*zeta_l*(1+m2))+s**2*(zeta_conj-2*zeta_conj*m2+zeta_l*(1+zeta_conj**2+m2))
    c3=s**3*zeta_conj+2*zeta_l*zeta_conj+s**2*(-1+2*zeta_conj*zeta_l-zeta_conj**2+m2)-s*(zeta_l+2*zeta_l*zeta_conj**2-2*zeta_conj*m2)
    c4=zeta_conj*(-1+2*s*zeta_conj+zeta_conj*zeta_l)-s*(-1+2*s*zeta_conj+zeta_conj*zeta_l+m2)
    c5=(s-zeta_conj)*zeta_conj
    coff=np.stack((c5,c4,c3,c2,c1,c0),axis=1)
    return coff
def verify(zeta_l,z_l,s,m1,m2):#verify whether the root is right
    return  np.abs(z_l-m1/(np.conj(z_l)-s)-m2/np.conj(z_l)-zeta_l)
def get_parity(z,s,m1,m2):#get the parity of roots
    de_conjzeta_z1=m1/(np.conj(z)-s)**2+m2/np.conj(z)**2
    return np.sign((1-np.abs(de_conjzeta_z1)**2))
def get_parity_error(z,s,m1,m2):
    de_conjzeta_z1=m1/(np.conj(z)-s)**2+m2/np.conj(z)**2
    return np.abs((1-np.abs(de_conjzeta_z1)**2))
def get_roots(sample_n,coff):
    roots=np.empty((sample_n,5),dtype=np.complex128)
    for k in range(sample_n):
        roots[k,:]=np.roots(coff[k,:])
    return roots
def dot_product(a,b):
    return np.real(a)*np.real(b)+np.imag(a)*np.imag(b)
## sort the image using a navie method, and don't promise the minimum distance,
## but may be sufficient for binary lens. To use Jax's shard_map api, we can't use while loop now.
# check here for more details: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
def find_nearest_sort(array1, parity1, array2, parity2):
    cost=np.abs(array2-array1[:,None])+np.abs(parity2-parity1[:,None])*5
    cost=np.where(np.isnan(cost),100,cost)
    if np.isnan(array1).sum()==2:
        idx=np.argmin(cost,axis=1)
        idx=np.where(~np.isnan(array1),idx,-1)
        diff_idx=np.setdiff1d(np.arange(array1.shape[0]),idx)
        used=0
        for i in range(array1.shape[0]):
            if np.isnan(array1[i]):
                idx[i]=diff_idx[used]
                used+=1
        return idx                
    elif (np.isnan(array1).sum()==0) & (np.isnan(array2).sum()==0):
        idx=np.argmin(cost,axis=1)
        return idx
    else:
        idx=np.argmin(cost,axis=0)
        idx=np.where(~np.isnan(array1),idx,-1)
        diff_idx=np.setdiff1d(np.arange(array1.shape[0]),idx)
        used=0
        for i in range(array1.shape[0]):
            if np.isnan(array2[i]):
                idx[i]=diff_idx[used]
                used+=1
        ## rearrange the idx
        row_resort=np.argsort(idx)
        col_idx=np.arange(array1.shape[0])
        return col_idx[row_resort]
def find_nearest(array1, parity1, array2, parity2):#线性分配问题
    cost=np.abs(array2-array1[:,None])+np.abs(parity2-parity1[:,None])*5#系数可以指定防止出现错误，系数越大鲁棒性越好，但是速度会变慢些
    cost=np.where(np.isnan(cost),100,cost)
    row_ind, col_idx = linear_sum_assignment(cost)
    #col_idx2=find_nearest_sort(array1, parity1, array2, parity2)
    return col_idx
def search(m_map,n_map,roots,parity,fir_val,Is_create):#图像匹配算法
    m=m_map[-1]
    n=n_map[-1]
    sample_n=np.shape(roots)[0]
    if (m>=sample_n)|(m<0):#循环列表
        m=m%(sample_n)
    m_next=(m-int(parity[m,n]))%sample_n
    nextisnan=(np.isnan(roots[m_next,n]))
    if len(m_map)!=1:
    #如果下一个已经闭合
        if np.isclose(roots[m,n],fir_val,rtol=1e-7):
            parity[m,n]=0
            roots[m,n]=np.nan
            return m_map,n_map,roots,parity  
    if ((m==sample_n-1)&(m_next==0))|((m==0)&(m_next==sample_n-1)):#处理首尾相连的问题(0与2pi)
        m_next=m_next%sample_n
        try:
            transit=np.where(np.isclose(roots[m_next,:],roots[m,n],rtol=1e-6))[0][0]
            roots[m,n]=np.nan
            parity[m,n]=0
            m_map+=[m_next];n_map+=[transit]
            m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
            return m_map_res,n_map_res,temp_roots,temp_parity
        except IndexError:
            parity[m,n]=0
            roots[m,n]=np.nan
            return m_map,n_map,roots,parity    
    #如果下一个不是nan，继续往下走
    if (~nextisnan):
        par=-int(parity[m,n])
        try:
            critial_m=np.where(np.isnan(roots[m::par,n]))[0][0]-1
            real_m=m+par*critial_m
        except IndexError:
            real_m=int(-1/2*(par+1))%sample_n
        roots[m:real_m:par,n]=np.nan
        parity[m:real_m:par,n]=0
        m_map+=[i for i in range(m+par,real_m+par,par)]
        n_map+=[n]*(abs(real_m-m))
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
        return m_map_res,n_map_res,temp_roots,temp_parity
    #如果下一个是nan并且当前还有别的列，就转换列,换列不能发生在最后一行因为 2*pi=0此时根个数相同不存在create destruct
    elif (nextisnan & (len(np.where(~np.isnan(roots[m,:]))[0])>1) & (Is_create[m,:]!=0).any()):
        #parity更换符号的位置，并且该位置的下一个位置为nan（撞墙了）
        transit=np.where((parity[m,:]==-parity[m,n])&(np.isnan(roots[m_next,:])))[0][0]
        roots[m,n]=np.nan
        parity[m,n]=0
        m_map+=[m];n_map+=[transit]
        #将遍历过的位置设置为nan，将parity设置为0
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
        return m_map_res,n_map_res,temp_roots,temp_parity
    #如果当前有一个不是nan，并且下一行也只有一个parity相同的,并且不是最后一行
    elif (len(np.where(parity[m_next,:]==parity[m,n])[0])==1):
        roots[m,n]=np.nan
        parity[m,n]=0
        m-=int(parity[m,n])
        transit=np.where((~np.isnan(roots[m,:])))[0][0]
        m_map+=[m];n_map+=[transit]
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
        return m_map_res,n_map_res,temp_roots,temp_parity
    else:
        parity[m,n]=0
        roots[m,n]=np.nan
        return m_map,n_map,roots,parity   
def search_first_postion(temp_roots,temp_parity):#搜索图像匹配开始的索引
    roots_now=temp_roots[0,:][~np.isnan(temp_roots[0,:])]
    change_sum=np.sum(temp_parity,axis=1)
    if (np.isin(1,change_sum).any()):#只处理parity 是 -1 1 -1的情况，否则转换为 -1 1 -1的情况
        temp_parity*=-1
    if np.shape(roots_now)[0]!=0:#如果第一行不是全部都为nan
        temp_cond=np.where((~np.isnan(temp_roots.real[0,:])) & (temp_parity[0,:]==-1))[0]#第一行存在的parity=sum(parity)的根
        initk=0
        if np.shape(temp_roots)[1]==2:#如果有两列
            try:
                initm=np.where(~np.isin(np.round(temp_roots[0,:],6),np.round(temp_roots[-1,:],6)))[0][0]#第一行不在最后一行的值
            except IndexError:
                initm=np.where(temp_parity[0,:]==-1)[0][0]
            if temp_parity[initk,initm]==1:
                temp_parity*=-1
        else:#第一行不是nan 并且不是两列
            Nan_idx=np.where(np.isnan(temp_roots[:,temp_cond]).any(axis=1))[0][0]
            initm=temp_cond[np.where(np.isnan(temp_roots[Nan_idx,temp_cond]))[0][0]]
    else:#如果有两列待链接,并且第一行全部为nan
        roots_last=temp_roots[-1,:][~np.isnan(temp_roots[-1,:])]#最后一行不是nan的地方
        if np.shape(roots_last)[0]==0:#如果最后一行都是nan
            initm=np.where((temp_parity==-1).any(axis=0))[0][0]
            initk=np.where(~np.isnan(temp_roots[:,initm]))[0][0]
        else:
            initk=-1
            initm=np.where(temp_parity[-1,:]==-1)[0][0]
            temp_parity*=-1
    return initk,initm,temp_parity