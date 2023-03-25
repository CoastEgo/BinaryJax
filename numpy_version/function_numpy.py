import numpy as np
def search(m_map,n_map,roots,parity,fir_val):
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
            m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val)
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
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val)
        return m_map_res,n_map_res,temp_roots,temp_parity
    #如果下一个是nan并且当前还有别的列，就转换列,换列不能发生在最后一行因为 2*pi=0此时根个数相同不存在create destruct
    elif (nextisnan & (len(np.where(~np.isnan(roots[m,:]))[0])>1)) :
        #parity更换符号的位置，并且该位置的下一个位置为nan（撞墙了）
        transit=np.where((parity[m,:]==-parity[m,n])&(np.isnan(roots[m_next,:])))[0][0]
        roots[m,n]=np.nan
        parity[m,n]=0
        m_map+=[m];n_map+=[transit]
        #将遍历过的位置设置为nan，将parity设置为0
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val)
        return m_map_res,n_map_res,temp_roots,temp_parity
    #如果当前有一个不是nan，并且下一行也只有一个parity相同的,并且不是最后一行
    elif (len(np.where(parity[m_next,:]==parity[m,n])[0])==1):
        roots[m,n]=np.nan
        parity[m,n]=0
        m-=int(parity[m,n])
        transit=np.where((~np.isnan(roots[m,:])))[0][0]
        m_map+=[m];n_map+=[transit]
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val)
        return m_map_res,n_map_res,temp_roots,temp_parity
    else:
        parity[m,n]=0
        roots[m,n]=np.nan
        return m_map,n_map,roots,parity   
def search_first_postion(temp_roots,temp_parity):
    sample_n=np.shape(temp_roots)[0]
    roots_now=temp_roots[0,:][~np.isnan(temp_roots[0,:])]
    change_sum=np.sum(temp_parity,axis=1)
    if (np.isin(1,change_sum).any()):#只处理parity 是 -1 1 -1的情况，否则转换为 -1 1 -1的情况
        temp_parity*=-1
    if np.shape(roots_now)[0]!=0:#如果第一行不是全部都为nan
        temp_cond=np.where((~np.isnan(temp_roots.real[0,:])) & (temp_parity[0,:]==-1))[0]#第一行存在的parity=sum(parity)的根
        initk=0
        if np.shape(temp_roots)[1]==2:
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
        roots_last=temp_roots[-1,:][~np.isnan(temp_roots[-1,:])]
        if np.shape(roots_last)[0]==0:
            initm=np.where((temp_parity==-1).any(axis=0))[0][0]
            initk=np.where(~np.isnan(temp_roots[:,initm]))[0][0]
        else:
            initk=-1
            initm=np.where(temp_parity[-1,:]==-1)[0][0]
            temp_parity*=-1
    return initk,initm,temp_parity