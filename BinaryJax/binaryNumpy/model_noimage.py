import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .basic_function import *
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
class model():#initialize parameter
    def __init__(self,par):
        self.solve_time=0
        self.t_0=par['t_0']
        self.u_0=par['u_0']
        self.t_E=par['t_E']
        self.rho=par['rho']
        self.q=par['q']
        self.s=par['s']
        self.alpha_rad=par['alpha_deg']*2*np.pi/360
        self.times=(par['times']-self.t_0)/self.t_E
        self.trajectory_n=len(self.times)
        self.m1=1/(1+self.q)
        self.m2=self.q/(1+self.q)
        self.trajectory_l=self.get_trajectory_l()
    def to_centroid(self,x):#change coordinate system to cetorid
        delta_x=self.s/(1+self.q)
        return -(np.conj(x)-delta_x)
    def to_lowmass(self,x):#change coordinaate system to lowmass
        delta_x=self.s/(1+self.q)
        return -np.conj(x)+delta_x
    def get_trajectory_l(self):
        alpha=self.alpha_rad
        b=self.u_0
        trajectory_l=self.to_lowmass(self.times*np.cos(alpha)-b*np.sin(alpha)+1j*(b*np.cos(alpha)+self.times*np.sin(alpha)))
        return trajectory_l
    def get_zeta_l(self,trajectory_centroid_l,theta):#获得等高线采样的zeta
        rho=self.rho
        rel_centroid=rho*np.cos(theta)+1j*rho*np.sin(theta)
        zeta_l=trajectory_centroid_l+rel_centroid
        return zeta_l
    def get_magnifaction2(self,tol,retol=0):
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        zeta_l=trajectory_l
        coff=get_poly_coff(zeta_l,self.s,self.m2)
        z_l=get_roots(trajectory_n,coff)
        error=verify(zeta_l[:,np.newaxis],z_l,self.s,self.m1,self.m2)
        cond=error<1e-6
        '''index=np.where((cond.sum(axis=1)!=3) & (cond.sum(axis=1)!=5))[0]
        if index.size!=0:
            sortidx=np.argsort(error[index],axis=1)
            cond[index]=False
            cond[index,sortidx[0:3]]=True'''
        z=np.where(cond,z_l,np.nan)
        zG=np.where(cond,np.nan,z_l)
        cond,mag=Quadrupole_test(self.rho,self.s,self.q,zeta_l,z,zG,tol)
        idx=np.where(~cond)[0]
        for i in idx:
            temp_mag=self.contour_integrate(trajectory_l[i],tol,i,retol)
            mag[i]=temp_mag
        return mag
    def contour_integrate(self,trajectory_l,epsilon,i,epsilon_rel=0):
        #sample_n=3;theta_init=np.array([0,np.pi,2*np.pi],dtype=np.float64)
        sample_n=np.int64(30);theta_init=np.linspace(0,2*np.pi,sample_n)
        error_hist=np.ones(1)
        mag=1
        outloop=False
        mag0=0
        while ((error_hist/np.abs(mag)>epsilon_rel/np.sqrt(sample_n)).any() & (np.abs(mag-mag0)>1/2*epsilon)):#相对误差
        #while ((np.sum(error_hist)/np.abs(mag)>epsilon_rel) & (np.abs(mag-mag0)>1/2*epsilon)):#相对误差
            mini_interval=np.min(np.abs(np.diff(theta_init)))
            if mini_interval<1e-14:
                print(f'Warnning! idx{i} theta sampling may lower than float64 limit')
                break
            if outloop:
                print(f'idx{i} did not reach the accuracy goal due to the ambigious of roots and parity')
                break
            mag0=mag
            mag=1
            if np.shape(error_hist)[0]==1:#第一次采样
                error_hist=np.zeros_like(theta_init)
                zeta_l=self.get_zeta_l(trajectory_l,theta_init)
                coff=get_poly_coff(zeta_l,self.s,self.m2)
                solution=Solution(self.q,self.s,zeta_l,coff,theta_init)
            else:#自适应采点插入theta
                idx=np.where(error_hist/np.abs(mag0)>epsilon_rel/np.sqrt(sample_n))[0]
                #idx=np.where(error_hist>np.median(error_hist))[0]
                #idx=np.array([np.argmax(error_hist)])
                add_max=5
                add_number=np.ceil((error_hist[idx]/np.abs(mag0)*np.sqrt(sample_n)/epsilon_rel)**0.2).astype(int)+1#至少要插入一个点，不包括相同的第一个
                add_number[add_number>add_max]=add_max
                add_theta=[np.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False,dtype=np.float64)[1:] for i in range(np.shape(idx)[0])]
                idx = np.repeat(idx, add_number-1) # create an index array with the same length as add_item
                add_theta = np.concatenate(add_theta) # concatenate the list of arrays into a 1-D array
                add_zeta_l=self.get_zeta_l(trajectory_l,add_theta)
                add_coff=get_poly_coff(add_zeta_l,self.s,self.m2)
                solution.add_points(idx,add_zeta_l,add_coff,add_theta)
                sample_n=solution.sample_n
                theta_init=solution.theta
                outloop=solution.outofloop
            arc=solution.roots
            arc_parity=solution.parity
            mag=1/2*np.nansum(np.nansum((arc.imag[0:-1]+arc.imag[1:])*(arc.real[0:-1]-arc.real[1:])*arc_parity[0:-1],axis=0))
            Error=Error_estimator(solution,self.rho)
            error_hist,magc,parab=Error.error_sum()
            mag+=magc
            mag+=parab
            mag=mag/(np.pi*self.rho**2)
            error_hist+=solution.buried_error
        #print(sample_n)
        return mag
class Solution(object):
    def __init__(self,q,s,zeta_l,coff,theta):
        self.theta=theta
        self.q=q;self.s=s;self.m1=1/(1+q);self.m2=q/(1+q)
        self.zeta_l=zeta_l
        self.coff=coff
        self.sample_n=np.shape(zeta_l)[0]
        roots,parity,self.ghost_roots_dis=self.get_real_roots(coff,zeta_l)#cond非nan为true
        self.buried_error=self.get_buried_error()
        self.roots,self.parity=self.get_sorted_roots(self.sample_n,roots,parity)
        self.sort_flag=np.repeat(np.array(True),self.sample_n)
        self.find_create_points()
        self.outofloop=False
    def add_points(self,idx,add_zeta,add_coff,add_theta):
        self.idx=idx
        self.sample_n+=np.size(idx)
        self.theta=np.insert(self.theta,idx,add_theta)
        self.zeta_l=np.insert(self.zeta_l,idx,add_zeta)
        self.coff=np.insert(self.coff,idx,add_coff,axis=0)
        add_roots,add_parity,add_ghost_roots=self.get_real_roots(add_coff,add_zeta)
        idx=self.idx
        self.ghost_roots_dis=np.insert(self.ghost_roots_dis,idx,add_ghost_roots,axis=0)
        self.buried_error=self.get_buried_error()
        self.sort_flag=np.insert(self.sort_flag,idx,np.array(False))
        self.roots,self.parity=self.add_sorted_roots(roots=np.insert(self.roots,idx,add_roots,axis=0),parity=np.insert(self.parity,idx,add_parity,axis=0))
        self.find_create_points()
    def get_buried_error(self):
        error_buried=np.zeros(self.sample_n)
        ghost_roots_dis=self.ghost_roots_dis.ravel()
        idx1=np.where(ghost_roots_dis[0:-2]>2*ghost_roots_dis[1:-1])[0]+1
        idx1=idx1[~np.isnan(ghost_roots_dis[idx1+1])]
        error_buried[idx1+1]+=(ghost_roots_dis[idx1]-ghost_roots_dis[idx1-1])**2
        error_buried[idx1]+=(ghost_roots_dis[idx1]-ghost_roots_dis[idx1-1])**2
        idx1=np.where(2*ghost_roots_dis[1:-1]<ghost_roots_dis[2:])[0]+1
        idx1=idx1[~np.isnan(ghost_roots_dis[idx1-1])]
        error_buried[idx1]+=(ghost_roots_dis[idx1+1]-ghost_roots_dis[idx1])**2
        error_buried[idx1+1]+=(ghost_roots_dis[idx1+1]-ghost_roots_dis[idx1])**2
        return error_buried
    def add_sorted_roots(self,roots,parity):
        sort_flag=self.sort_flag
        flase_i=np.where(~sort_flag)[0]
        for i in flase_i:
            sort_indices=find_nearest(roots[i-1],parity[i-1],roots[i],parity[i])
            roots[i]=roots[i][sort_indices]
            parity[i]=parity[i][sort_indices]
            if sort_flag[i+1]:
                sort_indices=find_nearest(roots[i],parity[i],roots[i+1],parity[i+1])
                roots[i+1:]=roots[i+1:,sort_indices]
                parity[i+1:]=parity[i+1:,sort_indices]
        self.sort_flag[:]=True
        return roots,parity
    def get_real_roots(self,coff,zeta_l):
        sample_n=np.shape(zeta_l)[0]
        roots=get_roots(sample_n,coff)
        parity=get_parity(roots,self.s,self.m1,self.m2)
        error=verify(zeta_l[:,None],roots,self.s,self.m1,self.m2)
        cond=error>1e-6
        nan_num=cond.sum(axis=1)
        ####计算verify,如果parity出现错误或者nan个数错误，则重新规定error最大的为nan
        idx_verify_wrong=np.where(((nan_num!=0)&(nan_num!=2)))[0]#verify出现错误的根的索引
        if np.size(idx_verify_wrong)!=0:
            sorted=np.argsort(error[idx_verify_wrong],axis=1)[:,-2:]
            cond[idx_verify_wrong]=False
            cond[idx_verify_wrong,sorted[:,0]]=True
            cond[idx_verify_wrong,sorted[:,1]]=True
        ####根的处理
        nan_num=cond.sum(axis=1)
        parity_sum=np.nansum(parity[~cond])
        real_roots=np.where(cond,np.nan+np.nan*1j,roots)
        real_parity=np.where(cond,np.nan,parity)
        parity_sum=np.nansum(real_parity,axis=1)
        idx_parity_wrong=np.where(parity_sum!=-1)[0]#parity计算出现错误的根的索引
        ##对于parity计算错误的点，分为fifth principal left center right，其中left center right 的parity为-1，1，-1
        if np.size(idx_parity_wrong)!=0:
            for i in idx_parity_wrong:
                temp=real_roots[i]
                if nan_num[i]==0:
                    prin_root=temp[np.sign(temp.imag)==np.sign(zeta_l.imag[i])][0][np.newaxis]
                    prin_root=np.concatenate([prin_root,temp[np.argmax(get_parity_error(temp,self.s,self.m1,self.m2))][np.newaxis]])
                    if (np.shape(real_parity)[0]==3)|(np.abs(zeta_l.imag[i])>1e-5):#初始三个点必须全部正确
                        real_parity[i][temp==prin_root[0]]=1;real_parity[i][temp==prin_root[1]]=-1#是否对主图像与负图像进行parity的赋值
                    other=np.setdiff1d(temp,prin_root)
                    x_sort=np.argsort(other.real)
                    real_parity[i][(temp==other[x_sort[0]])|(temp==other[x_sort[-1]])]=-1
                    real_parity[i][(temp==other[x_sort[1]])]=1
                else:##对于三个根怎么判断其parity更加合理
                    if (np.abs(zeta_l.imag[i])>1e-5)|((np.shape(real_parity)[0]==3)):#初始三个点必须全部正确
                        real_parity[i,~cond[i]]=-1
                        real_parity[i,np.sign(temp.imag)==np.sign(zeta_l.imag[i])]=1##通过主图像判断，与zeta位于y轴同一侧的为1'''
        parity_sum=np.nansum(real_parity,axis=1)
        if (parity_sum!=-1).any():
            idx=np.where(parity_sum!=-1)[0]
            self.sample_n-=np.size(idx)
            self.coff=np.delete(self.coff,idx,axis=0)
            self.zeta_l=np.delete(self.zeta_l,idx)
            self.theta=np.delete(self.theta,idx)
            real_parity=np.delete(real_parity,idx,axis=0)
            cond=np.delete(cond,idx,axis=0)
            roots=np.delete(roots,idx,axis=0)
            real_roots=np.delete(real_roots,idx,axis=0)
            self.idx=np.delete(self.idx,idx,axis=0)
            self.outofloop=True
        ###计算得到最终的
        ghost_roots=np.where(cond,roots,np.inf)
        ghost_roots=np.sort(ghost_roots,axis=1)[:,0:2]
        ghost_roots[ghost_roots==np.inf]=np.nan
        ghost_roots_dis=np.abs(np.diff(ghost_roots,axis=1))
        return real_roots,real_parity,ghost_roots_dis
    def find_create_points(self):
        cond=np.isnan(self.roots)
        Is_create=np.zeros_like(self.roots,dtype=int)
        idx_x,idx_y=np.where(np.diff(cond,axis=0))
        idx_x+=1
        for x,y in zip(idx_x,idx_y):
            if ~cond[x,y]:#如果这个不是nan
                Is_create[x,y]=1#这个是destruction
            elif (cond[x,y])&(Is_create[x-1,y]!=1):#如果这个不是
                Is_create[x-1,y]=-1
            else:
                Is_create[x-1,y]=10
        self.Is_create=Is_create
    def get_sorted_roots(self,sample_n,roots,parity):#非nan为true
        for k in range(1,sample_n):
            sort_indices=find_nearest(roots[k-1,:],parity[k-1,:],roots[k,:],parity[k,:])
            roots[k,:]=roots[k,:][sort_indices]
            parity[k,:]=parity[k,:][sort_indices]
        return roots,parity
    def root_polish(self,coff,roots,epsilon):
        p=np.poly1d(coff)
        derp=np.polyder(coff)
        for i in range(np.shape(roots)[0]):
            x_0=roots[i]
            temp=p(x_0)
            while p(x_0)>epsilon:
                if derp(x_0)<1e-14:
                    break
                x=x_0-p(x_0)/derp(x_0)
                x_0=x
            roots[i]=x_0
        return roots
    def roots_print(self):
        np.savetxt('result/roots.txt',self.roots,delimiter=',',fmt='%.4f')
        np.savetxt('result/parity.txt',self.parity,delimiter=',',fmt='%.0f')
        #np.savetxt('result/create.txt',self.Is_create,delimiter=',',fmt='%.0f')
        #np.savetxt('result/roots_diff.txt',np.abs(self.roots[1:]-self.roots[0:-1]),delimiter=',')
        #np.savetxt('result/roots_verify.txt',verify(np.array([self.zeta_l,self.zeta_l,self.zeta_l,self.zeta_l,self.zeta_l]).T,self.roots),delimiter=',')
class Error_estimator(object):
    def __init__(self,arc,rho):
        q=arc.q;s=arc.s;self.rho=rho;self.paritypar=arc.parity
        zeta_l=arc.roots
        self.parity=arc.parity
        self.zeta_l=zeta_l
        theta=arc.theta
        self.theta=arc.theta;self.Is_create=arc.Is_create
        self.delta_theta=np.diff(self.theta)
        zeta_conj=np.conj(self.zeta_l)
        parZetaConZ=1/(1+q)*(1/(zeta_conj-s)**2+q/zeta_conj**2)
        par2ConZetaZ=-2/(1+q)*(1/(zeta_l-s)**3+q/(zeta_l)**3)
        de_zeta=1j*self.rho*np.exp(1j*theta)
        detJ=1-np.abs(parZetaConZ)**2
        de_z=(de_zeta[:,np.newaxis]-parZetaConZ*np.conj(de_zeta[:,np.newaxis]))/detJ
        deXProde2X=(self.rho**2+np.imag(de_z**2*de_zeta[:,np.newaxis]*par2ConZetaZ))/detJ
        self.product=deXProde2X
        self.de_z=de_z
    def error_ordinary(self):
        deXProde2X=self.product
        delta_theta=self.delta_theta
        zeta_l=self.zeta_l
        e1=np.nansum(np.abs(1/48*np.abs(np.abs(deXProde2X[0:-1]-np.abs(deXProde2X[1:])))*delta_theta[:,np.newaxis]**3),axis=1)
        dAp_1=1/24*((deXProde2X[0:-1]+deXProde2X[1:]))*delta_theta[:,np.newaxis]
        dAp=dAp_1*delta_theta[:,np.newaxis]**2*self.parity[0:-1]
        delta_theta_wave=np.abs(zeta_l[0:-1]-zeta_l[1:])**2/np.abs(dot_product(self.de_z[0:-1],self.de_z[1:]))
        e2=np.nansum(3/2*np.abs(dAp_1*(delta_theta_wave-delta_theta[:,np.newaxis]**2)),axis=1)
        e3=np.nansum(1/10*np.abs(dAp)*delta_theta[:,np.newaxis]**2,axis=1)
        e_tot=(e1+e2+e3)/(np.pi*self.rho**2)
        return e_tot,np.nansum(dAp)#抛物线近似的补偿项
    def error_critial(self,i,pos_idx,neg_idx,Is_create):
        zeta_l=self.zeta_l
        de_z=self.de_z
        deXProde2X=self.product
        zeta_pos=zeta_l[i,pos_idx]
        zeta_neg=zeta_l[i,neg_idx]
        theta_wave=np.abs(zeta_pos-zeta_neg)/np.sqrt(np.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx])))
        ce1=1/48*np.abs(deXProde2X[i,pos_idx]+deXProde2X[i,neg_idx])*theta_wave**3
        ce2=3/2*np.abs(dot_product(zeta_pos-zeta_neg,de_z[i,pos_idx]-de_z[i,neg_idx])-Is_create*2*np.abs(zeta_pos-zeta_neg)*np.sqrt(np.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx]))))*theta_wave
        dAcP=self.parity[i,pos_idx]*1/24*(deXProde2X[i,pos_idx]-deXProde2X[i,neg_idx])*theta_wave**3
        ce3=1/10*np.abs(dAcP)*theta_wave**2
        ce_tot=(ce1+ce2+ce3)/(np.pi*self.rho**2)
        return ce_tot,np.sum(dAcP)#critial 附近的抛物线近似'''
    def error_sum(self):
        error_hist=np.zeros_like(self.theta)
        mag=0
        e_ord,parab=self.error_ordinary()
        error_hist[1:]=e_ord
        if  (self.Is_create!=0).any():
            critial_idx_row=np.where((self.Is_create!=0).any(axis=1))[0]
            for i in critial_idx_row:
                if np.abs(self.Is_create[i].sum())/2==1:
                    create=self.Is_create[i].sum()/2
                    idx1=np.where((self.Is_create[i]==create)&(self.parity[i]==-1*create))[0]
                    idx2=np.where((self.Is_create[i]==create)&(self.parity[i]==1*create))[0]
                    e_crit,dacp=self.error_critial(i,idx1,idx2,create)
                    error_hist[int(i-(create-1)/2)]+=e_crit
                    mag+=1/2*(self.zeta_l[i,idx1].imag+self.zeta_l[i,idx2].imag)*(self.zeta_l[i,idx1].real-self.zeta_l[i,idx2].real)
                    parab+=dacp
                else:
                    create=1
                    idx1=np.where(((self.Is_create[i]==1)|(self.Is_create[i]==10))&(self.parity[i]==-1*create))[0]
                    idx2=np.where(((self.Is_create[i]==1)|(self.Is_create[i]==10))&(self.parity[i]==1*create))[0]
                    mag+=1/2*(self.zeta_l[i,idx1].imag+self.zeta_l[i,idx2].imag)*(self.zeta_l[i,idx1].real-self.zeta_l[i,idx2].real)
                    e_crit,dacp=self.error_critial(i,idx1,idx2,create)
                    error_hist[int(i-(create-1)/2)]+=e_crit
                    parab+=dacp
                    create=-1
                    idx1=np.where(((self.Is_create[i]==-1)|(self.Is_create[i]==10))&(self.parity[i]==-1*create))[0]
                    idx2=np.where(((self.Is_create[i]==-1)|(self.Is_create[i]==10))&(self.parity[i]==1*create))[0]
                    mag+=1/2*(self.zeta_l[i,idx1].imag+self.zeta_l[i,idx2].imag)*(self.zeta_l[i,idx1].real-self.zeta_l[i,idx2].real)
                    e_crit,dacp=self.error_critial(i,idx1,idx2,create)
                    error_hist[int(i-(create-1)/2)]+=e_crit
                    parab+=dacp
        return error_hist,mag,parab



