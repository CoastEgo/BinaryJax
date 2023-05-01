import numpy as jnp
import jax.numpy as jnp
import jax
from jax import lax
from jax import custom_jvp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from basic_function_jax import *
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
class model():#initialize parameter
    def __init__(self,par):
        self.t_0=par['t_0']
        self.u_0=par['u_0']
        self.t_E=par['t_E']
        self.rho=par['rho']
        self.q=par['q']
        self.s=par['s']
        self.alpha_rad=par['alpha_deg']*2*jnp.pi/360
        self.times=(par['times']-self.t_0)/self.t_E
        self.trajectory_n=len(self.times)
        self.m1=1/(1+self.q)
        self.m2=self.q/(1+self.q)
        self.trajectory_l=self.get_trajectory_l()
    def to_centroid(self,x):#change coordinate system to cetorid
        delta_x=self.s/(1+self.q)
        return -(jnp.conj(x)-delta_x)
    def to_lowmass(self,x):#change coordinaate system to lowmass
        delta_x=self.s/(1+self.q)
        return -jnp.conj(x)+delta_x
    def get_trajectory_l(self):
        alpha=self.alpha_rad
        b=self.u_0
        trajectory_l=self.to_lowmass(self.times*jnp.cos(alpha)-b*jnp.sin(alpha)+1j*(b*jnp.cos(alpha)+self.times*jnp.sin(alpha)))
        return trajectory_l
    def get_zeta_l(self,trajectory_centroid_l,theta):#获得等高线采样的zeta
        rho=self.rho
        rel_centroid=rho*jnp.cos(theta)+1j*rho*jnp.sin(theta)
        zeta_l=trajectory_centroid_l+rel_centroid
        return zeta_l
    def get_magnifaction2(self,tol):
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        zeta_l=trajectory_l
        coff=get_poly_coff(zeta_l,self.s,self.m2)
        z_l=get_roots(trajectory_n,coff)
        error=verify(zeta_l[:,jnp.newaxis],z_l,self.s,self.m1,self.m2)
        cond=error<1e-6
        index=jnp.where((cond.sum(axis=1)!=3) & (cond.sum(axis=1)!=5))[0]
        if index.size!=0:
            sortidx=jnp.argsort(error[index],axis=1)
            cond.at[index].set(False)
            cond.at[index,sortidx[0:3]].set(True)
        z=jnp.where(cond,z_l,jnp.nan)
        zG=jnp.where(cond,jnp.nan,z_l)
        cond,mag=Quadrupole_test(self.rho,self.s,self.q,zeta_l,z,zG,tol)
        idx=jnp.where(~cond)[0]
        for i in idx:
            temp_mag=self.contour_integrate(trajectory_l[i],tol,i)
            mag=mag.at[i].set(temp_mag)
        return mag
    def contour_integrate(self,trajectory_l,epsilon,i,epsilon_rel=0):
        sample_n=3;theta_init=jnp.array([0,jnp.pi,2*jnp.pi])
        error_hist=jnp.ones(1)
        mag=1
        outloop=False
        while ((error_hist>epsilon/jnp.sqrt(sample_n)).any() & (error_hist/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n)).any()):#相对误差
            mini_interval=jnp.min(jnp.abs(jnp.diff(theta_init)))
            if mini_interval<1e-14:
                #print(f'Warnning! idx{i} theta sampling may lower than float64 limit')
                break 
            if outloop:
                #print(f'idx{i} did not reach the accuracy goal due to the ambigious of roots and parity')
                break
            mag=0
            if jnp.shape(error_hist)[0]==1:#第一次采样
                error_hist=jnp.zeros_like(theta_init)
                zeta_l=self.get_zeta_l(trajectory_l,theta_init)
                coff=get_poly_coff(zeta_l,self.s,self.m2)
                solution=Solution(self.q,self.s,zeta_l,coff,theta_init)
            else:#自适应采点插入theta
                idx=jnp.where(error_hist>epsilon/jnp.sqrt(sample_n))[0]#不满足要求的点
                add_number=jnp.ceil((error_hist[idx]/epsilon*jnp.sqrt(sample_n))**0.2).astype(int)+1#至少要插入一个点，不包括相同的第一个
                add_theta=[jnp.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False)[1:] for i in range(jnp.shape(idx)[0])]
                idx = jnp.repeat(idx, add_number-1) # create an index array with the same length as add_item
                add_theta = jnp.concatenate(add_theta) # concatenate the list of arrays into a 1-D array
                add_zeta_l=self.get_zeta_l(trajectory_l,add_theta)
                add_coff=get_poly_coff(add_zeta_l,self.s,self.m2)
                solution.add_points(idx,add_zeta_l,add_coff,add_theta)
                sample_n=solution.sample_n
                theta_init=solution.theta
                error_hist=jnp.zeros_like(theta_init)
                outloop=solution.outofloop
            arc=solution.roots
            arc_parity=solution.parity
            mag=1/2*jnp.nansum(jnp.nansum((arc.imag[0:-1]+arc.imag[1:])*(arc.real[0:-1]-arc.real[1:])*arc_parity[0:-1],axis=0))
            Error=Error_estimator(solution,self.rho)
            error_hist,magc,parab=Error.error_sum()
            mag+=magc
            mag+=parab
            mag=mag/(jnp.pi*self.rho**2)
            error_hist+=solution.buried_error
        return mag
class Solution(object):
    def __init__(self,q,s,zeta_l,coff,theta):
        self.theta=theta
        self.q=q;self.s=s;self.m1=1/(1+q);self.m2=q/(1+q)
        self.zeta_l=zeta_l
        self.coff=coff
        self.sample_n=jnp.shape(zeta_l)[0]
        roots,parity,self.ghost_roots_dis=self.get_real_roots(coff,zeta_l)#cond非nan为true
        self.buried_error=self.get_buried_error()
        self.roots,self.parity=get_sorted_roots(jnp.arange(1,self.sample_n),roots,parity)
        self.sort_flag=jnp.repeat(jnp.array(True),self.sample_n)
        self.find_create_points()
        self.outofloop=False
    def add_points(self,idx,add_zeta,add_coff,add_theta):
        self.idx=idx
        self.sample_n+=jnp.size(idx)
        self.theta=jnp.insert(self.theta,idx,add_theta)
        self.zeta_l=jnp.insert(self.zeta_l,idx,add_zeta)
        self.coff=jnp.insert(self.coff,idx,add_coff,axis=0)
        add_roots,add_parity,add_ghost_roots=self.get_real_roots(add_coff,add_zeta)
        idx=self.idx
        self.ghost_roots_dis=jnp.insert(self.ghost_roots_dis,idx,add_ghost_roots,axis=0)
        self.buried_error=self.get_buried_error()
        self.sort_flag=jnp.insert(self.sort_flag,idx,jnp.array(False))
        self.roots,self.parity=self.add_sorted_roots(roots=jnp.insert(self.roots,idx,add_roots,axis=0),parity=jnp.insert(self.parity,idx,add_parity,axis=0))
        self.find_create_points()
    def get_buried_error(self):
        error_buried=jnp.zeros(self.sample_n)
        ghost_roots_dis=self.ghost_roots_dis.ravel()
        idx1=jnp.where(ghost_roots_dis[0:-2]>2*ghost_roots_dis[1:-1])[0]+1
        idx1=idx1[~jnp.isnan(ghost_roots_dis[idx1+1])]
        error_buried=error_buried.at[idx1+1].add((ghost_roots_dis[idx1]-ghost_roots_dis[idx1-1])**2)
        error_buried=error_buried.at[idx1].add((ghost_roots_dis[idx1]-ghost_roots_dis[idx1-1])**2)
        idx1=jnp.where(2*ghost_roots_dis[1:-1]<ghost_roots_dis[2:])[0]+1
        idx1=idx1[~jnp.isnan(ghost_roots_dis[idx1-1])]
        error_buried=error_buried.at[idx1].add((ghost_roots_dis[idx1+1]-ghost_roots_dis[idx1])**2)
        error_buried=error_buried.at[idx1+1].add((ghost_roots_dis[idx1+1]-ghost_roots_dis[idx1])**2)
        return error_buried
    def add_sorted_roots(self,roots,parity):
        sort_flag=self.sort_flag
        flase_i=jnp.where(~sort_flag)[0]
        carry=get_sorted_roots(flase_i,roots,parity)
        resort_i=jnp.where((~sort_flag[0:-1])&(sort_flag[1:]))[0]
        carry,_=lax.scan(sort_body2,carry,resort_i)
        roots,parity=carry
        self.sort_flag=self.sort_flag.at[:].set(True)
        return roots,parity
    def get_real_roots(self,coff,zeta_l):
        sample_n=jnp.shape(zeta_l)[0]
        roots=get_roots(sample_n,coff)
        parity=get_parity(roots,self.s,self.m1,self.m2)
        error=verify(zeta_l[:,None],roots,self.s,self.m1,self.m2)
        cond=error>1e-6
        nan_num=cond.sum(axis=1)
        ####计算verify,如果parity出现错误或者nan个数错误，则重新规定error最大的为nan
        idx_verify_wrong=jnp.where(((nan_num!=0)&(nan_num!=2)))[0]#verify出现错误的根的索引
        if jnp.size(idx_verify_wrong)!=0:
            sorted=jnp.argsort(error[idx_verify_wrong],axis=1)[:,-2:]
            cond=cond.at[idx_verify_wrong].set(False)
            cond=cond.at[idx_verify_wrong,sorted[:,0]].set(True)
            cond=cond.at[idx_verify_wrong,sorted[:,1]].set(True)
        ####根的处理
        nan_num=cond.sum(axis=1)
        parity_sum=jnp.nansum(parity[~cond])
        real_roots=jnp.where(cond,jnp.nan+jnp.nan*1j,roots)
        real_parity=jnp.where(cond,jnp.nan,parity)
        parity_sum=jnp.nansum(real_parity,axis=1)
        idx_parity_wrong=jnp.where(parity_sum!=-1)[0]#parity计算出现错误的根的索引
        ##对于parity计算错误的点，分为fifth principal left center right，其中left center right 的parity为-1，1，-1
        if jnp.size(idx_parity_wrong)!=0:
            for i in idx_parity_wrong:
                temp=real_roots[i]
                if nan_num[i]==0:
                    prin_root=temp[jnp.sign(temp.imag)==jnp.sign(zeta_l.imag[i])][jnp.newaxis][0]
                    prin_root=jnp.concatenate([prin_root,temp[jnp.argmax(get_parity_error(temp,self.s,self.m1,self.m2))][jnp.newaxis]])
                    if (jnp.shape(real_parity)[0]==3)|(jnp.abs(zeta_l.imag[i])>1e-5):#初始三个点必须全部正确
                        real_parity=real_parity.at[i,temp==prin_root[0]].set(1);real_parity=real_parity.at[i,temp==prin_root[1]].set(-1)#是否对主图像与负图像进行parity的赋值
                    other=jnp.setdiff1d(temp,prin_root)
                    x_sort=jnp.argsort(other.real)
                    real_parity=real_parity.at[i,(temp==other[x_sort[0]])|(temp==other[x_sort[-1]])].set(-1)
                    real_parity=real_parity.at[i,(temp==other[x_sort[1]])].set(1)
                else:##对于三个根怎么判断其parity更加合理
                    if (jnp.abs(zeta_l.imag[i])>1e-5)|((jnp.shape(real_parity)[0]==3)):#初始三个点必须全部正确
                        real_parity=real_parity.at[i,~cond[i]].set(-1)
                        real_parity=real_parity.at[i,jnp.sign(temp.imag)==jnp.sign(zeta_l.imag[i])].set(1)##通过主图像判断，与zeta位于y轴同一侧的为1'''
        parity_sum=jnp.nansum(real_parity,axis=1)
        if (parity_sum!=-1).any():
            idx=jnp.where(parity_sum!=-1)[0]
            self.sample_n-=jnp.size(idx)
            self.coff=jnp.delete(self.coff,idx,axis=0)
            self.zeta_l=jnp.delete(self.zeta_l,idx)
            self.theta=jnp.delete(self.theta,idx)
            real_parity=jnp.delete(real_parity,idx,axis=0)
            cond=jnp.delete(cond,idx,axis=0)
            roots=jnp.delete(roots,idx,axis=0)
            real_roots=jnp.delete(real_roots,idx,axis=0)
            self.idx=jnp.delete(self.idx,idx,axis=0)
            self.outofloop=True
        ###计算得到最终的
        ghost_roots=jnp.where(cond,roots,jnp.inf)
        ghost_roots=jnp.sort(ghost_roots,axis=1)[:,0:2]
        ghost_roots=ghost_roots.at[ghost_roots==jnp.inf].set(jnp.nan)
        ghost_roots_dis=jnp.abs(jnp.diff(ghost_roots,axis=1))
        return real_roots,real_parity,ghost_roots_dis
    def find_create_points(self):
        cond=jnp.isnan(self.roots)
        Is_create=jnp.zeros_like(self.roots,dtype=int)
        idx_x,idx_y=jnp.where(jnp.diff(cond,axis=0))
        idx_x+=1
        for x,y in zip(idx_x,idx_y):
            if ~cond[x,y]:#如果这个不是nan
                Is_create=Is_create.at[x,y].set(1)#这个是destruction
            elif (cond[x,y])&(Is_create[x-1,y]!=1):#如果这个不是
                Is_create=Is_create.at[x-1,y].set(-1)
            else:
                Is_create=Is_create.at[x-1,y].add(-1)
        self.Is_create=Is_create
class Error_estimator(object):
    def __init__(self,arc,rho):
        q=arc.q;s=arc.s;self.rho=rho;self.paritypar=arc.parity
        zeta_l=arc.roots
        self.parity=arc.parity
        self.zeta_l=zeta_l
        theta=arc.theta
        self.theta=arc.theta;self.Is_create=arc.Is_create
        self.delta_theta=jnp.diff(self.theta)
        zeta_conj=jnp.conj(self.zeta_l)
        parZetaConZ=1/(1+q)*(1/(zeta_conj-s)**2+q/zeta_conj**2)
        par2ConZetaZ=-2/(1+q)*(1/(zeta_l-s)**3+q/(zeta_l)**3)
        de_zeta=1j*self.rho*jnp.exp(1j*theta)
        detJ=1-jnp.abs(parZetaConZ)**2
        de_z=(de_zeta[:,jnp.newaxis]-parZetaConZ*jnp.conj(de_zeta[:,jnp.newaxis]))/detJ
        deXProde2X=(self.rho**2+jnp.imag(de_z**2*de_zeta[:,jnp.newaxis]*par2ConZetaZ))/detJ
        self.product=deXProde2X
        self.de_z=de_z
    def error_ordinary(self):
        deXProde2X=self.product
        delta_theta=self.delta_theta
        zeta_l=self.zeta_l
        e1=jnp.nansum(jnp.abs(1/48*jnp.abs(jnp.abs(deXProde2X[0:-1]-jnp.abs(deXProde2X[1:])))*delta_theta[:,jnp.newaxis]**3),axis=1)
        dAp_1=1/24*((deXProde2X[0:-1]+deXProde2X[1:]))*delta_theta[:,jnp.newaxis]
        dAp=dAp_1*delta_theta[:,jnp.newaxis]**2*self.parity[0:-1]
        delta_theta_wave=jnp.abs(zeta_l[0:-1]-zeta_l[1:])**2/jnp.abs(dot_product(self.de_z[0:-1],self.de_z[1:]))
        e2=jnp.nansum(3/2*jnp.abs(dAp_1*(delta_theta_wave-delta_theta[:,jnp.newaxis]**2)),axis=1)
        e3=jnp.nansum(1/10*jnp.abs(dAp)*delta_theta[:,jnp.newaxis]**2,axis=1)
        e_tot=(e1+e2+e3)/(jnp.pi*self.rho**2)
        return e_tot,jnp.nansum(dAp)#抛物线近似的补偿项
    def error_critial(self,i,pos_idx,neg_idx,Is_create):
        zeta_l=self.zeta_l
        de_z=self.de_z
        deXProde2X=self.product
        zeta_pos=zeta_l[i,pos_idx]
        zeta_neg=zeta_l[i,neg_idx]
        theta_wave=jnp.abs(zeta_pos-zeta_neg)/jnp.sqrt(jnp.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx])))
        ce1=1/48*jnp.abs(deXProde2X[i,pos_idx]+deXProde2X[i,neg_idx])*theta_wave**3
        ce2=3/2*jnp.abs(dot_product(zeta_pos-zeta_neg,de_z[i,pos_idx]-de_z[i,neg_idx])-Is_create*2*jnp.abs(zeta_pos-zeta_neg)*jnp.sqrt(jnp.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx]))))*theta_wave
        dAcP=self.parity[i,pos_idx]*1/24*(deXProde2X[i,pos_idx]-deXProde2X[i,neg_idx])*theta_wave**3
        ce3=1/10*jnp.abs(dAcP)*theta_wave**2
        ce_tot=(ce1+ce2+ce3)/(jnp.pi*self.rho**2)
        return ce_tot,jnp.sum(dAcP)#critial 附近的抛物线近似'''
    def error_sum(self):
        error_hist=jnp.zeros_like(self.theta)
        mag=0
        e_ord,parab=self.error_ordinary()
        error_hist=error_hist.at[1:].set(e_ord)
        if  (self.Is_create!=0).any():
            critial_idx_row=jnp.where((self.Is_create!=0).any(axis=1))[0]
            for i in critial_idx_row:
                if jnp.abs(self.Is_create[i].sum())/2==1:
                    create=self.Is_create[i].sum()/2
                    idx1=jnp.where((self.Is_create[i]!=0)&(self.parity[i]==-1*create))[0]
                    idx2=jnp.where((self.Is_create[i]!=0)&(self.parity[i]==1*create))[0]
                    e_crit,dacp=self.error_critial(i,idx1,idx2,create)
                    error_hist=error_hist.at[int(i-(create-1)/2)].add(e_crit[0])
                    mag+=(1/2*(self.zeta_l[i,idx1].imag+self.zeta_l[i,idx2].imag)*(self.zeta_l[i,idx1].real-self.zeta_l[i,idx2].real))[0]
                    parab+=dacp
                else:
                    create=1
                    idx1=jnp.where(((self.Is_create[i]==1)|(self.Is_create[i]==10))&(self.parity[i]==-1*create))[0]
                    idx2=jnp.where(((self.Is_create[i]==1)|(self.Is_create[i]==10))&(self.parity[i]==1*create))[0]
                    mag+=(1/2*(self.zeta_l[i,idx1].imag+self.zeta_l[i,idx2].imag)*(self.zeta_l[i,idx1].real-self.zeta_l[i,idx2].real))[0]
                    e_crit,dacp=self.error_critial(i,idx1,idx2,create)
                    error_hist=error_hist.at[int(i-(create-1)/2)].add(e_crit[0])
                    parab+=dacp
                    create=-1
                    idx1=jnp.where(((self.Is_create[i]==-1)|(self.Is_create[i]==10))&(self.parity[i]==-1*create))[0]
                    idx2=jnp.where(((self.Is_create[i]==-1)|(self.Is_create[i]==10))&(self.parity[i]==1*create))[0]
                    mag+=(1/2*(self.zeta_l[i,idx1].imag+self.zeta_l[i,idx2].imag)*(self.zeta_l[i,idx1].real-self.zeta_l[i,idx2].real))[0]
                    e_crit,dacp=self.error_critial(i,idx1,idx2,create)
                    error_hist=error_hist.at[int(i-(create-1)/2)].add(e_crit[0])
                    parab+=dacp
        return error_hist,mag,parab



