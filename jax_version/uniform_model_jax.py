import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from basic_function_jax import *
jax.config.update("jax_enable_x64", True)
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
    def image_match(self,solution):#roots in lowmass coordinate
        sample_n=solution.sample_n;theta=solution.theta;roots=solution.roots;parity=solution.parity;roots_is_create=solution.Is_create
        theta_map=[];uncom_theta_map=[];uncom_sol_num=[];sol_num=[];uncom_curve=[];curve=[];parity_map=[];uncom_parity_map=[]
        roots_non_nan=jnp.isnan(roots).sum(axis=0)==0
        roots_first_eq_last=jnp.isclose(roots[0,:],roots[-1,:],rtol=1e-6)
        complete_cond=(roots_non_nan&roots_first_eq_last)
        uncomplete_cond=(roots_non_nan&(~roots_first_eq_last))
        curve+=list(roots[:,complete_cond].T)
        theta_map+=[theta]*np.asarray(jnp.sum(complete_cond)).item()
        sol_num+=list(roots_is_create[:,complete_cond].T)
        parity_map+=list(parity[:,complete_cond].T)
        flag=0;flag2=0
        if uncomplete_cond.any():
            flag2=1
            uncom_curve+=list(roots[:,uncomplete_cond].T)
            uncom_theta_map+=[theta]*np.asarray(jnp.sum(uncomplete_cond)).item()
            uncom_sol_num+=list(roots_is_create[:,uncomplete_cond].T)
            uncom_parity_map+=list(parity[:,uncomplete_cond].T)
        if ((jnp.isnan(roots).sum(axis=0)!=sample_n)&(~roots_non_nan)).any():
            flag=1
            temp_idx=jnp.where((jnp.isnan(roots).sum(axis=0)!=sample_n)&(~roots_non_nan))[0]#the arc crossing caustic we store it to temp
        if flag:##split all roots with nan and caulcate mag
            temp_roots=roots[:,temp_idx]
            temp_Is_create=roots_is_create[:,temp_idx]
            temp_parity=parity[:,temp_idx]
            real_parity=jnp.copy(temp_parity)
            while jnp.shape(temp_idx)[0]!=0:
                initk,initm,temp_parity=search_first_postion(temp_roots,temp_parity)
                if (temp_parity[initk,initm]==1)&(initk!=-1):
                    temp_parity*=-1
                roots_c=jnp.copy(temp_roots)
                m_map,n_map,temp_roots,temp_parity=search([initk],[initm],temp_roots,temp_parity,temp_roots[initk,initm],temp_Is_create)
                if (initk!=0)&(initk!=-1):
                    m_map+=[m_map[0]]
                    n_map+=[n_map[0]]
                temp_curve=roots_c[jnp.array(m_map),jnp.array(n_map)];temp_cur_theta_map=theta[jnp.array(m_map)];temp_parity_map=real_parity[jnp.array(m_map),jnp.array(n_map)]
                temp_cur_num_map=temp_Is_create[jnp.array(m_map),jnp.array(n_map)]
                temp_idx=jnp.where((~jnp.isnan(temp_roots)).any(axis=0))[0]
                temp_roots=temp_roots[:,temp_idx];temp_parity=temp_parity[:,temp_idx];temp_Is_create=temp_Is_create[:,temp_idx];real_parity=real_parity[:,temp_idx]
                if jnp.isclose(temp_curve[0],temp_curve[-1],rtol=1e-6):
                    curve+=[temp_curve];theta_map+=[temp_cur_theta_map];sol_num+=[temp_cur_num_map];parity_map+=[temp_parity_map]
                else:
                    uncom_curve+=[temp_curve];uncom_theta_map+=[temp_cur_theta_map];uncom_sol_num+=[temp_cur_num_map];uncom_parity_map+=[temp_parity_map]
                    flag2=1
        if flag2:#flag2 is the uncompleted arc so we store it to uncom_curve
            if len(uncom_curve)!=0:
                arc=uncom_curve[0]
                arc_theta=uncom_theta_map[0]
                arc_num=uncom_sol_num[0]
                arc_parity=uncom_parity_map[0]
                length=len(uncom_curve)-1
                while length>0:
                    for k in range(1,len(uncom_curve)):
                        tail=arc[-1]
                        head=uncom_curve[k][0]
                        if jnp.isclose(tail,head,rtol=1e-6):
                            arc=jnp.append(arc,uncom_curve[k][1:])
                            arc_theta=jnp.append(arc_theta,uncom_theta_map[k][1:])
                            arc_num=jnp.append(arc_num,uncom_sol_num[k][1:])
                            arc_parity=jnp.append(arc_parity,uncom_parity_map[k][1:])
                            length-=1
                        else:
                            head=uncom_curve[k][-1]
                            if jnp.isclose(tail,head,rtol=1e-6):
                                arc=jnp.append(arc,uncom_curve[k][-1::-1][1:])
                                arc_theta=jnp.append(arc_theta,uncom_theta_map[k][-1::-1][1:])
                                arc_num=jnp.append(arc_num,uncom_sol_num[k][-1::-1][1:])
                                arc_parity=jnp.append(arc_parity,uncom_parity_map[k][-1::-1][1:])
                                length-=1
                curve+=[arc]
                theta_map+=[arc_theta]
                sol_num+=[arc_num]
                parity_map+=[arc_parity]
        return curve,theta_map,sol_num,parity_map
    def Quadrupole_test(self,zeta,z,zG,tol,fz0,fz1,fz2,fz3,J):
        rho=self.rho;q=self.q
        s=self.s
        cQ=6;cG=2;cP=2
        ####Quadrupole test
        miu_Q=jnp.abs(-2*jnp.real(3*jnp.conj(fz1(z))**3*fz2(z)**2-(3-3*J(z)+J(z)**2/2)*jnp.abs(fz2(z))**2+J(z)*jnp.conj(fz1(z))**2*fz3(z))/(J(z)**5))
        miu_C=jnp.abs(6*jnp.imag(3*jnp.conj(fz1(z))**3*fz2(z)**2)/(J(z)**5))
        cond1=jnp.sum(miu_Q+miu_C)*cQ*(rho**2+1e-4*tol)<tol
        ####ghost image test
        cond2=jnp.array([True])
        if zG.size!=0:
            zwave=jnp.conj(zeta)-fz0(zG)
            J_wave=1-fz1(zG)*fz1(zwave)
            miu_G=1/2*jnp.abs(J(zG)*J_wave**2/(J_wave*fz2(jnp.conj(zG))*fz1(zG)-jnp.conj(J_wave)*fz2(zG)*fz1(jnp.conj(zG))*fz1(zwave)))
            cond2=(cG*(rho+1e-3)<miu_G).all()#any更加宽松，因为ghost roots应该是同时消失的，理论上是没问题的
        #####planet test
        cond3=True
        if self.q<1e-2:
            s=self.s
            cond3=(jnp.abs(zeta+1/s)**2>cP*(rho**2+9*q/s**2))|(rho*rho*s*s<q)
        return cond1&cond2&cond3,jnp.sum(jnp.abs(1/J(z)))
    def get_magnifaction2(self,tol):
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        mag=jnp.zeros(trajectory_n)
        m1=self.m1;m2=self.m2;s=self.s
        fz0=lambda z :-m1/(z-s)-m2/z
        fz1=lambda z :m1/(z-s)**2+m2/z**2
        fz2=lambda z :-2*m1/(z-s)**3-2*m2/z**3
        fz3=lambda z :6*m1/(z-s)**4+6*m2/z**4
        J=lambda z : 1-fz1(z)*jnp.conj(fz1(z))
        zeta_l=trajectory_l
        coff=get_poly_coff(zeta_l,self.s,self.m2)
        for i in range(trajectory_n):
            z_l=jnp.roots(coff[i])
            error=verify(zeta_l[i],z_l,self.s,self.m1,self.m2)
            cond=error<1e-6
            z=z_l[cond];zG=z_l[cond==False]
            if (cond.sum()!=3) & (cond.sum()!=5):
                sortidx=jnp.argsort(error)
                z=z_l[sortidx[0:3]];zG=z_l[sortidx[-2:]]
            cond,temp_mag=self.Quadrupole_test(zeta_l[i],z,zG,tol,fz0,fz1,fz2,fz3,J)
            if ~cond:
                temp_mag,curve=self.contour_integrate(trajectory_l[i],tol,i)
            mag=mag.at[i].set(temp_mag)
        return mag
    def contour_integrate(self,trajectory_l,epsilon,i,epsilon_rel=0):
        sample_n=3;theta_init=jnp.array([0,jnp.pi,2*jnp.pi],dtype=jnp.float64)
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
                add_theta=[jnp.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False,dtype=jnp.float64)[1:] for i in range(jnp.shape(idx)[0])]
                idx = jnp.repeat(idx, add_number-1) # create an index array with the same length as add_item
                add_theta = jnp.concatenate(add_theta) # concatenate the list of arrays into a 1-D array
                add_zeta_l=self.get_zeta_l(trajectory_l,add_theta)
                add_coff=get_poly_coff(add_zeta_l,self.s,self.m2)
                solution.add_points(idx,add_zeta_l,add_coff,add_theta)
                sample_n=solution.sample_n
                theta_init=solution.theta
                error_hist=jnp.zeros_like(theta_init)
                outloop=solution.outofloop
            try:
                curve,theta,sol_num,parity_map=self.image_match(solution)
            except IndexError:
                raise SyntaxError('image match function error')
            for k in range(len(curve)):
                cur=curve[k]
                theta_map_k=theta[k]
                parity_k=parity_map[k]
                mag_k=1/2*jnp.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))
                mag+=mag_k*parity_k[0]
                Error=Error_estimator(self.q,self.s,self.rho,cur,theta_map_k,theta_init,sol_num[k],parity_k)
                error_k,parab=Error.error_sum()
                error_hist+=error_k
                mag+=parab
            mag=mag/(jnp.pi*self.rho**2)
            error_hist+=solution.buried_error
        return (mag,curve)
class Solution(object):
    def __init__(self,q,s,zeta_l,coff,theta):
        self.theta=theta
        self.q=q;self.s=s;self.m1=1/(1+q);self.m2=q/(1+q)
        self.zeta_l=zeta_l
        self.coff=coff
        self.sample_n=jnp.shape(zeta_l)[0]
        roots,parity,self.ghost_roots_dis=self.get_real_roots(coff,zeta_l)#cond非nan为true
        self.buried_error=self.get_buried_error()
        self.roots,self.parity=get_sorted_roots(self.sample_n,roots,parity)
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
        for i in flase_i:
            sort_indices=find_nearest(roots[i-1],parity[i-1],roots[i],parity[i])
            roots=roots.at[i].set(roots[i][sort_indices])
            parity=parity.at[i].set(parity[i][sort_indices])
            if sort_flag[i+1]:
                sort_indices=find_nearest(roots[i],parity[i],roots[i+1],parity[i+1])
                roots=roots.at[i+1:].set(roots[i+1:,sort_indices])
                parity=parity.at[i+1:].set(parity[i+1:,sort_indices])
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
    '''def get_sorted_roots(self,sample_n,roots,parity):#非nan为true
        def loop_body(roots,partiy)
        for k in range(1,sample_n):
            sort_indices=find_nearest(roots[k-1,:],parity[k-1,:],roots[k,:],parity[k,:])
            roots=roots.at[k,:].set(roots[k,sort_indices])
            parity=parity.at[k,:].set(parity[k,sort_indices])
        return roots,parity'''
class Error_estimator(object):
    def __init__(self,q,s,rho,matched_image_l,theta_map,theta_init,sol_num,parity_map):
        self.q=q;self.s=s;self.rho=rho;self.cur_par=parity_map[0]
        zeta_l=matched_image_l;self.zeta_l=zeta_l
        theta=jnp.unwrap(theta_map);self.theta=theta;self.sol_num=sol_num#length=n
        self.theta_init=theta_init
        self.delta_theta=jnp.diff(theta)
        zeta_conj=jnp.conj(self.zeta_l)
        parZetaConZ=1/(1+q)*(1/(zeta_conj-s)**2+q/zeta_conj**2);self.parity=parity_map
        #par2ZetaConZ=-2/(1+q)*(1/(zeta_conj-s)**3+q/(zeta_conj)**3)
        par2ConZetaZ=-2/(1+q)*(1/(zeta_l-s)**3+q/(zeta_l)**3)
        de_zeta=1j*self.rho*jnp.exp(1j*theta)
        #de2_zeta=-self.rho*jnp.exp(1j*theta)
        detJ=1-jnp.abs(parZetaConZ)**2
        de_z=(de_zeta-parZetaConZ*jnp.conj(de_zeta))/detJ
        #de2_z=(de2_zeta-par2ZetaConZ*(jnp.conj(de_z)**2)-parZetaConZ*(jnp.conj(de2_zeta)-par2ConZetaZ*(de_z)**2))/detJ
        deXProde2X=(self.rho**2+jnp.imag(de_z**2*de_zeta*par2ConZetaZ))/detJ
        self.product=deXProde2X
        self.de_z=de_z
    def error_ordinary(self):
        deXProde2X=self.product
        delta_theta=self.delta_theta
        zeta_l=self.zeta_l
        e1=jnp.abs(1/48*jnp.abs(jnp.abs(deXProde2X[0:-1]-jnp.abs(deXProde2X[1:])))*delta_theta**3)
        dAp_1=1/24*((deXProde2X[0:-1]+deXProde2X[1:]))*delta_theta
        dAp=dAp_1*delta_theta**2
        delta_theta_wave=jnp.abs(zeta_l[0:-1]-zeta_l[1:])**2/jnp.abs(dot_product(self.de_z[0:-1],self.de_z[1:]))
        e2=3/2*jnp.abs(dAp_1*(delta_theta_wave-delta_theta**2))
        e3=1/10*jnp.abs(dAp)*delta_theta**2
        e_tot=(e1+e2+e3)/(jnp.pi*self.rho**2)
        return e_tot,self.cur_par*jnp.sum(dAp)#抛物线近似的补偿项
    def error_critial(self,critial_points):
        zeta_l=self.zeta_l
        de_z=self.de_z
        deXProde2X=self.product
        parity=self.parity
        pos_idx=critial_points;zeta_pos=zeta_l[pos_idx]
        neg_idx=critial_points+1;zeta_neg=zeta_l[neg_idx]
        theta_wave=jnp.abs(zeta_pos-zeta_neg)/jnp.sqrt(jnp.abs(dot_product(de_z[pos_idx],de_z[neg_idx])))
        ce1=1/48*jnp.abs(deXProde2X[pos_idx]+deXProde2X[neg_idx])*theta_wave**3
        Is_create=self.sol_num[pos_idx-(jnp.abs(self.sol_num[pos_idx])).astype(int)+1]#1 for ture -1 for false
        ce2=3/2*jnp.abs(dot_product(zeta_pos-zeta_neg,de_z[pos_idx]-de_z[neg_idx])-Is_create*2*jnp.abs(zeta_pos-zeta_neg)*jnp.sqrt(jnp.abs(dot_product(de_z[pos_idx],de_z[neg_idx]))))*theta_wave
        dAcP=parity[pos_idx]*1/24*(deXProde2X[pos_idx]-deXProde2X[neg_idx])*theta_wave**3
        ce3=1/10*jnp.abs(dAcP)*theta_wave**2
        ce_tot=(ce1+ce2+ce3)/(jnp.pi*self.rho**2)
        return ce_tot,jnp.sum(dAcP),Is_create#critial 附近的抛物线近似'''
    def error_sum(self):
        theta_init=self.theta_init
        e_ord,parab=self.error_ordinary()
        interval_theta=((self.theta[0:-1]+self.theta[1:])/2)#error 对应的区间的中心的theta值
        critial_points=jnp.nonzero(jnp.diff(self.parity))[0]
        if  jnp.shape(critial_points)[0]!=0:
            e_crit,dacp,Is_create=self.error_critial(critial_points)
            e_ord=e_ord.at[critial_points].set(e_crit)
            interval_theta=interval_theta.at[critial_points].add(-0.5*Is_create*jnp.min(jnp.abs(jnp.diff(theta_init))))
            parab+=dacp
        error_map=jnp.zeros_like(theta_init)#error 按照theta 排序
        indices = jnp.searchsorted(theta_init, interval_theta%(2*jnp.pi))
        error_map=error_map.at[indices].add(e_ord)
        return error_map,parab



