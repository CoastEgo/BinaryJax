import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from basic_function import *
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
    def image_match(self,solution):#roots in lowmass coordinate
        sample_n=solution.sample_n;theta=solution.theta;roots=solution.roots;parity=solution.parity;roots_is_create=solution.Is_create
        theta_map=[];uncom_theta_map=[];uncom_sol_num=[];sol_num=[];uncom_curve=[];curve=[];parity_map=[];uncom_parity_map=[]
        roots_non_nan=np.isnan(roots).sum(axis=0)==0
        roots_first_eq_last=np.isclose(roots[0,:],roots[-1,:],rtol=1e-6)
        complete_cond=(roots_non_nan&roots_first_eq_last)
        uncomplete_cond=(roots_non_nan&(~roots_first_eq_last))
        curve+=list(roots[:,complete_cond].T)
        theta_map+=[theta]*np.sum(complete_cond);sol_num+=list(roots_is_create[:,complete_cond].T);parity_map+=list(parity[:,complete_cond].T)
        flag=0;flag2=0
        if uncomplete_cond.any():
            flag2=1
            uncom_curve+=list(roots[:,uncomplete_cond].T)
            uncom_theta_map+=[theta]*np.sum(uncomplete_cond)
            uncom_sol_num+=list(roots_is_create[:,uncomplete_cond].T)
            uncom_parity_map+=list(parity[:,uncomplete_cond].T)
        if ((np.isnan(roots).sum(axis=0)!=sample_n)&(~roots_non_nan)).any():
            flag=1
            temp_idx=np.where((np.isnan(roots).sum(axis=0)!=sample_n)&(~roots_non_nan))[0]#the arc crossing caustic we store it to temp
        if flag:##split all roots with nan and caulcate mag
            temp_roots=roots[:,temp_idx]
            temp_Is_create=roots_is_create[:,temp_idx]
            temp_parity=parity[:,temp_idx]
            real_parity=np.copy(temp_parity)
            while np.shape(temp_idx)[0]!=0:
                initk,initm,temp_parity=search_first_postion(temp_roots,temp_parity)
                if (temp_parity[initk,initm]==1)&(initk!=-1):
                    temp_parity*=-1
                roots_c=np.copy(temp_roots)
                m_map,n_map,temp_roots,temp_parity=search([initk],[initm],temp_roots,temp_parity,temp_roots[initk,initm],temp_Is_create)
                if (initk!=0)&(initk!=-1):
                    m_map+=[m_map[0]]
                    n_map+=[n_map[0]]
                temp_curve=roots_c[m_map,n_map];temp_cur_theta_map=theta[m_map];temp_parity_map=real_parity[m_map,n_map]
                temp_cur_num_map=temp_Is_create[m_map,n_map]
                temp_idx=np.where((~np.isnan(temp_roots)).any(axis=0))[0]
                temp_roots=temp_roots[:,temp_idx];temp_parity=temp_parity[:,temp_idx];temp_Is_create=temp_Is_create[:,temp_idx];real_parity=real_parity[:,temp_idx]
                if np.isclose(temp_curve[0],temp_curve[-1],rtol=1e-6):
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
                        if np.isclose(tail,head,rtol=1e-6):
                            arc=np.append(arc,uncom_curve[k][1:])
                            arc_theta=np.append(arc_theta,uncom_theta_map[k][1:])
                            arc_num=np.append(arc_num,uncom_sol_num[k][1:])
                            arc_parity=np.append(arc_parity,uncom_parity_map[k][1:])
                            length-=1
                        else:
                            head=uncom_curve[k][-1]
                            if np.isclose(tail,head,rtol=1e-6):
                                arc=np.append(arc,uncom_curve[k][-1::-1][1:])
                                arc_theta=np.append(arc_theta,uncom_theta_map[k][-1::-1][1:])
                                arc_num=np.append(arc_num,uncom_sol_num[k][-1::-1][1:])
                                arc_parity=np.append(arc_parity,uncom_parity_map[k][-1::-1][1:])
                                length-=1
                curve+=[arc]
                theta_map+=[arc_theta]
                sol_num+=[arc_num]
                parity_map+=[arc_parity]
        return curve,theta_map,sol_num,parity_map
    def get_magnifaction2(self,tol,retol=0):
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        zeta_l=trajectory_l
        coff=get_poly_coff(zeta_l,self.s,self.m2)
        z_l=get_roots(trajectory_n,coff)
        error=verify(zeta_l[:,np.newaxis],z_l,self.s,self.m1,self.m2)
        cond=error<1e-6
        index=np.where((cond.sum(axis=1)!=3) & (cond.sum(axis=1)!=5))[0]
        if index.size!=0:
            sortidx=np.argsort(error[index],axis=1)
            cond[index]=False
            cond[index,sortidx[0:3]]=True
        z=np.where(cond,z_l,np.nan)
        zG=np.where(cond,np.nan,z_l)
        cond,mag=Quadrupole_test(self.rho,self.s,self.q,zeta_l,z,zG,tol)
        idx=np.where(~cond)[0]
        for i in idx:
            temp_mag,curve,theta=self.contour_integrate(trajectory_l[i],tol,i,retol)
            mag[i]=temp_mag
            '''fig,ax=plt.subplots()#绘制image图的代码
            zeta=self.get_zeta_l(trajectory_l[i],theta)
            zeta=self.to_centroid(zeta)
            ax.plot(zeta.real,zeta.imag,label='source')
            caustic_1=caustics.Caustics(self.q,self.s)
            caustic_1.plot(5000,s=2,label='caustic')
            x,y=caustic_1.get_caustics()
            x=caustic_1.critical_curve.x
            y=caustic_1.critical_curve.y
            ax.scatter(x,y,s=2,label='critical curve')
            for k in range(len(curve)):
                cur=self.to_centroid(curve[k])
                ax.plot(cur.real,cur.imag,label='image '+str(k))
            plt.axis('equal')
            plt.legend(loc='upper right')
            plt.show()'''
        return mag
    def contour_integrate(self,trajectory_l,epsilon,i,epsilon_rel=0):
        sample_n=3;theta_init=np.array([0,np.pi,2*np.pi],dtype=np.float64)
        error_hist=np.ones(1)
        add_max=np.ceil(-2*np.log(self.rho)-np.log(epsilon)/np.log(3))*10
        mag=1
        outloop=False
        mag0=0
        while ((error_hist>epsilon/np.sqrt(sample_n)).any() & (error_hist/np.abs(mag)>epsilon_rel/np.sqrt(sample_n)).any() & (np.abs(mag-mag0)>1/2*epsilon)):#相对误差
        #while ((np.sum(error_hist)>epsilon)  & (np.abs(mag-mag0)>1/2*epsilon)):#相对误差
            mini_interval=np.min(np.abs(np.diff(theta_init)))
            if mini_interval<1e-14:
                print(f'Warnning! idx{i} theta sampling may lower than float64 limit')
                break
            if outloop:
                print(f'idx{i} did not reach the accuracy goal due to the ambigious of roots and parity')
                break
            mag0=mag
            mag=0
            if np.shape(error_hist)[0]==1:#第一次采样
                error_hist=np.zeros_like(theta_init)
                zeta_l=self.get_zeta_l(trajectory_l,theta_init)
                coff=get_poly_coff(zeta_l,self.s,self.m2)
                solution=Solution(self.q,self.s,zeta_l,coff,theta_init)
            else:#自适应采点插入theta
                idx=np.where(error_hist>epsilon/np.sqrt(sample_n))[0]#不满足要求的点
                #idx=np.array([np.argmax(error_hist)])
                add_number=np.ceil((error_hist[idx]/epsilon*np.sqrt(sample_n))**0.2).astype(int)+1#至少要插入一个点，不包括相同的第一个
                add_number[add_number>add_max]=add_max
                add_theta=[np.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False,dtype=np.float64)[1:] for i in range(np.shape(idx)[0])]
                idx = np.repeat(idx, add_number-1) # create an index array with the same length as add_item
                add_theta = np.concatenate(add_theta) # concatenate the list of arrays into a 1-D array
                add_zeta_l=self.get_zeta_l(trajectory_l,add_theta)
                add_coff=get_poly_coff(add_zeta_l,self.s,self.m2)
                solution.add_points(idx,add_zeta_l,add_coff,add_theta)
                sample_n=solution.sample_n
                theta_init=solution.theta
                error_hist=np.zeros_like(theta_init)
                outloop=solution.outofloop
            try:
                curve,theta,sol_num,parity_map=self.image_match(solution)
            except IndexError:
                raise SyntaxError('image match function error')
            for k in range(len(curve)):
                cur=curve[k]
                theta_map_k=theta[k]
                parity_k=parity_map[k]
                mag_k=1/2*np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))
                mag+=mag_k*parity_k[0]
                Error=Error_estimator(self.q,self.s,self.rho,cur,theta_map_k,theta_init,sol_num[k],parity_k)
                error_k,parab=Error.error_sum()
                error_hist+=error_k
                mag+=parab
            mag=mag/(np.pi*self.rho**2)
            error_hist+=solution.buried_error
        print(sample_n)
        return mag,curve,solution.theta
    def get_magnifaction(self,tol):
        epsilon=tol
        rel_epsilon=tol/10
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        mag_curve=[]
        image_contour_all=[]
        #add_max=np.ceil(-2*np.log(self.rho)-np.log(tol)/np.log(5))*10
        for i in range(trajectory_n):
            sample_n=3;theta_init=np.array([0,np.pi,2*np.pi],dtype=np.float64)
            #error_tot_rel=np.ones(1)
            error_hist=np.ones(1)
            #while ((error_hist>epsilon/np.sqrt(sample_n)).any() & (np.abs(error_tot_rel)>rel_epsilon)):#相对误差
            #while ((error_hist>epsilon/np.sqrt(sample_n)).any() & ((error_tot_rel)>rel_epsilon/np.sqrt(sample_n)).any()):#相对误差
            while ((error_hist>epsilon/np.sqrt(sample_n)).any()):#多点采样但精度不同
                mag=0
                if np.shape(error_hist)[0]==1:#第一次采样
                    error_hist=np.zeros_like(theta_init)
                    zeta_l=self.get_zeta_l(trajectory_l[i],theta_init).astype(np.complex128)
                    coff=get_poly_coff(zeta_l,self.s,self.m2)
                    solution=Solution(self.q,self.s,zeta_l,coff,theta_init)
                else:#自适应采点插入theta
                    if 0:#单区间单点采样
                        idx=np.array([np.argmax(error_hist)]).reshape(1)
                        add_number=np.array([1])+1
                        add_theta=[np.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False)[1:] for i in range(np.shape(idx)[0])]
                    if 1:#多区间多点采样但精度设置不同
                        idx=np.argmax(error_hist)
                        #idx=np.where(error_hist>epsilon/np.sqrt(sample_n))[0]#不满足要求的点
                        add_number=np.ceil((error_hist[idx]/epsilon*np.sqrt(sample_n))**0.33).astype(int)+1#至少要插入一个点，不包括相同的第一个
                        add_theta=[np.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False,dtype=np.float64)[1:] for i in range(np.shape(idx)[0])]
                    idx = np.repeat(idx, add_number-1) # create an index array with the same length as add_item
                    add_theta = np.concatenate(add_theta) # concatenate the list of arrays into a 1-D array
                    add_zeta_l=self.get_zeta_l(trajectory_l[i],add_theta).astype(np.complex128)
                    add_coff=get_poly_coff(add_zeta_l,self.s,self.m2)
                    solution.add_points(idx,add_zeta_l,add_coff,add_theta)
                    sample_n=solution.sample_n
                    theta_init=solution.theta
                    error_hist=np.zeros_like(theta_init)
                try:
                    curve,theta,sol_num,parity_map=self.image_match(solution)
                except IndexError:
                    print(f'idx{i} occured indexerror please check')
                    quit()
                for k in range(len(curve)):
                    cur=curve[k]
                    theta_map_k=theta[k]
                    parity_k=parity_map[k]
                    mag_k=1/2*np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))
                    mag+=mag_k*parity_k[0]
                    Error=Error_estimator(self.q,self.s,self.rho,cur,theta_map_k,theta_init,sol_num[k],parity_k)
                    error_k,parab=Error.error_sum()
                    error_hist+=error_k
                    mag+=parab
                error_hist+=solution.buried_error
                #error_tot_rel=error_hist/np.abs(mag)*(np.pi*self.rho**2)
            mag=mag/(np.pi*self.rho**2)
            mag_curve+=[mag]
            image_contour_all+=[curve]
        self.image_contour_all=image_contour_all
        return np.array(mag_curve)
    def draw_anim(self,fig,axis):#given  a series of roots return picture
        ims=[]
        theta=np.linspace(0,2*np.pi,100)
        trajectory_n=self.trajectory_n
        trajectory_l=self.trajectory_l
        curve=self.image_contour_all
        for i in range(trajectory_n):
            zeta=self.to_centroid(self.get_zeta_l(trajectory_l[i],theta))
            img_root=[]
            img2,=axis.plot(zeta.real,zeta.imag,color='r',label=str(i))
            ttl = plt.text(0.5, 1.01, i, horizontalalignment='center', verticalalignment='bottom', transform=axis.transAxes)
            for k in range(len(curve[i])):
                img1,=axis.plot(self.to_centroid(curve[i][k]).real,self.to_centroid(curve[i][k]).imag)
                img_root+=[img1]
            ims.append(img_root+[img2]+[ttl])
        ani=animation.ArtistAnimation(fig,ims,interval=100,repeat_delay=1000)
        writervideo = animation.FFMpegWriter(fps=30) 
        ani.save('picture/animation.mp4',writer=writervideo)
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
                Is_create[x-1,y]-=1
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
    def __init__(self,q,s,rho,matched_image_l,theta_map,theta_init,sol_num,parity_map):
        self.q=q;self.s=s;self.rho=rho;self.cur_par=parity_map[0]
        zeta_l=matched_image_l;self.zeta_l=zeta_l
        theta=np.unwrap(theta_map);self.theta=theta;self.sol_num=sol_num#length=n
        self.theta_init=theta_init
        self.delta_theta=np.diff(theta)
        zeta_conj=np.conj(self.zeta_l)
        parZetaConZ=1/(1+q)*(1/(zeta_conj-s)**2+q/zeta_conj**2);self.parity=parity_map
        #par2ZetaConZ=-2/(1+q)*(1/(zeta_conj-s)**3+q/(zeta_conj)**3)
        par2ConZetaZ=-2/(1+q)*(1/(zeta_l-s)**3+q/(zeta_l)**3)
        de_zeta=1j*self.rho*np.exp(1j*theta)
        #de2_zeta=-self.rho*np.exp(1j*theta)
        detJ=1-np.abs(parZetaConZ)**2
        de_z=(de_zeta-parZetaConZ*np.conj(de_zeta))/detJ
        #de2_z=(de2_zeta-par2ZetaConZ*(np.conj(de_z)**2)-parZetaConZ*(np.conj(de2_zeta)-par2ConZetaZ*(de_z)**2))/detJ
        deXProde2X=(self.rho**2+np.imag(de_z**2*de_zeta*par2ConZetaZ))/detJ
        self.product=deXProde2X
        self.de_z=de_z
    def error_ordinary(self):
        deXProde2X=self.product
        delta_theta=self.delta_theta
        zeta_l=self.zeta_l
        e1=np.abs(1/48*np.abs(np.abs(deXProde2X[0:-1]-np.abs(deXProde2X[1:])))*delta_theta**3)
        dAp_1=1/24*((deXProde2X[0:-1]+deXProde2X[1:]))*delta_theta
        dAp=dAp_1*delta_theta**2
        delta_theta_wave=np.abs(zeta_l[0:-1]-zeta_l[1:])**2/np.abs(dot_product(self.de_z[0:-1],self.de_z[1:]))
        e2=3/2*np.abs(dAp_1*(delta_theta_wave-delta_theta**2))
        e3=1/10*np.abs(dAp)*delta_theta**2
        e_tot=(e1+e2+e3)/(np.pi*self.rho**2)
        return e_tot,self.cur_par*np.sum(dAp)#抛物线近似的补偿项
    def error_critial(self,critial_points):
        zeta_l=self.zeta_l
        de_z=self.de_z
        deXProde2X=self.product
        parity=self.parity
        pos_idx=critial_points;zeta_pos=zeta_l[pos_idx]
        neg_idx=critial_points+1;zeta_neg=zeta_l[neg_idx]
        theta_wave=np.abs(zeta_pos-zeta_neg)/np.sqrt(np.abs(dot_product(de_z[pos_idx],de_z[neg_idx])))
        ce1=1/48*np.abs(deXProde2X[pos_idx]+deXProde2X[neg_idx])*theta_wave**3
        Is_create=self.sol_num[pos_idx-(np.abs(self.sol_num[pos_idx])).astype(int)+1]#1 for ture -1 for false
        ce2=3/2*np.abs(dot_product(zeta_pos-zeta_neg,de_z[pos_idx]-de_z[neg_idx])-Is_create*2*np.abs(zeta_pos-zeta_neg)*np.sqrt(np.abs(dot_product(de_z[pos_idx],de_z[neg_idx]))))*theta_wave
        dAcP=parity[pos_idx]*1/24*(deXProde2X[pos_idx]-deXProde2X[neg_idx])*theta_wave**3
        ce3=1/10*np.abs(dAcP)*theta_wave**2
        ce_tot=(ce1+ce2+ce3)/(np.pi*self.rho**2)
        return ce_tot,np.sum(dAcP),Is_create#critial 附近的抛物线近似'''
    def error_sum(self):
        theta_init=self.theta_init
        e_ord,parab=self.error_ordinary()
        interval_theta=((self.theta[0:-1]+self.theta[1:])/2)#error 对应的区间的中心的theta值
        critial_points=np.nonzero(np.diff(self.parity))[0]
        if  np.shape(critial_points)[0]!=0:
            e_crit,dacp,Is_create=self.error_critial(critial_points)
            e_ord[critial_points]=e_crit
            interval_theta[critial_points]-=0.5*Is_create*np.min(np.abs(np.diff(theta_init)))
            parab+=dacp
        error_map=np.zeros_like(theta_init)#error 按照theta 排序
        indices = np.searchsorted(theta_init, interval_theta%(2*np.pi))
        np.add.at(error_map,indices,e_ord)
        return error_map,parab



