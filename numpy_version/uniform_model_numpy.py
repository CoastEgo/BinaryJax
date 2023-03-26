import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from function_numpy import search,search_first_postion
import sys
from matplotlib import colors
import time
#实现自适应采点算法
#首先将轨道与同一点的采样分开
#其次将theta与contour采样绑定
#实现误差计算
#实现自适应采点算法
sys.setrecursionlimit(10000)
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
        self.alpha_deg=par['alpha_deg']
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
        trajectory_c=np.array([i*np.cos(alpha)-b*np.sin(alpha)+1j*(b*np.cos(alpha)+i*np.sin(alpha)) for i in self.times])
        trajectory_l=self.to_lowmass(trajectory_c)
        return trajectory_l
    def get_zeta_l(self,trajectory_centroid_l,theta):#获得等高线采样的zeta
        rho=self.rho
        rel_centroid=rho*np.cos(theta)+1j*rho*np.sin(theta)
        zeta_l=trajectory_centroid_l+rel_centroid
        return zeta_l
    def get_poly_coff(self,zeta_l):
        s=self.s
        m2=self.m2
        zeta_conj=np.conj(zeta_l)
        c0=s**2*zeta_l*m2**2
        c1=-s*m2*(2*zeta_l+s*(-1+s*zeta_l-2*zeta_l*zeta_conj+m2))
        c2=zeta_l-s**3*zeta_l*zeta_conj+s*(-1+m2-2*zeta_conj*zeta_l*(1+m2))+s**2*(zeta_conj-2*zeta_conj*m2+zeta_l*(1+zeta_conj**2+m2))
        c3=s**3*zeta_conj+2*zeta_l*zeta_conj+s**2*(-1+2*zeta_conj*zeta_l-zeta_conj**2+m2)-s*(zeta_l+2*zeta_l*zeta_conj**2-2*zeta_conj*m2)
        c4=zeta_conj*(-1+2*s*zeta_conj+zeta_conj*zeta_l)-s*(-1+2*s*zeta_conj+zeta_conj*zeta_l+m2)
        c5=(s-zeta_conj)*zeta_conj
        coff=np.stack((c5,c4,c3,c2,c1,c0),axis=1)
        return coff
    def image_match(self,solution):#roots in lowmass coordinate
        sample_n=solution.sample_n;theta=solution.theta;roots=solution.roots;parity=solution.parity;roots_is_create=solution.Is_create
        theta_map=[];uncom_theta_map=[];uncom_sol_num=[];sol_num=[];uncom_curve=[];curve=[]
        roots_non_nan=np.isnan(roots).sum(axis=0)==0
        roots_first_eq_last=np.isclose(roots[0,:],roots[-1,:],rtol=1e-6)
        complete_cond=(roots_non_nan&roots_first_eq_last)
        uncomplete_cond=(roots_non_nan&(~roots_first_eq_last))
        curve+=list(roots[:,complete_cond].T)
        theta_map+=[theta]*np.sum(complete_cond);sol_num+=list(roots_is_create[:,complete_cond].T)
        flag=0;flag2=0
        if uncomplete_cond.any():
            flag2=1
            uncom_curve+=list(roots[:,uncomplete_cond].T)
            uncom_theta_map+=[theta]*np.sum(uncomplete_cond)
            uncom_sol_num+=list(roots_is_create[:,uncomplete_cond].T)
        if ((np.isnan(roots).sum(axis=0)!=sample_n)&(~roots_non_nan)).any():
            flag=1
            temp_idx=np.where((np.isnan(roots).sum(axis=0)!=sample_n)&(~roots_non_nan))[0]#the arc crossing caustic we store it to temp
        if flag:##split all roots with nan and caulcate mag
            temp_roots=roots[:,temp_idx]
            temp_Is_create=roots_is_create[:,temp_idx]
            temp_parity=parity[:,temp_idx]
            while np.shape(temp_idx)[0]!=0:
                initk,initm,temp_parity=search_first_postion(temp_roots,temp_parity)
                roots_c=np.copy(temp_roots)
                m_map,n_map,temp_roots,temp_parity=search([initk],[initm],temp_roots,temp_parity,temp_roots[initk,initm])
                if (initk!=0)|(initk!=-1):
                    m_map+=[m_map[0]]
                    n_map+=[n_map[0]]
                temp_curve=roots_c[m_map,n_map];temp_cur_theta_map=theta[m_map]
                temp_cur_num_map=temp_Is_create[m_map,n_map]
                temp_idx=np.where((~np.isnan(temp_roots)).all(axis=0))[0]
                temp_roots=temp_roots[:,temp_idx]
                temp_parity=temp_parity[:,temp_idx]
                temp_Is_create=temp_Is_create[:,temp_idx]
                if np.isclose(temp_curve[0],temp_curve[-1],rtol=1e-6):
                    curve+=[temp_curve];theta_map+=[temp_cur_theta_map];sol_num+=[temp_cur_num_map]
                else:
                    uncom_curve+=[temp_curve];uncom_theta_map+=[temp_cur_theta_map];uncom_sol_num+=[temp_cur_num_map]
                    flag2=1
        if flag2:#flag2 is the uncompleted arc so we store it to uncom_curve
            if len(uncom_curve)!=0:
                arc=uncom_curve[0]
                arc_theta=uncom_theta_map[0]
                arc_num=uncom_sol_num[0]
                length=len(uncom_curve)-1
                while length>0:
                    for k in range(1,len(uncom_curve)):
                        tail=arc[-1]
                        head=uncom_curve[k][0]
                        if np.isclose(tail,head,rtol=1e-6):
                            arc=np.append(arc,uncom_curve[k][1:])
                            arc_theta=np.append(arc_theta,uncom_theta_map[k][1:])
                            arc_num=np.append(arc_num,uncom_sol_num[k][1:])
                            length-=1
                        else:
                            head=uncom_curve[k][-1]
                            if np.isclose(tail,head,rtol=1e-6):
                                arc=np.append(arc,uncom_curve[k][-1::-1])
                                arc_theta=np.append(arc_theta,uncom_theta_map[k][-1::-1][1:])
                                arc_num=np.append(arc_num,uncom_sol_num[k][-1::-1][1:])
                                length-=1
                curve+=[arc]
                theta_map+=[arc_theta]
                sol_num+=[arc_num]
        return curve,theta_map,sol_num
    def get_magnifaction(self):
        epsilon=1e-2
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        mag_curve=[]
        image_contour_all=[]
        for i in range(trajectory_n):
            sample_n=1000;theta_init=np.linspace(0,2*np.pi,sample_n)
            error_tot=np.ones(1);error_hist=np.ones(1)
            print(i)
            if i==34:
                print(i)
                #solution.roots_print()
            while ((error_tot>epsilon)):
                mag=0
                if np.shape(error_hist)[0]==1:#第一次采样
                    error_hist=np.zeros_like(theta_init)
                    theta_init=np.linspace(0,2*np.pi,sample_n)
                    zeta_l=self.get_zeta_l(trajectory_l[i],theta_init)
                    coff=self.get_poly_coff(zeta_l)
                    solution=Solution(self.q,self.s,zeta_l,coff,theta_init)
                else:#自适应采点插入theta
                    if 0:#单点采样
                        idx=np.array([np.argmax(error_hist)]).reshape(1)
                        add_number=np.ceil((error_hist[idx]/epsilon*np.sqrt(sample_n))**0.2).astype(int)+1
                        add_theta=[np.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False)[1:] for i in range(np.shape(idx)[0])]
                    if 1:#多点采样
                        idx=np.where(error_hist>(epsilon/sample_n))[0]#不满足要求的点
                        add_number=np.ceil((error_hist[idx]/epsilon*np.sqrt(sample_n))**0.2).astype(int)+1#至少要插入一个点，不包括相同的第一个
                        add_theta=[np.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False)[1:] for i in range(np.shape(idx)[0])]
                    idx = np.repeat(idx, add_number-1) # create an index array with the same length as add_item
                    add_theta = np.concatenate(add_theta) # concatenate the list of arrays into a 1-D array
                    add_zeta_l=self.get_zeta_l(trajectory_l[i],add_theta)
                    add_coff=self.get_poly_coff(add_zeta_l)
                    solution.add_points(idx,add_zeta_l,add_coff,add_theta)
                    sample_n=solution.sample_n
                    theta_init=solution.theta
                    error_hist=np.zeros_like(theta_init)
                try:
                    curve,theta,sol_num=self.image_match(solution)
                except IndexError:
                    print(f'idx{i} occured indexerror please check')
                    quit()
                #theta_init=np.insert(theta_init,838,(theta_init[837]+theta_init[838])/2)
                for k in range(len(curve)):
                    cur=curve[k]
                    theta_map_k=theta[k]
                    mag_k=1/2*np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))
                    parity_k=solution.get_parity(cur[0])*np.sign(theta_map_k[1]-theta_map_k[0])
                    mag+=parity_k*mag_k
                    Error=Error_estimator(self.q,self.s,self.rho,cur,theta_map_k,theta_init,sol_num[k],parity_k)
                    error_k,parab=Error.error_sum()
                    error_hist+=error_k
                    mag+=parab
                error_tot=np.sum(error_hist)
            mag=mag/(np.pi*self.rho**2)
            mag_curve+=[mag]
            #error_curve+=[error_total]
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
        roots=self.get_real_roots(coff,zeta_l)
        self.roots,self.parity=self.get_sorted_roots(roots)
        self.find_create_points()
    def add_points(self,idx,add_zeta,add_coff,add_theta):
        self.sample_n+=np.size(idx)
        self.theta=np.insert(self.theta,idx,add_theta)
        self.zeta_l=np.insert(self.zeta_l,idx,add_zeta)
        self.coff=np.insert(self.coff,idx,add_coff,axis=0)
        add_roots=self.get_real_roots(add_coff,add_zeta)
        self.roots,self.parity=self.get_sorted_roots(np.insert(self.roots,idx,add_roots,axis=0))
        self.find_create_points()
    def get_real_roots(self,coff,zeta_l):
        sample_n=np.shape(zeta_l)[0]
        roots=np.empty((sample_n,5),dtype=complex)
        cond=np.empty((sample_n,5),dtype=bool)
        for k in range(sample_n):
            temp=np.roots(coff[k,:])
            cond[k,:]=np.abs(self.verify(zeta_l[k],temp))>1e-6
            parity_sum=np.nansum(self.get_parity(temp)[~cond[k,:]])
            if (parity_sum==-1)&((cond[k,:].sum()==0)|(cond[k,:].sum()==2)):
                pass
            else:
                error=self.verify(zeta_l[k],temp)
                sorted=np.argsort(error)
                cond[k,:]=np.where((error==error[sorted[-1]])|(error==error[sorted[-2]]),True,False)
            temp[cond[k,:]]=np.nan+1j*np.nan
            roots[k,:]=temp
        return roots
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
    def get_sorted_roots(self,roots):
        sample_n=self.sample_n
        for k in range(sample_n):
            if k!=0:
                root_i_re_1=roots[k-1,:]
            root_i=roots[k,:]
            if k==0:
                temp=root_i
            else:
                temp=self.find_nearest(root_i_re_1,root_i)
            roots[k,:]=temp
        return roots,self.get_parity(roots)
    def find_nearest(self,array1, array2):
        parity1=self.get_parity(array1)
        parity2=self.get_parity(array2)
        array1=array1+10*parity1
        array2=array2+10*parity2
        idx1=np.where(~np.isnan(array1.real))[0]
        idx1_n=len(idx1)
        idx2=np.where(~np.isnan(array2.real))[0]
        idx2_n=len(idx2)
        if (idx1_n==3) & (idx2_n==5):
            temp=np.copy(array2)
            value=array1[~np.isnan(array1.real)]
            for value_i in value:
                i=np.where(array1==value_i)[0][0]
                idx=np.nanargmin(np.abs(value_i-temp))
                array2[idx],array2[i]=array2[i],array2[idx]
                parity2[idx],parity2[i]=parity2[i],parity2[idx]
                temp[idx],temp[i]=temp[i],temp[idx]
                temp[i]=np.nan
        else:
            temp=np.copy(array1)
            value=array2[~np.isnan(array2.real)]
            for value_i in value:
                i=np.where(array2==value_i)[0][0]
                idx=np.nanargmin(np.abs(temp-value_i))
                array2[idx],array2[i]=array2[i],array2[idx]
                parity2[idx],parity2[i]=parity2[i],parity2[idx]
                temp[idx]=np.nan
        return array2-10*parity2
    def verify(self,zeta_l,z_l):#verify whether the root is right
        return  z_l-self.m1/(np.conj(z_l)-self.s)-self.m2/np.conj(z_l)-zeta_l
    def get_parity(self,z):#get the parity of roots
        de_conjzeta_z1=self.m1/(np.conj(z)-self.s)**2+self.m2/np.conj(z)**2
        return np.sign((1-np.abs(de_conjzeta_z1)**2))
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
        np.savetxt('result/create.txt',self.Is_create,delimiter=',',fmt='%.0f')
        #np.savetxt('result/roots_diff.txt',np.abs(self.roots[1:]-self.roots[0:-1]),delimiter=',')
        #np.savetxt('result/roots_verify.txt',self.verify(np.array([self.zeta_l,self.zeta_l,self.zeta_l,self.zeta_l,zeta_l]).T,roots),delimiter=',')
class Error_estimator(object):
    def __init__(self,q,s,rho,matched_image_l,theta_map,theta_init,sol_num,curve_par):
        self.q=q;self.s=s;self.rho=rho;self.cur_par=curve_par
        zeta_l=matched_image_l;self.zeta_l=zeta_l
        theta=np.unwrap(theta_map);self.theta=theta;self.sol_num=sol_num#length=n
        self.theta_init=theta_init
        self.delta_theta=np.diff(theta)
        zeta_conj=np.conj(self.zeta_l)
        parZetaConZ=1/(1+q)*(1/(zeta_conj-s)**2+q/zeta_conj**2);self.parity=s=np.sign((1-np.abs(parZetaConZ)**2))
        par2ZetaConZ=-2/(1+q)*(1/(zeta_conj-s)**3+q/(zeta_conj)**3);par2ConZetaZ=-2/(1+q)*(1/(zeta_l-s)**3+q/(zeta_l)**3)
        de_zeta=1j*self.rho*np.exp(1j*theta)
        de2_zeta=-self.rho*np.exp(1j*theta)
        detJ=1-np.abs(parZetaConZ)**2
        de_z=(de_zeta-parZetaConZ*np.conj(de_zeta))/detJ
        de2_z=(de2_zeta-par2ZetaConZ*(np.conj(de_z)**2)-parZetaConZ*(np.conj(de2_zeta)-par2ConZetaZ*(de_z)**2))/detJ
        deXProde2X=(self.rho**2+np.imag(de_z**2*de_zeta*par2ConZetaZ))/detJ
        self.product=deXProde2X
        self.de_z=de_z
    def dot_product(self,a,b):
        return np.real(a)*np.real(b)+np.imag(a)*np.imag(b)
    def error_ordinary(self):
        deXProde2X=self.product
        delta_theta=self.delta_theta
        zeta_l=self.zeta_l
        e1=np.abs(1/48*np.abs(np.abs(deXProde2X[0:-1]-np.abs(deXProde2X[1:])))*delta_theta**3)
        dAp_1=1/24*((deXProde2X[0:-1]+deXProde2X[1:]))*delta_theta
        dAp=dAp_1*delta_theta**2
        delat_theta_wave=np.abs(zeta_l[0:-1]-zeta_l[1:])**2/np.abs(self.dot_product(self.de_z[0:-1],self.de_z[1:]))
        e2=3/2*np.abs(dAp_1*(delat_theta_wave-delta_theta**2))
        e3=1/10*np.abs(dAp)*delta_theta**2
        e_tot=e1+e2+e3
        return e_tot/(np.pi*self.rho**2),self.cur_par*np.sum(dAp)#抛物线近似的补偿项
    def error_critial(self,critial_points):
        zeta_l=self.zeta_l
        de_z=self.de_z
        deXProde2X=self.product
        parity=self.parity
        pos_idx=critial_points;zeta_pos=zeta_l[pos_idx]
        neg_idx=critial_points+1;zeta_neg=zeta_l[neg_idx]
        theta_wave=np.abs(zeta_pos-zeta_neg)/np.sqrt(np.abs(self.dot_product(de_z[pos_idx],de_z[neg_idx])))
        ce1=1/48*np.abs(deXProde2X[pos_idx]+deXProde2X[neg_idx])*theta_wave**3
        Is_create=self.sol_num[pos_idx-(np.abs(self.sol_num[pos_idx])).astype(int)+1]#1 for ture -1 for false
        ce2=3/2*np.abs(self.dot_product(zeta_pos-zeta_neg,de_z[pos_idx]-de_z[neg_idx])-Is_create*2*np.abs(zeta_pos-zeta_neg)*np.sqrt(np.abs(self.dot_product(de_z[pos_idx],de_z[neg_idx]))))*theta_wave
        dAcP=parity[pos_idx]*1/24*(deXProde2X[pos_idx]-deXProde2X[neg_idx])*theta_wave**3
        ce3=1/10*np.abs(dAcP)*theta_wave**2
        ce_tot=ce1+ce2+ce3
        return ce_tot/(np.pi*self.rho**2),np.sum(dAcP),Is_create#critial 附近的抛物线近似'''
    def error_sum(self):
        theta_init=self.theta_init
        e_ord,parab=self.error_ordinary()
        interval_theta=((self.theta[0:-1]+self.theta[1:])/2)#error 对应的区间的中心的theta值
        critial_points=np.nonzero(np.diff(self.parity))[0]
        if  np.shape(critial_points)[0]!=0:
            e_crit,dacp,Is_create=self.error_critial(critial_points)
            e_ord[critial_points]=e_crit
            interval_theta[critial_points]-=1e-9*Is_create#如果error出现在creat则theta减小，缶则theta增加
            parab+=dacp
        error_map=np.zeros_like(theta_init)#error 按照theta 排序
        indices = np.searchsorted(theta_init, interval_theta%(2*np.pi))
        np.add.at(error_map,indices,e_ord)
        return error_map,parab
if __name__=='__main__':
    if 0:
        b_map=np.linspace(-0.08,0.04,1200)
        b=b_map[747]
        t_0=2452848.06;t_E=61.5
        q=1e-4;alphadeg=90;s=1.0;rho=1e-3
        trajectory_n=300
        alpha=alphadeg*2*np.pi/360
        times=np.linspace(t_0-0.015*t_E,t_0+0.015*t_E,trajectory_n)
        model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                            'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
        tra=model_uniform.trajectory_l[75]
        s=model_uniform.s
        m1=model_uniform.m1
        m2=model_uniform.m2
        theta_init=np.linspace(0,2*np.pi,10000)
        zeta=model_uniform.get_zeta_l(tra,theta_init)
        i=7532
        i=1
        coff=model_uniform.get_poly_coff(zeta)[i]
        zeta=zeta[i]
        roots_all=np.roots(coff)
        print(np.sum(model_uniform.parity(roots_all)))
        roots_polished=model_uniform.root_polish(coff,roots_all,1e-10)
        print(model_uniform.verify(zeta,roots_all))
    if 0:
        theta_init=np.linspace(0,10,11)
        idx=np.array([1,5,4,6])
        add_number=np.array([3]*4)
        add_item=[np.linspace(theta_init[idx[i]-1],theta_init[idx[i]],add_number[i],endpoint=False)[0:-1] for i in range(np.shape(idx)[0])]
        print(add_item)
        theta_init=np.insert(theta_init,idx,add_item)
        print(theta_init)
    if 0:
        roots_number=np.array([5,3,5,5,5])
        cre_des_idx=np.where(np.diff(roots_number))[0]+1
        for i in cre_des_idx:
            if (roots_number[i]==5):
                roots_number[i]=1
            elif (roots_number[i]==3)&(roots_number[i-1]!=1):
                roots_number[i-1]=-1
            else:
                roots_number[i-1]-=1
        print(roots_number)
    if 1:
        roots=np.array([[1,np.nan,np.nan],[2,3,4],[np.nan,np.nan,5]])
        cond=np.isnan(roots)
        Is_create=np.zeros_like(roots)
        idx_x,idx_y=np.where(np.diff(cond,axis=0))
        idx_x+=1
        for x,y in zip(idx_x,idx_y):
            if ~cond[x,y]:#如果这个不是nan
                Is_create[x,y]=1#这个是destruction
            elif (cond[x,y])&(Is_create[x-1,y]!=1):#如果这个不是
                Is_create[x-1,y]=-1
            else:
                Is_create[x-1,y]-=1
        print(Is_create)



