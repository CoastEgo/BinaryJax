import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from function_numpy import search,search_first_postion
import sys
from mpl_toolkits.mplot3d import Axes3D
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
    def verify(self,zeta_l,z_l):#verify whether the root is right
        return  z_l-self.m1/(np.conj(z_l)-self.s)-self.m2/np.conj(z_l)-zeta_l
    def parity(self,z):#get the parity of roots
        de_conjzeta_z1=self.m1/(np.conj(z)-self.s)**2+self.m2/np.conj(z)**2
        return np.sign((1-np.abs(de_conjzeta_z1)**2))
    def to_centroid(self,x):#change coordinate system to cetorid
        delta_x=self.s/(1+self.q)
        return -(np.conj(x)-delta_x)
    def to_lowmass(self,x):#change coordinaate system to lowmass
        delta_x=self.s/(1+self.q)
        return -np.conj(x)+delta_x
    def find_nearest(self,array1, array2):
        parity1=self.parity(array1)
        parity2=self.parity(array2)
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
    def poly_root_c(self,zeta_l,coff):
        #find and  sort the roots of one trajectory
        sample_n=len(zeta_l)
        roots=np.empty((sample_n,5),dtype=complex)
        roots_unverify=np.empty((sample_n,5),dtype=complex)
        roots_parity=np.zeros((sample_n,5),dtype=int)
        temp=np.zeros(5,dtype=complex)
        for k in range(sample_n):
            temp=np.roots(coff[k,:])
            roots_unverify[k,:]=temp
            cond=np.round(self.verify(zeta_l[k],temp),4)!=0
            temp[cond]=np.nan+1j*np.nan
            roots[k,:]=temp
        for k in range(sample_n):
            if k!=0:
                root_i_re_1=roots[k-1,:]
            root_i=roots[k,:]
            if k==0:
                temp=root_i
            else:
                temp=self.find_nearest(root_i_re_1,root_i)
            roots[k,:]=temp
        roots_parity=self.parity(roots)
        return roots,roots_parity#get roots in lowmass coordinate
    def roots_print(self,roots,roots_parity):
        np.savetxt('result/roots.txt',np.around(roots,4),delimiter=',',fmt='%1.4f')
        np.savetxt('result/parity.txt',np.around(roots_parity,4),delimiter=',',fmt='%1.1f')
    def image_match(self,theta,roots,parity):#roots in lowmass coordinate
        sample_n=np.shape(roots)[0]
        theta_map=[]
        uncom_theta_map_pos=[]
        uncom_curve_pos=[]
        curve=[]
        temp_idx=[]
        temp_parity=[]
        flag=0
        flag2=0
        for k in range(5):
            parity_ik=parity[:,k][~np.isnan(parity.real[:,k])]
            if len(parity_ik)==0:
                continue
            if (len(parity_ik)==sample_n)&(np.abs(np.round(roots[0,k]-roots[-1,k],4))==0):
                cur=roots[:,k]
                curve+=[cur];theta_map+=[theta]
            elif (len(parity_ik)==sample_n) & (np.abs(np.round(roots[0,k]-roots[-1,k],4))!=0):
                flag2=1#flag2 is the uncompleted arc so we store it to uncom_curve
                uncom_curve_pos+=[roots[:,k]];uncom_theta_map_pos+=[theta]
            else:
                flag=1#flag1 is the arc crossing caustic so we store it to temp
                temp_idx+=[k]
        if flag==1:##split all roots with nan and caulcate mag
            temp_roots=roots[:,temp_idx]
            temp_parity=parity[:,temp_idx]
            while len(temp_idx)!=0:
                initk,initm,temp_parity=search_first_postion(temp_roots,temp_parity)
                cur,cur_theta_map,temp_roots,temp_parity=search(initk,initm,temp_roots,temp_parity,[],theta,[])
                temp_curve=cur
                temp_cur_theta_map=cur_theta_map
                if (initk!=0)&(initk!=-1):
                    temp_curve=np.append(temp_curve,temp_curve[0])
                    temp_cur_theta_map=np.append(temp_cur_theta_map,temp_cur_theta_map[0])
                theta_map+=[temp_cur_theta_map]
                temp_idx=np.where(~np.isnan(temp_roots).all(axis=0))[0]
                temp_roots=temp_roots[:,temp_idx]
                temp_parity=temp_parity[:,temp_idx]
                if np.around(temp_curve[0]-temp_curve[-1],4)==0:
                    curve+=[temp_curve]
                    theta_map+=[temp_cur_theta_map]
                else:
                    uncom_curve_pos+=[temp_curve]
                    uncom_theta_map_pos+=[temp_cur_theta_map]
                    flag2=1
        if flag2:
            if len(uncom_curve_pos)!=0:
                uncom_curve=uncom_curve_pos
                uncom_theta_map=uncom_theta_map_pos
                cur=uncom_curve[0]
                cur_theta_map=uncom_theta_map[0]
                uncom_curve_n=len(uncom_curve)-1
                while uncom_curve_n>0:
                    for k in range(1,len(uncom_curve)):
                        tail=cur[-1]
                        head=uncom_curve[k][0]
                        if np.around(tail-head,4)==0:
                            cur=np.append(cur,uncom_curve[k][1:])
                            cur_theta_map=np.append(cur_theta_map,uncom_theta_map[k][1:])
                            uncom_curve_n-=1
                        else:
                            head=uncom_curve[k][-1]
                            if np.around(tail-head,4)==0:
                                cur=np.append(cur,uncom_curve[k][-1::-1])
                                cur_theta_map=np.append(cur_theta_map,uncom_theta_map[k][-1::-1])
                                uncom_curve_n-=1
                curve+=[cur]
                theta_map+=[cur_theta_map]
        return curve,theta_map
    def get_magnifaction(self):
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        mag_curve=[]
        #error_curve=[]
        image_contour_all=[]
        for i in range(trajectory_n):
            mag=0
            #error=0
            theta_init=np.linspace(0,2*np.pi,100)
            zeta_l=self.get_zeta_l(trajectory_l[i],theta_init)
            coff=self.get_poly_coff(zeta_l)
            roots,parity=self.poly_root_c(zeta_l,coff)
            if i==14:
                #self.roots_print(roots,parity)
                print(10)
            print(i)
            curve,theta_map=self.image_match(theta_init,roots,parity)
            for k in range(len(curve)):
                cur=curve[k]
                theta_map_k=theta_map[k]
                mag_k=1/2*np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))
                parity=self.parity(cur[0])
                mag+=parity*mag_k
                '''Error=error_estimator(self.q,self.s,self.rho,cur,theta_map_k,theta_init)
                error,dap=Error.error_ordinary()
                error_c,dacp,critial_idx=Error.error_critial()
                error_total=np.sum(error+error_c)
                mag+=parity*np.sum(dap)#抛物线近似的补偿项
                mag+=parity*np.sum(dacp)#critial 附近的抛物线近似'''
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
class error_estimator(object):
    def __init__(self,q,s,rho,matched_image_l,theta_map,theta_init):
        self.q=q;self.s=s;self.rho=rho
        zeta_l=matched_image_l;self.zeta_l=zeta_l
        theta=theta_map;self.theta=theta
        self.theta_init=theta_init
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
        self.delta_theta_smooth()
    def delta_theta_smooth(self):
        theta=self.theta
        delta_theta=np.diff(theta)
        delta_theta[np.where(delta_theta>np.pi)]-=2*np.pi
        delta_theta[np.where(delta_theta<-np.pi)]+=2*np.pi
        self.delta_theta=delta_theta
    def dot_product(self,a,b):
        return np.real(a)*np.real(b)+np.conj(a)*np.conj(b)
    def error_ordinary(self):
        theta=self.theta
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
        return e_tot/(np.pi*self.rho**2),dAp
    def error_critial(self):
        zeta_l=self.zeta_l
        de_z=self.de_z
        deXProde2X=self.product
        parity=self.parity
        parity_diff=np.diff(parity)
        critial_points=np.nonzero(parity_diff)[0]
        temp=np.copy(parity_diff)
        temp[critial_points+1]=1
        cond=(temp!=0)
        pos_idx=np.argwhere((parity[0:-1]==1)&(cond));zeta_pos=zeta_l[pos_idx]
        neg_idx=np.argwhere((parity[0:-1]==-1)&(cond));zeta_neg=zeta_l[neg_idx]
        theta_wave=np.abs(zeta_pos-zeta_neg)/np.sqrt(np.abs(self.dot_product(zeta_pos,zeta_neg)))
        dAcP=1/24*(deXProde2X[pos_idx]-deXProde2X[neg_idx])*theta_wave**3
        ce1=1/48*np.abs(deXProde2X[pos_idx]+deXProde2X[neg_idx])*theta_wave**3
        Is_create=np.greater(neg_idx,pos_idx)*2-1#1 for ture -1 for false
        ce2=3/2*np.abs(self.dot_product(zeta_pos-zeta_neg,de_z[pos_idx]-de_z[neg_idx])-Is_create*2*np.abs(zeta_pos-zeta_neg)*np.sqrt(np.abs(self.dot_product(zeta_pos,zeta_neg))))*theta_wave
        ce3=1/10*np.abs(dAcP)*theta_wave**2
        ce_tot=ce1+ce2+ce3
        return ce_tot/(np.pi*self.rho**2),dAcP,critial_points-Is_create
    def error_sum(self):
        e_ord,dap=self.error_ordinary()
        e_crit,dacp,insert_point=self.error_critial()
        e_ord[insert_point]+=e_crit
        error_zeta=np.zeros(len(e_ord))
        transform=np.sign(self.delta_theta)
        transform[transform>=0]=0
        sample_n=len(self.delta_theta)
        for i in range(sample_n-1):
            error_zeta[i]=e_ord[i-int(transform[i])]
        error_theta=np.zeros(len(self.theta_init))
        theta_map_idx=np.nonzero(np.isin(self.theta_init,self.theta))[0]#找到theta map对应的theta的索引
        #np.add.at(error_theta,theta_map_idx,error_zeta)
        fig=plt.figure()
        ax=Axes3D(fig)
        ax.plot3D(self.zeta_l.real[0:-1],self.zeta_l.imag[0:-1],error_zeta)
        plt.show()