import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from function_numpy import search,search_first_postion
import sys
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
        array1=array1*parity1
        array2=array2*parity2
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
        return array2/parity2
    def get_trajectory_l(self):
        alpha=self.alpha_rad
        b=self.u_0
        trajectory_c=np.array([i*np.cos(alpha)-b*np.sin(alpha)+1j*(b*np.cos(alpha)+i*np.sin(alpha)) for i in self.times])
        trajectory_l=self.to_lowmass(trajectory_c)
        return trajectory_l
    def get_zeta_l(self,trajectory_centroid_l,theta):#获得等高线采样的zeta
        rho=self.rho
        rel_centroid=rho*np.cos(theta)+1j*rho*np.sin(theta)
        zeta_l=self.to_lowmass(self.to_centroid(trajectory_centroid_l)+rel_centroid)
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
        return self.to_centroid(roots),self.to_centroid(roots_unverify),roots_parity
    def roots_print(self,roots,roots_parity):
        np.savetxt('result/roots.txt',np.around(roots,4),delimiter=',',fmt='%1.4f')
        np.savetxt('result/parity.txt',np.around(roots_parity,4),delimiter=',',fmt='%1.1f')
    def image_match(self,theta,roots,parity):#roots in centroid coordinate
        sample_n=np.shape(roots)[0]
        theta_map=[]
        uncom_theta_map=[]
        rho=self.rho
        mag=0
        curve=[]
        uncom_curve=[]
        temp_idx=[]
        temp_parity=[]
        flag=0
        flag2=0
        for k in range(5):
            parity_ik=parity[:,k][~np.isnan(parity.real[:,k])]
            if len(parity_ik)==0:
                continue
            if (len(parity_ik)==sample_n) & ((np.round(roots[0,k]-roots[-1,k],4))==0):
                cur=roots[:,k]
                curve+=[cur];theta_map+=[theta]
                value,counts=np.unique(parity_ik,return_counts=True)
                ind=np.argmax(counts)
                parity_most=value[ind]
                mag+=parity_most*np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))
            elif (len(parity_ik)==sample_n) & (np.abs(np.round(roots[0,k]-roots[-1,k],4))!=0):
                uncom_curve+=[roots[:,k]];uncom_theta_map+=[theta]
                flag2=1#flag2 is the uncompleted arc so we store it to uncom_curve
            else:
                flag=1#flag1 is the arc crossing caustic so we store it to temp
                temp_idx+=[k]
        if flag==1:##split all roots with nan and caulcate mag
            temp_roots=roots[:,temp_idx]
            temp_parity=parity[:,temp_idx]
            initk,initm,counts=search_first_postion(temp_roots,temp_parity,temp_idx)
            cur,cur_theta_map=search(initk,initm,temp_roots,temp_parity,[],counts,theta,[])
            cur=np.array(cur)
            if np.around(cur[0]-cur[-1],4)!=0:
                uncom_curve+=[cur]
                uncom_theta_map+=[cur_theta_map]
                flag2=1
            else:
                curve+=[cur]
                theta_map+=[cur_theta_map]
                mag+=np.abs(np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:])))
        if flag2:
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
            curve+=[cur]
            theta_map+=[cur_theta_map]
            mag+=np.abs(np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:])))
        return mag/(2*(np.pi*rho**2)),curve,theta_map
    def derivative(self,matched_image):
        zeta_l=matched_image
        zeta_conj=np.conj(zeta_l)
        q=self.q
        s=self.s
        theta=self.theta
        parZetaConZ=1/(1+q)*(1/(zeta_conj-s)**2+q/zeta_conj**2)
        par2ZetaConZ=-2/(1+q)*(1/(zeta_conj-s)**3+q/(zeta_conj)**3)
        par2ConZetaZ=-2/(1+q)*(1/(zeta_l-s)**3+q/(zeta_l)**3)
        de_zeta=1j*self.rho*np.exp(1j*theta)
        de2_zeta=-self.rho*np.exp(1j*theta)
        detJ=1-np.abs(parZetaConZ)**2
        de_z=(de_zeta-parZetaConZ*np.conj(de_zeta))/detJ
        de2_z=(de2_zeta-par2ZetaConZ*(np.conj(de_z)**2)-parZetaConZ*(np.conj(de2_zeta)-par2ConZetaZ*(de_z)**2))/detJ
        deXProde2X=(self.rho**2+np.imag(de_z**2*de_zeta*par2ConZetaZ))/detJ
        return deXProde2X
    def get_magnifaction(self):
        theta=np.linspace(0,2*np.pi,1000)
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        light_curve=[]
        curve_all=[]
        for i in range(trajectory_n):
            zeta_l=self.get_zeta_l(trajectory_l[i],theta)
            coff=self.get_poly_coff(zeta_l)
            roots,roots_unverify,parity=self.poly_root_c(zeta_l,coff)
            mag,curve,theta_map=self.image_match(theta,roots,parity)
            light_curve+=[mag]
            curve_all+=[curve]
        self.curve=curve_all
        return np.array(light_curve)
    def draw_anim(self,fig,axis):#given  a series of roots return picture
        ims=[]
        theta=np.linspace(0,2*np.pi,100)
        trajectory_n=self.trajectory_n
        trajectory_l=self.trajectory_l
        curve=self.curve
        for i in range(trajectory_n):
            zeta=self.to_centroid(self.get_zeta_l(trajectory_l[i],theta))
            img_root=[]
            rgb=['b','g','y','c','m']
            img2,=axis.plot(zeta.real,zeta.imag,color='r',label=str(i))
            ttl = plt.text(0.5, 1.01, i, horizontalalignment='center', verticalalignment='bottom', transform=axis.transAxes)
            for k in range(len(curve[i])):
                img1,=axis.plot(curve[i][k].real,curve[i][k].imag,color=rgb[k])
                img_root+=[img1]
            ims.append(img_root+[img2]+[ttl])
        ani=animation.ArtistAnimation(fig,ims,interval=100,repeat_delay=1000)
        writervideo = animation.FFMpegWriter(fps=30) 
        ani.save('picture/animation.mp4',writer=writervideo)
