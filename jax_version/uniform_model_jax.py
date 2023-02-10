import jax.numpy as np
import jax
import numpy as nnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from function import search,search_first_postion
import sys
sys.setrecursionlimit(10000)
#np.seterr(divide='ignore', invalid='ignore')
class model():
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
        self.sample_n=par['sample_n']
        self.delta_x=self.s/(1+self.q)
        self.m1=1/(1+self.q)
        self.m2=self.q/(1+self.q)
        self.trajectory=self.to_centroid(self.trajectory_l())
        self.roots,self.roots_unverify,self.roots_parity=self.poly_root_c()
        self.mag,self.curve=self.image_match()
    def exchange(self,input,idx1,idx2):
        temp=input[idx1]
        input=input.at[idx1].set(input.at[idx2].get())
        input=input.at[idx2].set(temp)
        return input
    def verify(self,zeta_l,z_l):
        return  z_l-self.m1/(np.conj(z_l)-self.s)-self.m2/np.conj(z_l)-zeta_l
    def parity(self,z):
        de_conjzeta_z1=self.m1/(np.conj(z)-self.s)**2+self.m2/np.conj(z)**2
        return np.sign((1-np.abs(de_conjzeta_z1)**2))
    def to_centroid(self,x):
        return -(np.conj(x)-self.delta_x)
    def to_lowmass(self,x):
        return -np.conj(x)+self.delta_x
    def find_nearest(self,array1, array2):
        parity1=self.parity(array1)
        parity2=self.parity(array2)
        array1=array1*parity1
        array2=array2*parity2
        idx1_n=np.where(~np.isnan(array1.real),1,0).sum()
        idx2_n=np.where(~np.isnan(array2.real),1,0).sum()
        if (idx1_n==3) & (idx2_n==5):
            temp=np.copy(array2)
            value=array1[~np.isnan(array1.real)]
            for value_i in value:
                i=np.where(array1==value_i)[0][0]
                idx=np.nanargmin(np.abs(value_i-temp))
                array2=self.exchange(array2,idx,i)
                parity2=self.exchange(parity2,idx,i)
                temp=self.exchange(temp,idx,i)
                temp=temp.at[i].set(np.nan)
        else:
            temp=np.copy(array1)
            value=array2[~np.isnan(array2.real)]
            for value_i in value:
                i=np.where(array2==value_i)[0][0]
                idx=np.nanargmin(np.abs(temp-value_i))
                array2=self.exchange(array2,idx,i)
                parity2=self.exchange(parity2,idx,i)
                temp=temp.at[idx].set(np.nan)
        return array2/parity2
    def trajectory_l(self):
        alpha=self.alpha_rad
        b=self.u_0
        sample_n=self.sample_n
        trajectory_n=self.trajectory_n
        rho=self.rho
        trajectory_centroid=np.array([i*np.cos(alpha)-b*np.sin(alpha)+1j*(b*np.cos(alpha)+i*np.sin(alpha)) for i in self.times]).reshape(trajectory_n,1)
        rel_centroid=np.array([rho*np.cos(k)+1j*rho*np.sin(k) for k in np.linspace(0,2*np.pi,sample_n)]).reshape(1,sample_n)
        zeta_l=self.to_lowmass(trajectory_centroid+rel_centroid)
        return zeta_l
    def poly_coff(self):
        s=self.s
        m2=self.m2
        zeta_l=self.trajectory_l()
        zeta_conj=np.conj(zeta_l)
        c0=s**2*zeta_l*m2**2
        c1=-s*m2*(2*zeta_l+s*(-1+s*zeta_l-2*zeta_l*zeta_conj+m2))
        c2=zeta_l-s**3*zeta_l*zeta_conj+s*(-1+m2-2*zeta_conj*zeta_l*(1+m2))+s**2*(zeta_conj-2*zeta_conj*m2+zeta_l*(1+zeta_conj**2+m2))
        c3=s**3*zeta_conj+2*zeta_l*zeta_conj+s**2*(-1+2*zeta_conj*zeta_l-zeta_conj**2+m2)-s*(zeta_l+2*zeta_l*zeta_conj**2-2*zeta_conj*m2)
        c4=zeta_conj*(-1+2*s*zeta_conj+zeta_conj*zeta_l)-s*(-1+2*s*zeta_conj+zeta_conj*zeta_l+m2)
        c5=(s-zeta_conj)*zeta_conj
        coff=np.dstack((c5,c4,c3,c2,c1,c0))
        return coff
    def poly_root_c(self):
        #find and  sort the roots
        sample_n=self.sample_n
        trajectory_n=self.trajectory_n
        coff=self.poly_coff()
        zeta_l=self.trajectory_l()
        roots=np.empty((trajectory_n,sample_n,5),dtype=complex)
        roots_unverify=np.empty((trajectory_n,sample_n,5),dtype=complex)
        roots_parity=np.zeros((trajectory_n,sample_n,5),dtype=int)
        temp=np.zeros(5,dtype=complex)
        for i in range(trajectory_n):
            for k in range(sample_n):
                temp=np.roots(coff[i,k,:],strip_zeros=False)
                roots_unverify=roots_unverify.at[i,k,:].set(temp)
                cond=np.round(self.verify(zeta_l[i,k],temp),4)!=0
                temp=np.where(cond,np.nan+1j*np.nan,temp)
                roots=roots.at[i,k,:].set(temp)
        for i in range(trajectory_n):
            for k in range(sample_n):
                if k!=0:
                    root_i_re_1=roots[i,k-1,:]
                root_i=roots.at[i,k,:].get()
                if k==0:
                    temp=root_i
                else:
                    temp=self.find_nearest(root_i_re_1,root_i)
                roots=roots.at[i,k,:].set(temp)
        roots_parity=self.parity(roots)
        return self.to_centroid(roots),self.to_centroid(roots_unverify),roots_parity
    '''def roots_print(self):
        roots=self.roots
        sample_n=self.sample_n
        trajectory_n=self.trajectory_n
        roots_parity=self.parity(roots)
        nnp.savetxt('result/roots.txt',np.around(roots,4).reshape(sample_n*trajectory_n,5),delimiter=',',fmt='%1.4f')
        nnp.savetxt('result/parity.txt',np.around(roots_parity,4).reshape(sample_n*trajectory_n,5),delimiter=',',fmt='%1.1f')'''
    def image_match(self):
        sample_n=self.sample_n
        trajectory_n=self.trajectory_n
        mag=np.zeros(trajectory_n)
        roots=self.roots
        parity=self.roots_parity
        rho=self.rho
        curve=[]
        for i in range(trajectory_n):
            curve_i=[]
            uncom_curve=[]
            temp_idx=[]
            temp_parity=[]
            flag=0
            flag2=0
            for k in range(5):
                parity_ik=parity[i,:,k][~np.isnan(parity.real[i,:,k])]
                if len(parity_ik)==0:
                    continue
                if (len(parity_ik)==sample_n) & ((np.round(roots[i,0,k]-roots[i,-1,k],4))==0):
                    cur=roots[i,:,k]
                    curve_i+=[cur]
                    value,counts=np.unique(parity_ik,return_counts=True)
                    ind=np.argmax(counts)
                    parity_most=value[ind]
                    mag=mag.at[i].add(parity_most*np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:])))
                elif (len(parity_ik)==sample_n) & (np.abs(np.round(roots[i,0,k]-roots[i,-1,k],4))!=0):
                    uncom_curve+=[roots[i,:,k]]
                    flag2=1
                else:
                    flag=1
                    temp_idx+=[k]
            if flag==1:##split all roots with nan and caulcate mag
                temp_roots=roots[i,:,temp_idx].T
                temp_parity=parity[i,:,temp_idx].T
                initk,initm,counts=search_first_postion(temp_roots,temp_parity,temp_idx)
                cur=search(initk,initm,temp_roots,temp_parity,[],counts)
                cur=np.array(cur)
                if np.around(cur[0]-cur[-1],4)!=0:
                    uncom_curve+=[cur]
                    flag2=1
                else:
                    curve_i+=[cur]
                    mag=mag.at[i].add(np.abs(np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))))
            if flag2:
                cur=uncom_curve[0]
                uncom_curve_n=len(uncom_curve)-1
                while uncom_curve_n>0:
                    for k in range(1,len(uncom_curve)):
                        tail=cur[-1]
                        head=uncom_curve[k][0]
                        if np.around(tail-head,4)==0:
                            cur=np.append(cur,uncom_curve[k][1:])
                            uncom_curve_n-=1
                curve_i+=[cur]
                mag=mag.at[i].add(np.abs(np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))))
            curve.append(curve_i)
        return mag/(2*(np.pi*rho**2)),curve
    '''def draw_anim(self,fig,axis):#given  a series of roots return picture
        ims=[]
        trajectory_n=self.trajectory_n
        zeta=self.trajectory
        curve=self.curve
        for i in range(trajectory_n):
            img_root=[]
            rgb=['b','g','y','c','m']
            img2,=axis.plot(zeta[i].real,zeta[i].imag,color='r',label=str(i))
            ttl = plt.text(0.5, 1.01, i, horizontalalignment='center', verticalalignment='bottom', transform=axis.transAxes)
            for k in range(len(curve[i])):
                img1,=axis.plot(curve[i][k].real,curve[i][k].imag,color=rgb[k])
                img_root+=[img1]
            ims.append(img_root+[img2]+[ttl])
        ani=animation.ArtistAnimation(fig,ims,interval=100,repeat_delay=1000)
        writervideo = animation.FFMpegWriter(fps=30) 
        ani.save('picture/animation.mp4',writer=writervideo)'''
