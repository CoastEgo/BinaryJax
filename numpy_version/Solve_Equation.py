import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MulensModel import Model,caustics
from function_numpy import search,search_first_postion
import sys
sys.setrecursionlimit(10000)
np.seterr(divide='ignore', invalid='ignore')
def find_nearest(array1, array2):
    parity1=parity(array1)
    parity2=parity(array2)
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
            i=np.where(array1==value_i)[0]
            idx=np.nanargmin(np.abs(value_i-temp))
            array2[idx],array2[i]=array2[i],array2[idx]
            parity2[idx],parity2[i]=parity2[i],parity2[idx]
            temp[idx],temp[i]=temp[i],temp[idx]
            temp[i]=np.nan
    else:
        temp=np.copy(array1)
        value=array2[~np.isnan(array2.real)]
        for value_i in value:
            i=np.where(array2==value_i)[0]
            idx=np.nanargmin(np.abs(temp-value_i))
            array2[idx],array2[i]=array2[i],array2[idx]
            parity2[idx],parity2[i]=parity2[i],parity2[idx]
            temp[idx]=np.nan
    return array2/parity2
def roots_isnan(roots):
    z=roots.real
    cond=~np.isnan(z)
    return cond
def parity(z):
    de_conjzeta_z1=m1/(np.conj(z)-s)**2+m2/np.conj(z)**2
    return np.sign((1-np.abs(de_conjzeta_z1)**2))
def to_centroid(x):
    return -(np.conj(x)-delta_x)
def to_lowmass(x):
    return -np.conj(x)+delta_x
def verify(zeta_l,z):
    return  z-m1/(np.conj(z)-s)-m2/np.conj(z)-zeta_l
def poly_root(s,alpha,b,rho,sample_n,trajectory_n):#given cofficient of complex polynomial return roots of it      #coff shape:(trajectoy_n,sample_n,coff_n)
    trajectory_centroid=np.array([i*np.cos(alpha)-b*np.sin(alpha)+1j*(b*np.cos(alpha)+i*np.sin(alpha)) for i in np.linspace(-1.5,1.5,trajectory_n)]).reshape(trajectory_n,1)
    rel_centroid=np.array([rho*np.cos(k)+1j*rho*np.sin(k) for k in np.linspace(0,2*np.pi,sample_n)]).reshape(1,sample_n)
    zeta_l=to_lowmass(trajectory_centroid+rel_centroid)
    zeta_conj=np.conj(zeta_l)
    c0=s**2*zeta_l*m2**2
    c1=-s*m2*(2*zeta_l+s*(-1+s*zeta_l-2*zeta_l*zeta_conj+m2))
    c2=zeta_l-s**3*zeta_l*zeta_conj+s*(-1+m2-2*zeta_conj*zeta_l*(1+m2))+s**2*(zeta_conj-2*zeta_conj*m2+zeta_l*(1+zeta_conj**2+m2))
    c3=s**3*zeta_conj+2*zeta_l*zeta_conj+s**2*(-1+2*zeta_conj*zeta_l-zeta_conj**2+m2)-s*(zeta_l+2*zeta_l*zeta_conj**2-2*zeta_conj*m2)
    c4=zeta_conj*(-1+2*s*zeta_conj+zeta_conj*zeta_l)-s*(-1+2*s*zeta_conj+zeta_conj*zeta_l+m2)
    c5=(s-zeta_conj)*zeta_conj
    coff=np.dstack((c5,c4,c3,c2,c1,c0))
    roots=np.empty((trajectory_n,sample_n,5),dtype=complex)
    roots_unverify=np.empty((trajectory_n,sample_n,5),dtype=complex)
    roots_parity=np.zeros((trajectory_n,sample_n,5),dtype=int)
    temp=np.zeros(5,dtype=complex)
    for i in range(trajectory_n):
        for k in range(sample_n):
            temp=np.roots(coff[i,k,:])
            roots_unverify[i,k,:]=temp
            cond=np.round(verify(zeta_l[i,k],temp),4)!=0
            temp[cond]=np.nan+1j*np.nan
            roots[i,k,:]=temp
    for i in range(trajectory_n):
        for k in range(sample_n):
            if k!=0:
                root_i_re_1=root_i
            root_i=roots[i,k,:]
            if k==0:
                temp=root_i
            else:
                temp=find_nearest(root_i_re_1,root_i)
            roots[i,k,:]=temp
    roots_parity=parity(roots)
    if 1:
        np.savetxt('result/roots.txt',np.around(roots,4).reshape(sample_n*trajectory_n,5),delimiter=',',fmt='%1.4f')
        np.savetxt('result/roots_unverify.txt',np.around(roots_unverify,4).reshape(sample_n*trajectory_n,5),delimiter=',',fmt='%1.4f')
        np.savetxt('result/parity.txt',np.around(roots_parity,4).reshape(sample_n*trajectory_n,5),delimiter=',',fmt='%1.1f')
    return to_centroid(zeta_l),to_centroid(roots),to_centroid(roots_unverify),roots_parity
def magnification(roots,parity):
    mag=np.zeros(trajectory_n)
    cond=roots_isnan(roots)
    curve=[]
    for i in range(trajectory_n):
        curve_i=[]
        uncom_curve=[]
        temp_idx=[]
        temp_parity=[]
        flag=0
        flag2=0
        for k in range(5):
            parity_ik=parity[i,:,k][cond[i,:,k]]
            if len(parity_ik)==0:
                continue
            if (len(parity_ik)==sample_n) & ((np.round(roots[i,0,k]-roots[i,-1,k],4))==0):
                cur=roots[i,:,k]
                curve_i+=[cur]
                value,counts=np.unique(parity_ik,return_counts=True)
                ind=np.argmax(counts)
                parity_most=value[ind]
                mag[i]+=parity_most*np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:]))
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
                mag[i]+=np.abs(np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:])))
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
            mag[i]+=np.abs(np.sum((cur.imag[0:-1]+cur.imag[1:])*(cur.real[0:-1]-cur.real[1:])))
        curve.append(curve_i)
    return mag/(2*(np.pi*rho**2)),curve
def draw_anim(curve):#given  a series of roots return picture
    ims=[]
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
    ani.save('picture/animation.mp4',writer=writervideo)
if __name__=="__main__":
    t_0=2452848.06;t_E=61.5;q=0.05;s=0.8;alphadeg=30;b=0.1;rho=0.05;trajectory_n=100;sample_n=500;alpha=alphadeg*2*np.pi/360;delta_x=s/(1+q);m1=1/(1+q);m2=q/(1+q);posi=np.array([0,0])##m1较大 m2较小
    zeta,roots,roots_unverify,roots_parity=poly_root(s,alpha,b,rho,sample_n,trajectory_n)##cenproduce
    image_pos_x=np.round(roots.real,4);image_pos_y=np.round(roots.imag,4)
    ##### parameter set#################################################
    model_1S2L = Model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha': alphadeg})
    model_1S2L.set_magnification_methods([t_0-1.5*t_E, 'VBBL',t_0+1.5*t_E])
    times=np.linspace(t_0-1.5*t_E,t_0+1.5*t_E,trajectory_n)
    Mulens_mag=model_1S2L.get_magnification(time=times)
    uniform_mag,all_curve=magnification(roots,roots_parity)
    if 1:
        fig=plt.figure(figsize=(6,6))
        axis=plt.axes(xlim=(-2,2),ylim=(-2,2))
        axis.plot([], [], color="gold",label='Time: 0')
        model_1S2L.plot_trajectory()
        caustic_1=caustics.Caustics(q,s)
        caustic_1.plot(5000,s=0.5)
        x,y=caustic_1.get_caustics()
        x=caustic_1.critical_curve.x
        y=caustic_1.critical_curve.y
        plt.scatter(x,y,s=0.05)
        draw_anim(all_curve)
    if 1:
        plt.figure('mag')
        plt.plot(times,Mulens_mag,label='MulensModel')
        plt.plot(times,uniform_mag,label='uniform')
        plt.legend()
        plt.savefig('picture/magnification.png')
    if 1:
        plt.figure('de-mag')
        plt.plot(times,np.abs(Mulens_mag-uniform_mag)/Mulens_mag,label='$\Delta$')
        plt.legend()
        plt.savefig('picture/delta_magnification.png')