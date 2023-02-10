import jax.numpy as np
def exchange(input,idx1,idx2):
    temp=input[idx1]
    input=input.at[idx1].set(input.at[idx2].get())
    input=input.at[idx2].set(temp)
    return input
def find_nearest(array1, array2):
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
            array2=exchange(array2,idx,i)
            temp=exchange(temp,idx,i)
            temp=temp.at[i].set(np.nan)
    else:
        temp=np.copy(array1)
        value=array2[~np.isnan(array2.real)]
        for value_i in value:
            i=np.where(array2==value_i)[0][0]
            idx=np.nanargmin(np.abs(temp-value_i))
            array2=exchange(array2,idx,i)
            temp=temp.at[idx].set(np.nan)
    return array2
a=np.array([ 1.8234432 +0.32878557j, np.nan+np.nan*1j,np.nan+np.nan*1j, -0.20591831-0.05379141j,        0.09574446-0.00876968j])
b=np.array([ 1.8303977 +0.31758365j,np.nan+np.nan*1j,       -0.20402198-0.05220728j,np.nan+np.nan*1j,        0.09654127-0.00885313j])
print(find_nearest(a,b))
