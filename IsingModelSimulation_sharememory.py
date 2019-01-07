"""
Author: Jun-Yi Tsai(zazart)
Metropolis-Hastings Sampler is the most common Markov-Chain-Monte-Carlo (MCMC) 
"""
import numpy as np
from numba import jit 
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
#import progressbar



#9 elements
def neighbors(arr,x,y,n=3):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    return arr[:n,:n]

@jit(nopython=True, parallel=True)
def add_neispin(arr,x,y,n=3):
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    c_x = int(arr[:n,:n].shape[0]/2)
    c_y = int(arr[:n,:n].shape[1]/2)
    temp = arr[:n,:n]
    #avoid calculate spin twice!
    return temp[c_x, c_y]*temp[c_x+1, c_y] +\
            temp[c_x, c_y]*temp[c_x, c_y+1]
    

#def cal_hami(arr):
#    h = 0.01
#    interact_en=0
#    for i, col in enumerate(arr):
#        for j, elm in enumerate(col):
#            interact_en += add_neispin(arr,j,i)
#    return interact_en-h*np.sum(arr)

@jit(nopython=True)
def cal_hami_test(arr):
    h = 0.01
    interact_en=0
    for i in np.arange(1,arr.shape[1]-1):
        for j in np.arange(1,arr.shape[0]-1):
            interact_en = arr[j,i]*arr[j+1,i]+\
                        arr[j,i]*arr[j,i+1]
            
    return interact_en-h*np.sum(arr)

@jit(nopython=True)
def cal_hemi_nearest(arr):
    h = 0.01

    return -np.sum(arr*np.full((3,3),arr[int(arr.shape[0]/2),\
                               int(arr.shape[1]/2)]))\
           -h*np.sum(arr)
           
@jit(nopython=True)
def flipspin(spin):
    if(spin==1):
        return -1
    elif(spin==-1):
        return 1




#for index, t_steps in enumerate(temp):

def temperature(t_steps):
    #setting some constants
    mask = np.array([[0,1,0],\
                [1,1,1],\
                [0,1,0]],dtype=bool)
    lsize = (100,100)
    ensemble_times = 10**8
    
    #random initial condition
    arr_ini = np.random.randint(-1, high=1, size=lsize)
    arr_ini[arr_ini==0] = 1
    

    
    print("---new temp %f---"%t_steps)
    tLoop = time.time()
    arr_spin = np.copy(arr_ini)
    sum_energy = server.Value('f',0.0)
    sum_partition = server.Value('f',0.0)
    avg_energysquare = server.Value('f',0.0)
    avg_energy = server.Value('f',0.0)
    avg_mag = server.Value('f',0.0)
    #do the ensemble average
    print("start loop")
    for avg_times in range(ensemble_times):
        pos = np.random.randint(100,size=2)
        #only takeout the target spin @ (i,j)
        arr_s_ori = neighbors(arr_spin,pos[0],pos[1])
        arr_s_test = np.copy(arr_s_ori)
        if((avg_times%(ensemble_times/10000))==0):
            print("{:.f} % complete \r".format(100*avg_times/ensemble_times))
        #center of the test array
        c_x = int(arr_s_test.shape[0]/2)
        c_y = int(arr_s_test.shape[1]/2)
        arr_s_test[c_x, c_y] = flipspin(arr_s_test[c_x, c_y])
    
        #calcuate the nearby changes
        en_ori = cal_hemi_nearest(np.ma.array(arr_s_ori, mask = ~mask))
        en_test = cal_hemi_nearest(np.ma.array(arr_s_test, mask = ~mask))
        
        if(en_test-en_ori<0):
            arr_spin[pos[0],pos[1]] = flipspin(arr_spin[pos[0],pos[1]])
        elif(en_test-en_ori>0):
            p = np.exp(-(en_test-en_ori)/t_steps)
            if(np.random.ranf() > p):
                arr_spin[pos[0],pos[1]] = flipspin(arr_spin[pos[0],pos[1]])
        
        arr_edge = np.pad(arr_spin,1,'edge')
        en_avg_hami = cal_hami_test(arr_edge)     
        
        sum_partition.value += np.exp(-en_avg_hami/t_steps)
        sum_energy.value += en_avg_hami*np.exp(-en_avg_hami/t_steps)
        avg_energy.value += en_avg_hami*np.exp(-en_avg_hami/t_steps)/ensemble_times
        avg_energysquare.value += (en_avg_hami**2)*np.exp(-en_avg_hami/t_steps)/ensemble_times
        avg_mag.value += np.sum(arr_spin)/np.size(arr_spin)/ensemble_times
    
     
    #calculate the exact value under the condition
    energy = sum_energy.value/sum_partition.value
    mag = avg_mag.value
    heactcap = (avg_energysquare.value-avg_energy.value**2)/t_steps
    
    return energy, heactcap, mag
    
    tEnd = time.time()#start time
    print("It cost %.3f sec for a temp loop" % (tEnd - tLoop))

tStart = time.time()#start time

temp = np.linspace(1, 3,num=12)
#    temp = np.array([2.2,2.4,2.6,2.8])
#setting the sharing memories 
server = mp.Manager()
t_energy = np.zeros(temp.shape)
t_mag = np.zeros(temp.shape)
t_heactcap = np.zeros(temp.shape)

with mp.Pool(4) as pool:
    results = pool.map(temperature, temp)

tEnd = time.time()#start time
print("It cost %.3f sec" % (tEnd - tStart))

for index, res in enumerate(results):
    t_energy[index] = res[0]
    t_heactcap[index] = res[1]
    t_mag[index] = res[2]


fig, ax = plt.subplots(3)
ax[0].plot(temp, t_energy)
ax[0].set_ylabel('energy')
ax[1].plot(temp, t_heactcap)
ax[1].set_ylabel('heactcapacity')
ax[2].plot(temp, t_mag)
ax[2].set_ylabel('magnitization')
fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)    
#        
#        for n in np.arange(1e1):
#            #random position to flip
#            pos = np.random.randint(101,size=2)   
#            spin = arr_spin[pos[0],pos[1]]
#            arr_s_ori = neighbors(arr_spin,pos[0],pos[1])
#            arr_s_test = np.copy(arr_s_ori)
#            
#            #center of the test array
#            c_x = int(arr_s_test.shape[0]/2)
#            c_y = int(arr_s_test.shape[1]/2)
#            arr_s_test[c_x, c_y] = flipspin(arr_s_test[c_x, c_y])
#    
#            
#            en_ori = cal_hemi_nearest(np.ma.array(arr_s_ori, mask = ~mask))
#            en_test = cal_hemi_nearest(np.ma.array(arr_s_test, mask = ~mask))
#            
#            
#            if(en_test-en_ori<0):
#                arr_spin[pos[0],pos[1]] = flipspin(arr_spin[pos[0],pos[1]])
#            elif(en_test-en_ori>0):
#                p = np.exp(-(en_test-en_ori)/temp)
#                if(np.random.ranf() > p):
#                    arr_spin[pos[0],pos[1]] = flipspin(arr_spin[pos[0],pos[1]])
#            
#            
#            en_avg_hami = cal_hami(arr_spin)
#            partition += np.exp(-en_avg_hami/temp)
#            avg_energy += en_avg_hami*np.exp(-en_avg_hami/temp)
#            avg_energysquare += (en_avg_hami**2)*np.exp(-en_avg_hami/temp)
#            avg_mag = np.sum(arr_spin)/np.size(arr_spin)
#            
##plt.contour(result)
#    
##periodic boundary condition
##E = 
##np.random.randint(-1, high=1, size=1)
#temp=np.zeros(2)
#test = []
#loop=100000
#for i in range(loop):
#    test.append(np.random.ranf())
##    temp += np.random.randint(101,size=2)/loop
#print(test)
##    
#@jit(nopython=True)
#def sum2d(arr):
#    M, N = arr.shape
#    result = 0.0
#    for i in range(M):
#        for j in range(N):
#            result += arr[i,j]
#    return result
#
#a = np.arange(99999999).reshape(33333333,3)
#tStart = time.time()
#print(sum2d(a))
#tEnd = time.time()
#print("It cost %.3f sec" % (tEnd - tStart))
