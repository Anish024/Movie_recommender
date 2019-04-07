import numpy as np
import math
    
num_movies = 1682
num_users = 943

ratings_count = np.zeros((num_users,1),dtype=float)
valid_users = np.zeros((0,1),dtype=float)
movies = np.zeros((num_movies,20),dtype=float)

ratings = np.zeros((num_users+1,num_movies+1),dtype=float)
ratings[0,1:] = range(1,num_movies+1)
ratings[1:,0] = range(1,num_users+1)

with open('ml-100k/u.data') as f:
    for line in f:
        line = line.split("\t")
        ratings_count[int(line[0])-1,0]+=1
        ratings[int(line[0]),int(line[1])]=int(line[2])

t=[]
for i in range(0,num_users):
    if(ratings_count[i]<60):
        t.append(i+1)

ratings = np.delete(ratings,t,0)

t=0
t = ratings[:,1:].T
np.random.shuffle(t)
ratings[:,1:] = t.T

num_users = ratings.shape[0]-1
num_movies = ratings.shape[1]-1

split_size = 50
sm=0
iteration = 5
for split in range(iteration):
    active = np.random.choice(range(num_users),split_size,replace=False)
    active_data = (ratings[1:,:])[active,:]
    active_data = np.concatenate((ratings[0,:][np.newaxis,:],active_data),axis=0)
    active_data_train = active_data[:,:int(math.ceil(0.66*1682))+1]
    active_data_test = np.concatenate(((active_data[:,0])[:,np.newaxis],active_data[:,int(math.ceil(0.66*1682))+1:]),axis=1)
    rem_data = np.delete(ratings,active+1,0)
    
    corr = np.zeros((active_data.shape[0],rem_data.shape[0]),dtype=float)
    corr[1:,0] = active_data[1:,0]
    corr[0,1:] = rem_data[1:,0]
    
    mx = np.divide(np.sum(active_data_train[1:,1:],axis=1),np.sum(active_data_train[1:,1:]>0,axis=1))
    my = np.divide(np.sum(rem_data[1:,1:],axis=1),np.sum(rem_data[1:,1:]>0,axis=1))
    
    for i in range(corr.shape[0]-1):
        for j in range(corr.shape[1]-1):
            t = 0
            u = 0
            v = 0
            for k in range(active_data_train.shape[1]-1):
                if (active_data_train[i+1,k+1]!=0 and rem_data[j+1,k+1]!=0):
                    t += (active_data_train[i+1,k+1]-mx[i]) * (rem_data[j+1,k+1]-my[j])
                    u += (active_data_train[i+1,k+1]-mx[i]) ** 2
                    v += (rem_data[j+1,k+1]-my[j]) ** 2
            corr[i+1,j+1] = (t / (math.sqrt(u*v))) if u*v else 0

    rec = (np.argsort(corr[1:,1:],axis=1))
    rec = np.fliplr(rec)
    
    pred = np.zeros(active_data_test.shape)
    pred[:,0] = active_data_test[:,0]
    pred[0,:] = active_data_test[0,:]
    
    for i in range(active_data_test.shape[0]-1):
        for j in range(active_data_test.shape[1]-1):
            t=0
            u=0
            for m in range(30):
                if ((rem_data[(rec[i,m]+1),j+active_data_train.shape[1]]!=0) and (corr[i+1,rec[i,m]+1]>=0)):
                    t += corr[i+1,rec[i,m]+1] * (rem_data[(rec[i,m]+1),j+active_data_train.shape[1]] - my[rec[i,m]])
                    u += (corr[i+1,rec[i,m]+1])
            pred[i+1,j+1] = (mx[i] + (t / u)) if u else 1

    (pred[1:,1:])[np.where((pred[1:,1:])<0)] = 1
    (pred[1:,1:])[np.where(np.multiply((pred[1:,1:]>0),(pred[1:,1:]<1)))] = 1
    (pred[1:,1:])[np.where((pred[1:,1:])>5)] = 5
    pred[1:,1:] = np.around(pred[1:,1:])
    pred[1:,1:] = np.multiply(active_data_test[1:,1:]>0,pred[1:,1:])
    
    mae = np.divide(np.sum(np.absolute(pred[1:,1:]-active_data_test[1:,1:]), axis=1),np.sum(pred[1:,1:]>0, axis=1))
    mae = np.sum(mae)/(pred.shape[0]-1)
    sm += mae;
    print("mae =",mae)
avg = sm/iteration;
print("avg =",avg)
