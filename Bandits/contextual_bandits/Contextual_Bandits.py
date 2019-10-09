
# coding: utf-8

# In[5]:


# %load template.py
import numpy as np
from tqdm import tqdm
from ads import UserAdvert
import matplotlib.pyplot as plt
import random

ACTION_SIZE = 3
STATE_SIZE = 4
TRAIN_STEPS = 10000  # Change this if needed
LOG_INTERVAL = 10

#baseline = 2,   learning rate = 0.01
bn = 0
lr = 0.01

def softmax(Context):
    exp_Q = np.exp(Context)
    policy = exp_Q/np.sum(exp_Q)
    return policy



def learnBandit():
    env = UserAdvert()
    rew_vec = []
    
    #Random Initialization of weights
    W = np.random.normal(0,1,size = (ACTION_SIZE,STATE_SIZE))

    for train_step in tqdm(range(TRAIN_STEPS)):
        state = env.getState()
        stateVec = state["stateVec"]
        stateId = state["stateId"]

        # ---- UPDATE code below ------j
        # Sample from policy = softmax(stateVec X W) [W learnable params]
        # policy = function (stateVec)
        
        Context = np.matmul(W,stateVec)
        policy = softmax(Context)
        action = np.random.choice(range(ACTION_SIZE),1,p=policy)
        
        #action = int(np.random.choice(range(3)))
        reward = env.getReward(stateId, int(action))
        # ----------------------------

        # ---- UPDATE code below ------
        # Update policy using reward
        #policy = [1/3.0, 1/3.0, 1/3.0]
        grad = -policy
        grad[action] +=1
        grad = np.expand_dims(grad,axis=1)
        stateVec = np.expand_dims(stateVec,axis=1)
        grad = np.matmul(grad,np.transpose(stateVec))
        
        W = W + lr*(reward-bn)*grad
        # ----------------------------

        if train_step % LOG_INTERVAL == 0:
            #print("Testing at: " + str(train_step))
            count = 0
            test = UserAdvert()
            for e in range(450):
                teststate = test.getState()
                testV = teststate["stateVec"]
                testI = teststate["stateId"]
                # ---- UPDATE code below ------
                # Policy = function(testV)
                policy = [1/3.0, 1/3.0, 1/3.0]
                Context = np.matmul(W,testV)
                policy = softmax(Context)
                # ----------------------------
                act = int(np.random.choice(range(3), p=policy))
                reward = test.getReward(testI, act)
                count += (reward/450.0)
            rew_vec.append(count)

    # ---- UPDATE code below ------
    # Plot this rew_vec list
    x= [i for i in range(int(TRAIN_STEPS/LOG_INTERVAL))]
    fig1=plt.figure(figsize=(10,7)).add_subplot(111)
    fig1.set_xlabel('epochs')
    fig1.set_ylabel('Average Returns')
    fig1.plot(x,rew_vec)
    plt.show()
    
    
    #plt.plot(x,rew_vec)
    #print(rew_vec)


if __name__ == '__main__':
    learnBandit()


# In[27]:


x = np.random.normal(0,1,size = (3,4))
y = np.random.normal(0,1,size = (1,4))
x

