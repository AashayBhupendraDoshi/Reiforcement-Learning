
# coding: utf-8

# In[24]:


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


# In[25]:


#Building the test-bed

# 10 bandit problem run for 1000 steps, 2000 times
n_bandit=2000
k=10
n_pulls=1000

#Initializing the testbed
q_true=np.random.normal(0,1,(n_bandit,k))
#Optimal arm for each run
true_opt_arms=np.argmax(q_true,1) 


# In[29]:


c = [0.01, 0.1, 1, 10]
#c = [1, 2, 5, 10]
col=['g','r','b','k']

fig1=plt.figure(figsize=(10,7)).add_subplot(111)
fig2=plt.figure(figsize=(10,7)).add_subplot(111)

fig1.set_xlabel('epochs')
fig1.set_ylabel('Average Returns')
fig2.set_xlabel('epochs')
fig2.set_ylabel('Optimal Arm Rate')



for eps in range(0,len(c)) :
    print(c[eps])
    
    Q=np.zeros((n_bandit,k))
    
    #Running all the arms once
    #Q=np.random.normal(q_true,1)
    #N=np.ones((n_bandit,k))
    #R_avg = np.array([0,np.mean(Q)])
    
    #Random Initialization
    #Q=np.random.normal(np.zeros((n_bandit,k)),1)
    N=np.ones((n_bandit,k))
    R_avg = np.array([0,np.mean(Q)])
    
    
    opt_arm_rate = np.array([])
    
    # Iterate through timesteps
    for j in tqdm(range(1,n_pulls)):
        
        Rt = np.array([])
        #optimal_pulls = np.array([])
        opt_arm_pull = 0

        # Iterate through bandit problems
        for pull in range(0,n_bandit):
            
            
            #UCB-1 action selection
            ucb_Q = Q[pull] + np.sqrt(c[eps]*np.log(j)/N[pull])
            action = np.argmax(ucb_Q)
            
            if action == true_opt_arms[pull]:
                opt_arm_pull +=1
                
                
            Reward = np.random.normal(q_true[pull,action],1)
            N[pull,action] +=1
            Q[pull,action] = Q[pull,action] + (Reward - Q[pull,action])/N[pull,action]
            
            Rt = np.append(Rt,Reward)
            
        R_avg = np.append(R_avg,np.mean(Rt))
        opt_arm_rate = np.append(opt_arm_rate,opt_arm_pull/20)
        
    #print(len(R_avg),'   ',n_pulls-1,'  ',len(opt_arm_rate))
    fig1.plot(range(0,n_pulls+1),R_avg,col[eps])
    fig2.plot(range(2,n_pulls+1),opt_arm_rate,col[eps])
                
        
        
fig1.legend(['c=0.01','c=0.1','c=1','c=10'])
fig2.legend(['c=0.01','c=0.1','c=1','c=10'])

#fig1.legend(['c=1','c=2','c=5','c=10'])
#fig2.legend(['c=1','c=2','c=5','c=10'])


#plt.savefig('UCB-1_optimal.png',bbox_inches='tight')
plt.show()

