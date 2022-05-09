#!/usr/bin/env python
# coding: utf-8

# In[84]:


# ============== Program begins here ==================================
import numpy as np
import math as m
import matplotlib.pyplot as plt


# In[85]:


# finite volume method for 5 grid points 
nodes = 5                  # number of grid points
L=0.02                     # length of plate in m
k=0.5                      #thermal conductivity of plate
q=1000000                     #uniform heat generation
A=1                          #considering cross-sectional area as 1 m2
Ta=100                       #temperature at face A
Tb=200                        #temperature at face B
del_x    = L/(nodes)        # grid size 
gridpoint    = np.zeros(nodes+2)        # grid points
T_exact   = np.zeros(nodes+2)        #exact solution
T_est    = np.zeros(nodes+2)        # soltion by TDMA method
ae=np.zeros(nodes)     # coeffcients of the equation
aw=np.zeros(nodes)     # coeffcients of the equation
sp=np.zeros(nodes)     # coeffcients of the equation
ap=np.zeros(nodes)     # coeffcients of the equation
su=np.zeros(nodes)     # coeffcients of the equation


# In[86]:


# location of grid points
for i in range(0, len(gridpoint)):
    if(i==0):
        gridpoint[i]=0.00
    elif i==nodes+1:
        gridpoint[i]=0.02
    elif i==1:
        gridpoint[i] = gridpoint[i-1] + del_x/2
    else:
        gridpoint[i] = gridpoint[i-1] + del_x
        


# In[88]:



#calculation of coeffcients

for i in range(0,nodes):
    if  i==0:
        ae[i]=k*A/del_x
        aw[i]=0
        sp[i]=-2*k*A/del_x
        su[i]=q*A*del_x + (2*k*A*Ta)/del_x
        ap[i]=aw[i]+ae[i]-sp[i]
    elif  i==nodes-1:
        ae[i]=0
        aw[i]=k*A/del_x
        sp[i]=-2*k*A/del_x
        su[i]=q*A*del_x + (2*k*A*Tb)/del_x
        ap[i]=aw[i]+ae[i]-sp[i]
        
    else:
        ae[i]=k*A/del_x
        aw[i]=k*A/del_x
        sp[i]=0
        su[i]=q*A*del_x 
        ap[i]=aw[i]+ae[i]-sp[i]
    


# In[89]:





# In[90]:


# function to apply boundary condition
def apply_bc(u,node):
    u[0] = 100          #temp at face A
    u[nodes+1] = 200    #temp at face B
    


# In[91]:


apply_bc(T_exact,nodes)
apply_bc(T_est,nodes)
T_exact


# In[92]:


# Diagonal elements of system matrix
d    = np.zeros(nodes)        # main diagonal elements
u    = np.zeros(nodes)        # upper diagonal
l    = np.zeros(nodes)        # lower diagonal


# In[93]:


# RHS of the discretized linear system of equation
f    = np.zeros(nodes)


# In[94]:


for i in range(0,nodes):
    f[i]=su[i]


# In[95]:


#intilization of values for matrix
for i in range(0,nodes ):
    if i==0:
        d[i] = ap[i]
        l[i] = -ae[i]
        u[i] = -aw[i+1]
        
    elif i==nodes-1:
        d[i] = ap[i]
        l[i] = -ae[i-1]
        u[i] = -aw[i]
        
    else:
        d[i] = ap[i]
        l[i] = -ae[i]
        u[i] = -aw[i]
        
    


# In[96]:


#algorithm to solve the system of linear equation using TDMA
def Thomas_Algorithm(num, d, u, l, R, sol):
    d1    = np.zeros(num)
    r1    = np.zeros(num)
    d1[0] = d[0]
    r1[0] = R[0]
    for i in range(1, num):
        d1[i] = d[i] - l[i]*u[i-1]/d1[i-1]
        r1[i] = R[i] - r1[i-1]*l[i]/d1[i-1]
        
    sol[num]=r1[num-1]/d1[num-1]
    
    for i in range(len(d)-1,0,-1):
        sol[i] = (r1[i-1]-u[i-1]*sol[i+1])/d1[i-1]




# In[97]:


#calculation of estimated solution
Thomas_Algorithm(nodes,d,u,l,f,T_est)

#calculation o fexact solution 
for i in range(1,nodes+1):
    T_exact[i]=Ta+(((Tb-Ta)/L)+0.5*q*(L-gridpoint[i])/k)*gridpoint[i]


# In[98]:


T_est


# In[103]:


#plot of result
plt.plot(gridpoint,T_est,'b-o')
plt.plot(gridpoint,T_exact,'r-*')
plt.xlabel('x(in m)')
plt.ylabel('T(°C)')
plt.legend(["estimated sol ","exact sol"], loc ="lower right")
plt.show()


# In[100]:


# solution obtained by changing the no of grid points to 8 and 16
T_est8=[100.  , 131.25, 181.25, 218.75, 243.75, 256.25, 256.25, 243.75,
       218.75, 200.    ]
gridpoint8=[0.     , 0.00125, 0.00375, 0.00625, 0.00875, 0.01125, 0.01375,
       0.01625, 0.01875, 0.02 ]
gridpoint16=[0.      , 0.000625, 0.001875, 0.003125, 0.004375, 0.005625,
       0.006875, 0.008125, 0.009375, 0.010625, 0.011875, 0.013125,
       0.014375, 0.015625, 0.016875, 0.018125, 0.019375, 0.02  ]
T_est16=[100.   , 115.625, 143.75 , 168.75 , 190.625, 209.375, 225.   ,
       237.5  , 246.875, 253.125, 256.25 , 256.25 , 253.125, 246.875,
       237.5  , 225.   , 209.375, 200.]


# In[107]:


#plot of comparision of result in 8 and 16 grid points
plt.plot(gridpoint,T_est,'b-o')
plt.plot(gridpoint8,T_est8,'r-*')
plt.plot(gridpoint16,T_est16,'g-+')
plt.xlabel('x(in m)')
plt.ylabel('T(°C)')
plt.legend(["est sol for 5 grid point ","est sol for 8 grid point","est sol for 16 grid point"], loc ="lower right")
plt.show()


# In[ ]:





# In[ ]:




