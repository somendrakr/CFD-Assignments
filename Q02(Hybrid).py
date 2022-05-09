#!/usr/bin/env python
# coding: utf-8

# In[492]:


#Name:-Somendra kumar
#Roll.no:-1906336
###Assignment Q2(Hybrid)


# In[523]:


import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.integrate as integr
import seaborn as sns


# In[524]:


H = 1.0
L = 20.0
num_x = 100
num_y = 100
del_x = 0.01
del_y = 0.01
k = 1.0
c = 100.0
rho = 1
T_wall = 100.0
T_A = 50.0
error = 1.0e-4
v = 0.0
converged=False
umean = 1.0


# In[525]:


x_grid = np.zeros(num_x)
y_grid = np.zeros(num_y)
y_center_grid_ = np.linspace(-0.5, 0.5, num_y)
num_sol_new = np.zeros((num_y, num_x))
num_sol_old = np.zeros((num_y, num_x))
F_e_w = np.zeros(num_y)
F_n_s = np.zeros(num_y)
D_e_w = np.zeros(num_y)
D_n_s = np.zeros(num_y)
u = np.zeros(num_y)
Pe = np.zeros(num_y)
aE = np.zeros(num_y)
aW = np.zeros(num_y)
aN = np.zeros(num_y)
aS = np.zeros(num_y)
aP = np.zeros(num_y)
v_temp = np.zeros(num_x)
T_Bulk = np.zeros(num_x)
h = np.zeros(num_x)
Nu = np.zeros(num_x)


# In[ ]:





# In[526]:


# Calculating grid points
for i in range(0, num_x):
    x_grid[i] = i * del_x
for i in range(0, num_y):
    y_grid[i] = i * del_y


# In[ ]:





# In[527]:



# Calculating the Peclet number and the velocity 
for i in range(0, num_y):
    u[i] = 1.5 * (1 - 4 * (y_center_grid_[i] ** 2))
    Pe[i] = (rho * u[i] * del_x) / k


# In[528]:





# In[529]:


# Calculating  coefficients
for i in range(0, num_y):
    F_e_w[i] = rho *c* u[i] * del_y
    F_n_s[i] = rho * c*v * del_x
    D_e_w[i] = (k * del_y) / del_x
    D_n_s[i] = (k * del_x) / del_y


# In[ ]:





# In[550]:


def Hybrid(peclet_no, a_w, a_e, a_n, a_s, ap, f_e_w, f_n_s, d_e_w, d_n_s):
    # Setting the coefficients according to the peclet_number
    for i in range(0, len(peclet_no)):
        if peclet_no[i] < -2:  # Corresponds to Upwind scheme
            a_e[i] = -f_e_w[i]  # ae
            a_w[i] = 0.0  # aw
            a_n[i] = -f_n_s[i]  # an
            a_s[i] = 0.0  # as
        elif -2 <= peclet_no[i] <= 2:  # Corresponds to CDS scheme
            a_e[i] = d_e_w[i] - f_e_w[i] / 2  # ae
            a_w[i] = d_e_w[i] + f_e_w[i] / 2  # aw
            a_n[i] = d_n_s[i] - f_n_s[i] / 2  # an
            a_s[i] = d_n_s[i] + f_n_s[i] / 2  # as
        elif peclet_no[i] > 2:  # Corresponds to Upwind scheme
            a_e[i] = 0.0  # ae
            a_w[i] = f_e_w[i]  # aw
            a_n[i] = 0.0  # an
            a_s[i] = f_n_s[i]  # as

        ap[i] = a_e[i] + a_w[i] + a_n[i] + a_s[i]  # ap


# In[552]:


Hybrid(Pe, aW, aE, aN, aS, aP, F_e_w, F_n_s, D_e_w, D_n_s)


# In[ ]:





# In[532]:



# Defining and applying the boundary conditions
def apply_bc(num_sol,num_x,num_y):
    for i in range(0, num_x):
        num_sol_new[0][i] = 100.0
        num_sol_new[num_y - 1][i] = 100.0
    for k in range(0, num_y):
        num_sol_new[k][0] = 50.0


# In[533]:


apply_bc(num_sol_new,num_x,num_y)


# In[534]:


num_sol_new


# In[535]:


iter_count=0
while not converged:
    iter_count = iter_count + 1
    
    #  Point by Point Gauss-Seidel method
    for i in range(1, num_x - 1):
        for j in range(1, num_y - 1):
            num_sol_new[j][i] = (aW[j] * num_sol_new[j][i - 1] + aN[j] * num_sol_new[j + 1][i] + aS[j] * num_sol_new[j - 1][i] + aE[j] *num_sol_new[j][i + 1]) / aP[j]

    # Checking for convergence
    error1 = 0.0  #  maximum error in each iteration
    for i in range(1, num_y - 2):
        for j in range(1, num_x - 2):
            error_abs = abs(num_sol_old[i][j] - num_sol_new[i][j])
            if  error_abs> error1:
                error1 = error_abs
                
    if error1 < error:
        converged = True

    # Updating the solution
    for i in range(0, num_y - 1):
        for j in range(0, num_x - 1):
            num_sol_old[i][j] = num_sol_new[i][j]   


# In[536]:


iter_count


# In[ ]:





# In[537]:



#  boundary condition on the end
for i in range(0, num_y - 1):
    num_sol_new[i][num_x - 1] = num_sol_new[i][num_x - 2]


# In[538]:


num_sol_new


# In[548]:


plt.imshow(num_sol_new, cmap='hot', vmin=num_sol_new.min(), vmax=num_sol_new.max(), extent=[0, 20, 0, 1],
               interpolation='bilinear', origin='lower', aspect='auto')
plt.colorbar()
plt.title('Numerical Solution after {0} iterations in hybrid scheme'.format(iter_count), fontweight="bold")
plt.xlabel('$axial-direction$', fontsize=14)
plt.ylabel('$Temperature$', fontsize=14)


# In[540]:


##a)part
for i in range(0, num_x):
    for j in range(0, num_y):
        v_temp[i] = (num_sol_new[j][i] * u[i])/(H*umean)

# the bulk mean temperature
for i in range(0, num_x):
    T_Bulk[i] = integr.trapz(y_grid, v_temp)


# In[546]:


#b) Calculating the heat transfer coefficient
for i in range(0, num_x):
    h[i] = (k * (T_wall - num_sol_new[num_y - 1][i])) / ((T_wall - T_Bulk[i]) * del_y)

# c)Calculating the Nusselt number
for i in range(0, num_x):
    Nu[i] = (h[i] * H) / k


# In[549]:


# Plotting the values of h, Nu
plt.plot(x_grid, h, label="h")
plt.plot(x_grid, Nu, label="Nu")
plt.xlabel('$x$', fontsize=15)
plt.ylabel('h and Nu', fontsize=15)
plt.legend(fontsize='large', shadow=True, loc='upper left')
plt.title('Variation of h and Nu in hybrid scheme')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




