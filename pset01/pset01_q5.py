"""
Name: Harsh Bhate
Mail: hbhate3@gatech.edu
Date: August 24, 2018
GTID: 903424029
Problem Set: 1
Problem Number: 5
About: This file finds the contour plot of a function
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

PI = 3.14

def freq_response(X,Y):
    '''Magnitude of Frequency Response H(w1,w2)'''
    Z = np.square(-4*np.sin(Y)*(1+np.cos(X)))
    Z = np.sqrt(Z)
    return Z

#Defining the space
w_1 = np.linspace(-PI,PI,100)
w_2 = np.linspace(-PI,PI,100)

W_1,W_2 = np.meshgrid(w_1,w_2)

#Computing the Frequency Respone
H = freq_response(W_1,W_2)

#Plotting
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(W_1, W_2, H, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#Plotting
plt.figure(2)
CS = plt.contour(W_1,W_2,H)
plt.clabel(CS,inline=2.3, fontsize=10)
plt.show()
