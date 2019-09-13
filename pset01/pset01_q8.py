"""
Name: Harsh Bhate
Mail: hbhate3@gatech.edu
Date: August 31, 2018
GTID: 903424029
Problem Set: 1
Problem Number: 8
"""

import dippykit as dip
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 100 
GRID_VAL = 50  

w_1 = np.arange(-GRID_VAL,GRID_VAL)
w_2 = np.arange(-GRID_VAL,GRID_VAL)
print (w_1[49])
m,n = np.meshgrid(w_1,w_2)
x = np.zeros((GRID_SIZE,GRID_SIZE))
x[np.logical_and(m<+n,n>=0)] = 1

g = dip.utilities.conv2d_grid([-50,-50],[-50,-50],x,x)
print (g)
plt.plot(g)
plt.show()