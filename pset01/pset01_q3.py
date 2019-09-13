"""
Name: Harsh Bhate
Mail: hbhate3@gatech.edu
Date: August 24, 2018
GTID: 903424029
Problem Set: 1
Problem Number: 3
About: This file performs convolution of two signle dimension vectors
"""

import numpy as np

'''Insert System Input Here'''
x_n = [1,1,3,5,3,1,1]
h_n = [1, -1]
verbosity = True 

def conv(x_n, h_n):
    '''Function to perform convolution
        y[n] = x[n] * h[n]
        y[n] = [Sum over k] x[k]h[n-k]
        We assume the signal to begin from 0, that is, 
        the first instance of the signal is 0.
    '''
    x_n = np.array(x_n, int)
    h_n = np.array(h_n, int)
    y_n = []
    l1 = len(h_n)
    l2 = len(x_n)
    diff = abs(l1-l2)
    zero_vec = np.zeros(l1, int)
    buffered_x = np.concatenate([zero_vec , x_n, zero_vec])
    l3 = len(buffered_x)
    diff = abs(l1-l3)
    inverted_h = np.flip(h_n)
    inverted_h = np.concatenate([inverted_h, np.zeros(diff, int)])
    if verbosity:
        print ("The vector buffers, types have been sorted.")
    for n in buffered_x:
        val = 0
        if verbosity:
            print ("Current Element:",n)
            print ("x[n]", buffered_x)
            print ("h[n]", inverted_h)
        val = val + np.sum(inverted_h*buffered_x)
        if verbosity:
            print ("Value:", val)
        y_n.append(val)
        inverted_h = list(inverted_h)
        ded = inverted_h.pop()
        inverted_h.insert(0,0)
        inverted_h = np.array(inverted_h, int)

        print (y_n)

if __name__ == "__main__":
    conv(x_n,h_n)
