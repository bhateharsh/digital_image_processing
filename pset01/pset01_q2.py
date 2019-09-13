"""
Name: Harsh Bhate
Mail: hbhate3@gatech.edu
Date: August 24, 2018
GTID: 903424029
Problem Set: 1
Problem Number: 2
"""

import dippykit as dip
import numpy as np

picture_link = "/home/harshbhate/Pictures/cameraman.tif"
save_link_1 = "/home/harshbhate/Pictures/cameraman_add.tif"
save_link_2 = "/home/harshbhate/Pictures/cameraman_square.tif"
save_link_3 = "/home/harshbhate/Pictures/cameraman_fourier.tif"

#(c) Reading an image
X = dip.im_read(picture_link)

#(d) Converting the image to normalized floating point space
X = dip.im_to_float(X)
X *= 255

#(e) Adding Constant to Image
Y = X + 75

#(f) Renormalize the image and covert to integer
Y = dip.float_to_im(Y/255)

#(g) Writing an image to disk
dip.im_write(Y, save_link_1)

#(h) Square Intenstiy and write image to disk
Z = X**2
Z = dip.float_to_im(Z/255)
Z = dip.im_write(Z, save_link_2)

#(i) Compute FFT of X
fX = dip.fft2(X)
fX = dip.fftshift(fX)
fX = np.log(np.abs(fX))

#(j) Save and show the resulting spectrum
dip.imshow(fX)
dip.show()
