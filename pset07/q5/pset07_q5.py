import dippykit as dip
import numpy as np

#Resolution Guide
#4CIF Resolution = 704 x 480
#CIF Resolution = 352 x 240

X = dip.im_read("coatOfArms.png")
X = dip.im_to_float(X)
X = X*255

#Filter 
SIZE = 25
filterSize = (SIZE, SIZE)
minor = 5**2
mild = 25**2
severe = 100**2

minorWindow = dip.window_2d (filterSize, window_type='gaussian', variance = minor )
mildWindow = dip.window_2d (filterSize, window_type='gaussian', variance = mild )
severeWindow = dip.window_2d (filterSize, window_type='gaussian', variance = severe )

#Output of the Filter
minorX = dip.convolve2d(X, minorWindow, mode = 'same', boundary = 'wrap')
mildX = dip.convolve2d(X, mildWindow, mode = 'same', boundary = 'wrap')
severeX = dip.convolve2d(X, severeWindow, mode = 'same', boundary = 'wrap')

#Laplacian Filtering
LaplaceKernel = np.array([[0,1,0], 
                        [1,-4,1],
                        [0,1,0]],
                        dtype=np.float)

edgeMinorX = dip.convolve2d(X, LaplaceKernel, mode = 'same', boundary = 'wrap')
edgeMildX = dip.convolve2d(X, LaplaceKernel, mode = 'same', boundary = 'wrap')
edgeSevereX = dip.convolve2d(X, LaplaceKernel, mode = 'same', boundary = 'wrap')

#PSNR Calculations

PSNR_minor = dip.metrics.PSNR(dip.float_to_im(X/255), dip.float_to_im(edgeMinorX/255))
PSNR_mild = dip.metrics.PSNR(dip.float_to_im(X/255), dip.float_to_im(edgeMildX/255))
PSNR_severe = dip.metrics.PSNR(dip.float_to_im(X/255), dip.float_to_im(edgeSevereX/255))

#Displaying PSNR Output
print ("\n\tPSNR for Laplacian with minor blurring is %.2f \n" %PSNR_minor)
print ("\tPSNR for Laplacian with mild blurring is %.2f \n" %PSNR_mild)
print ("\tPSNR for Laplacian with severe blurring is %.2f \n" %PSNR_severe)

#Extended Laplacian Filtering
extLaplaceKernel = np.array([[1,1,1],
                            [1,-8,1],
                            [1,1,1]],
                            dtype=np.float)                        

extEdgeMinorX = dip.convolve2d(X, extLaplaceKernel, mode = 'same', boundary = 'wrap')
extEdgeMildX = dip.convolve2d(X, extLaplaceKernel, mode = 'same', boundary = 'wrap')
extEdgeSevereX = dip.convolve2d(X, extLaplaceKernel, mode = 'same', boundary = 'wrap')

#PSNR Calculations
PSNR_minor = dip.metrics.PSNR(dip.float_to_im(X/255), dip.float_to_im(extEdgeMinorX/255))
PSNR_mild = dip.metrics.PSNR(dip.float_to_im(X/255), dip.float_to_im(extEdgeMildX/255))
PSNR_severe = dip.metrics.PSNR(dip.float_to_im(X/255), dip.float_to_im(extEdgeSevereX/255))

#Displaying PSNR Output
print ("\n\tPSNR for extended Laplacian with minor blurring is %.2f" %PSNR_minor)
print ("\n\tPSNR for extended Laplacian with mild blurring is %.2f" %PSNR_mild)
print ("\n\tPSNR for extended Laplacian with severe blurring is %.2f" %PSNR_severe)

#Image Output
dip.figure()
dip.subplot(3, 4, 1)
dip.imshow(X, 'gray')
dip.title('Original Image', fontsize='x-small')
dip.subplot(3, 4, 2)
dip.imshow(minorX, 'gray')
dip.title('Minor Blurring', fontsize='x-small')
dip.subplot(3, 4, 3)
dip.imshow(mildX, 'gray')
dip.title('Mild Blurring', fontsize='x-small')
dip.subplot(3, 4, 4)
dip.imshow(severeX, 'gray')
dip.title('Severe Blurring', fontsize='x-small')
dip.subplot(3, 4, 5)
dip.imshow(edgeMinorX, 'gray')
dip.title('Laplacian on Minor Blurring', fontsize='x-small')
dip.subplot(3, 4, 6)
dip.imshow(edgeMildX, 'gray')
dip.title('Laplacian on Mild Blurring', fontsize='x-small')
dip.subplot(3, 4, 7)
dip.imshow(edgeSevereX, 'gray')
dip.title('Laplacian on Severe Blurring', fontsize='x-small')
dip.subplot(3, 4, 8)
dip.imshow(extEdgeMinorX, 'gray')
dip.title('Extended Laplacian on Minor Blurring', fontsize='x-small')
dip.subplot(3, 4, 9)
dip.imshow(extEdgeMildX, 'gray')
dip.title('Extended Laplacian on Mild Blurring', fontsize='x-small')
dip.subplot(3, 4, 10)
dip.imshow(extEdgeSevereX, 'gray')
dip.title('Extended Laplacian on Severe Blurring', fontsize='x-small')
dip.show()