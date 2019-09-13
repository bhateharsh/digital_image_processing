import dippykit as dip
import numpy as np
from skimage.restoration import denoise_bilateral as bilateral
from skimage.filters import gaussian

#Loading the image and assigning to f
f = dip.im_read("cameraman.tif")
f = dip.im_to_float(f)

#Assigning the noise to g
SIGMA = 20/255
g = dip.image_noise(f, mode='gaussian', mean=0, var=(SIGMA**2))


#Finding the Gaussian Noise
LEN, WIDTH = g.shape
SIZE = 6
sigma = 1/255
TEENY = 0.0000000000001
gaussianImage = np.zeros(g.shape)
# gaussianImage = gaussian(g,\
#                         sigma=1.0,\
#                         mode='wrap',\
#                         multichannel=False)
for n in range(LEN):
    for m in range(WIDTH):
        coeffSum = 0
        outputSum = 0
        for k in range(SIZE):
            for l in range(SIZE):
                xIndex = n - k
                yIndex = m - l
                if (xIndex < 0):
                    xIndex = LEN + xIndex 
                if (yIndex < 0):
                    yIndex = WIDTH + yIndex
                coeff = (k**2+l**2)/(2*(sigma**2))
                h_kl = np.exp(-coeff)
                coeffSum = coeffSum + h_kl
                outputSum = outputSum + h_kl*g[xIndex,yIndex]
        gaussianImage[n,m] = (outputSum+TEENY)/(coeffSum+TEENY)
#Displaying the Images
dip.figure()
dip.subplot(1, 5, 1)
dip.imshow(f, 'gray')
dip.title('Original Image', fontsize='x-small')
dip.subplot(1, 5, 2)
dip.imshow(g, 'gray')
dip.title('Noised Image', fontsize='x-small')
dip.subplot(1, 5, 3)
dip.imshow(gaussianImage, 'gray')
dip.title('Gaussian Filtering', fontsize='x-small')
#Printing the MSE
mseGaussian = dip.metrics.MSE(f*255, gaussianImage*255)
print ("\n\tThe MSE for Gaussian Transform is %.2f.\n"%mseGaussian)

#Computing the Sigma Filter
SIZE = 6
p = 40/255
TEENY = 0.0000000000001
sigmaImage = np.zeros(g.shape)
for n in range(LEN):
    for m in range(WIDTH):
        coeffSum = 0
        outputSum = 0
        for k in range(SIZE):
            for l in range(SIZE):
                xIndex = n - k
                yIndex = m - l
                if (xIndex < 0):
                    xIndex = LEN + xIndex 
                if (yIndex < 0):
                    yIndex = WIDTH + yIndex
                coeff = (g[xIndex,yIndex]**2 - g[n,m])**2
                coeff = coeff/(2*(p**2))
                h_kl = np.exp(-coeff)
                coeffSum = coeffSum + h_kl
                outputSum = outputSum + h_kl*g[xIndex,yIndex]
        sigmaImage[n,m] = (outputSum+TEENY)/(coeffSum+TEENY)
# Displaying the Images
dip.subplot(1, 5, 4)
dip.imshow(sigmaImage, 'gray')
dip.title('Sigma Filtering', fontsize='x-small')
#Printing the MSE
mseSigma = dip.metrics.MSE(f*255, sigmaImage*255)
print ("\n\tThe MSE for Sigma Transform is %.2f.\n"%mseSigma)


                           
#Computing the Bilateral Filter
SIZE = 12
p = 50.0/255
sigma = 2/255
TEENY = 0.0000000000001
bilateralImage = np.zeros(g.shape)

# bilateralImage = bilateral(g,\
#                  win_size=SIZE, \
#                 sigma_spatial=1.0,\
#                 mode='symmetric',\
#                 multichannel=False)

for n in range(LEN):
    for m in range(WIDTH):
        coeffSum = 0
        outputSum = 0
        for k in range(SIZE):
            for l in range(SIZE):
                xIndex = n - k
                yIndex = m - l
                if (xIndex < 0):
                    xIndex = LEN + xIndex 
                if (yIndex < 0):
                    yIndex = WIDTH + yIndex
                coeff_1 = (g[xIndex,yIndex]**2 - g[n,m])**2
                coeff_1 = coeff_1/(2*(p**2))
                coeff_2 = (k**2+l**2)/(2*(sigma**2))
                h_kl = np.exp(-coeff_1)*np.exp(-coeff_2)
                coeffSum = coeffSum + h_kl
                outputSum = outputSum + h_kl*g[xIndex,yIndex]
        bilateralImage[n,m] = (outputSum+TEENY)/(coeffSum+TEENY)
#Printing the MSE
mseBilateral = dip.metrics.MSE(f*255, bilateralImage*255)
print ("\n\tThe MSE for Bilateral Transform is %.5f.\n"%mseBilateral)
# Displaying the Images
dip.subplot(1, 5, 5)
dip.imshow(bilateralImage, 'gray')
dip.title('Bilateral Filtering', fontsize='x-small')
dip.show()

