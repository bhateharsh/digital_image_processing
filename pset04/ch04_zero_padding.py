import dippykit as dip
import numpy as np


# Part (a)
I1 = dip.im_read('/home/harshbhate/Pictures/lena.png')  # Specify your image here
I1 = dip.im_to_float(I1) #Converting image to float
I1 *= 255 #Normalizing
# Part (b)
# Take the Fourier transform of the image
H1 = dip.fft2(I1)  # Fourier transform of I1
H1 = dip.fftshift(H1)
H1 = np.log(np.abs(H1))
print ("Shape of I1:"+str(I1.shape))
print("Shape of H1"+str(H1.shape))
# Part (c)
# Downsample the image by 2 in both directions (and take its Fourier transform)
x_scaling = 2
y_scaling = 2
sampling_matrix = np.array([[x_scaling, 0],[0, y_scaling]])
I2 = dip.sampling.resample(I1, sampling_matrix)  # Downsampled I1
H2 = dip.fft2(I2)  # Fourier transform of I2
# Part (d)
# Pad the downsampled image's spectrum (H2) with zeros and then take its
# inverse Fourier transform
H3 = np.pad(H2,(128, 128), 'constant', constant_values = (0,0))  # Zero-padded H2
I3 = np.abs(dip.ifft2(H3)) # Interpolated image
I3 = I3/(np.amax(I3))*255   #Normalizing
#Converting everything back to int and normalizing
I1 = dip.float_to_im(I1/255)
I2 = dip.float_to_im(I2/255)
I3 = dip.float_to_im(I3/255)
H2 = dip.fftshift(H2)
H2 = np.log(np.abs(H2))
H3 = np.pad(H2,(128, 128), 'constant', constant_values = (0,0))
# Plotting
dip.figure()
dip.subplot(2, 3, 1)
dip.imshow(I1, 'gray')
dip.title('Original Image')
dip.subplot(2, 3, 2)
dip.imshow(I2, 'gray')
dip.title('Downsampled Image')
dip.subplot(2, 3, 3)
dip.imshow(I3, 'gray')
dip.title('Interpolated image')
dip.subplot(2, 3, 4)
dip.imshow(H1, 'gray')
dip.title('Spectrum of Original Image')
dip.subplot(2, 3, 5)
dip.imshow(H2, 'gray')
dip.title('Spectrum of Downsampled Image')
dip.subplot(2, 3, 6)
dip.imshow(H3, 'gray')
dip.title('Spectrum of Interpolated image')
dip.show()