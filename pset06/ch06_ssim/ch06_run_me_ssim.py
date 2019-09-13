import numpy as np
import dippykit as dip


# Read the image
# ============================ EDIT THIS PART =============================
im = dip.im_read("ch06_ssim/barbara.png")

# Add noise to the original image to create new images
# ============================ EDIT THIS PART =============================
im_gaussian = dip.image_noise(im, "gaussian")
im_poisson = dip.image_noise(im, "poisson")
im_salt_pepper = dip.image_noise(im, "s&p")
im_speckle = dip.image_noise(im, "speckle")

# Compute the SSIM values and images
# ============================ EDIT THIS PART =============================
mssim_gaussian, ssim_image_gaussian = dip.metrics.SSIM(im, im_gaussian)
mssim_poisson, ssim_image_poisson = dip.metrics.SSIM(im, im_poisson)
mssim_salt_pepper, ssim_image_salt_pepper = dip.metrics.SSIM(im, im_salt_pepper)
mssim_speckle, ssim_image_speckle = dip.metrics.SSIM(im, im_speckle)

dip.figure()
dip.subplot(2, 4, 1)
dip.imshow(im_gaussian, 'gray')
dip.title('Distortion type: gaussian', fontsize='x-small')
dip.subplot(2, 4, 2)
dip.imshow(im_poisson, 'gray')
dip.title('Distortion type: poisson', fontsize='x-small')
dip.subplot(2, 4, 3)
dip.imshow(im_salt_pepper, 'gray')
dip.title('Distortion type: salt & pepper', fontsize='x-small')
dip.subplot(2, 4, 4)
dip.imshow(im_speckle, 'gray')
dip.title('Distortion type: speckle', fontsize='x-small')

dip.subplot(2, 4, 5)
dip.imshow(ssim_image_gaussian, 'gray')
dip.title('SSIM Map; MSSIM={0:.2f}'
          .format(mssim_gaussian), fontsize='x-small')
dip.subplot(2, 4, 6)
dip.imshow(ssim_image_poisson, 'gray')
dip.title('SSIM Map; MSSIM={0:.2f}'
          .format(mssim_poisson), fontsize='x-small')
dip.subplot(2, 4, 7)
dip.imshow(ssim_image_salt_pepper, 'gray')
dip.title('SSIM Map; MSSIM={0:.2f}'
          .format(mssim_salt_pepper), fontsize='x-small')
dip.subplot(2, 4, 8)
dip.imshow(ssim_image_speckle, 'gray')
dip.title('SSIM Map; MSSIM={0:.2f}'
          .format(mssim_speckle), fontsize='x-small')

dip.show()

