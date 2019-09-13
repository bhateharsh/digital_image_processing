import dippykit as dip
import numpy as np
from matplotlib import pyplot as plt
import sys

np.set_printoptions(threshold=np.nan)

#Part 2(a)
def read_image(img_path):
    '''Function to read image, convert to float and normalize'''
    X = dip.image_io.im_read(img_path)
    X = dip.im_to_float(X)
    X *= 255
    return X

#Part 2(b)
def floater(X):
    return dip.float_to_im(X/255)

def show_image(IMG, IMG_DCT, IMG_DIFF, IMG_RECONS):
    '''Function to display image'''
    IMG = floater(IMG)
    IMG_DCT = floater(IMG_DCT)
    IMG_DIFF = floater(IMG_DIFF)
    IMG_RECONS = floater(IMG_RECONS)
    
    dip.figure()
    dip.subplot(2, 2, 1)
    dip.imshow(IMG, 'gray')
    dip.title('Grayscale Image', fontsize='x-small')
    
    dip.subplot(2, 2, 2)
    dip.imshow(IMG_DCT, 'gray')
    dip.title('DCT of Image', fontsize='x-small')
    
    dip.subplot(2, 2, 3)
    dip.imshow(IMG_DIFF)
    dip.colorbar()
    dip.title('Error image', fontsize='x-small')
    
    dip.subplot(2, 2, 4)
    dip.imshow(IMG_RECONS, 'gray')
    dip.title('Reconstructed Image', fontsize='x-small')
    
    dip.show()

#Part 2(c)
def energy(X):
    '''Compute the energy of image'''
    return np.power(np.linalg.norm(X, 'fro'),2)

#Part 2(d)
def dct(X):
    '''Compute 2-D DCT of 8x8 non-overlaping blocks'''
    block_size = (8,8)
    return dip.block_process(X, dip.dct_2d , block_size)

#Part 2(e)
def inverse_dct(X):
    '''Compute the inverse 2-D DCT of 8x8 non-overlapping blocks'''
    block_size = (8,8)
    return dip.block_process(X, dip.idct_2d , block_size)

#Part 2(g)
def error_image(X,Y):
    '''Return the difference between the original image and difference image'''
    return np.abs(X-Y)

#Part 2(h)
def masking(coeff):
    '''Return the inverse DCT with the first 15 coefficients only'''
    block_size = (8,8)
    mask = np.zeros(block_size)
    idx = dip.utilities.zigzag_indices(mask.shape, coeff)
    mask[idx] = 1
    return mask

def save(X):
    '''Save the blocks'''
    np.save('sub_blocks/saved.npy',X)
    sys.exit()

def inverse_dct_reduced_coeff(X, coeff):
    '''Return the inverse DCT with reduced Coeff'''
    mask = masking(coeff)
    block_size = (8,8)
    masker = lambda X: np.multiply(X,mask)
    dct_recons = dip.block_process(X, masker , block_size)
    return dip.block_process(dct_recons, dip.idct_2d , block_size)

#Part 2(l)
def energy_plot(X):
    '''Plot the engery'''
    coeffs = np.linspace(1,64,64, dtype=np.int)
    energies = []
    for i in coeffs:
        energies.append(energy(inverse_dct_reduced_coeff(X,i)))
    plt.plot(coeffs, energies)
    plt.autoscale(enable=True)
    plt.title('Energy vs Nos of DCT Coefficients')
    plt.show()

if __name__=="__main__":
    '''Main Function'''
    image_path = "/home/harshbhate/Codes/DIP/images/lena_gray.png"
    IMG = read_image(image_path)
    E = energy(IMG)
    print ("E : "+str(E))
    IMG_dct = dct(IMG)
    IMG_hat_1 = inverse_dct(IMG_dct)
    E_e = energy(IMG_hat_1)
    print ("Ee: "+str(E_e))
    IMG_hat_2 = inverse_dct_reduced_coeff(IMG_dct,15)
    E_h = energy(IMG_hat_2)
    IMG_diff = error_image(IMG, IMG_hat_2)
    print ("Eh: "+str(E_h))
    show_image(IMG, IMG_dct, IMG_diff ,IMG_hat_2)
    energy_plot(IMG_dct)