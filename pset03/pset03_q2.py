"""
Author: Harsh Bhate
GTID: 903424029
Homework #: 3
Question #: 2
Date: 18 Sept 2018
"""

import argparse
import dippykit as dip
import numpy as np
import os
import sys
import pdb

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_image(args, convert_to_float = True, normalize = True):
    '''Loading the image and converting to float'''
    if args.verbose:
        msg1 = bcolors.OKBLUE+"Loading the image, converting to float and normalizing"+bcolors.ENDC
        msg2 = bcolors.OKBLUE+"Loading the image, converting to float"+bcolors.ENDC
        msg3 = bcolors.OKBLUE+"Loading the image"+bcolors.ENDC
        if (convert_to_float and normalize):
            print (msg1)
        elif (convert_to_float and not normalize):
            print (msg2)
        else:
            print (msg3)
    if (os.path.exists(args.path)):
        X = dip.image_io.im_read(args.path)
    else:
        print (bcolors.FAIL+"File Path not found, aborting!"+bcolors.ENDC)
        sys.exit()
    if (convert_to_float):
        X = dip.im_to_float(X)
    if (convert_to_float and normalize):
        X *= 255
    return X

def display_image(args, IMAGES):
    '''Display a float image and save'''
    if args.verbose:
        msg = bcolors.OKBLUE+"Displaying and saving the image"+bcolors.ENDC
        name = "lena_interp.png"
    IMAGES = [dip.float_to_im(IMG/255) for IMG in IMAGES]
    dip.figure()
    dip.subplot(2, 2, 1)
    dip.imshow(IMAGES[0], 'gray')
    dip.title('Original Image')
    dip.subplot(2, 2, 2)
    dip.imshow(IMAGES[1], 'gray')
    dip.title('Interpolated Image')
    dip.subplot(2, 2, 3)
    dip.imshow(IMAGES[2], 'gray')
    dip.title('Reconstructed Image')
    dip.subplot(2, 2, 4)
    dip.imshow(IMAGES[3], 'gray')
    dip.title('Difference Image')
    dip.show()

def M_matrix(args):
    '''Generate the resampling matrix based on the interpolation'''
    if args.verbose:
        msg = bcolors.OKBLUE+"Computing the M matrix based on input parameters"+bcolors.ENDC
        print (msg)
    theta = np.array([args.rot], dtype = np.float)
    rotation_matrix = np.array([[np.cos(theta)[0], -np.sin(theta)[0]],         
                    [np.sin(theta)[0], np.cos(theta)[0]]], dtype = np.float)
    scaling_matrix = np.array([[args.v_zoom, 0],[0, args.h_zoom]]   #The scaling matrix is changed on purpose to account for the error in resampling matrix
                    , dtype=np.float)
    return np.matmul(rotation_matrix,scaling_matrix) 

def L_matrix(M_matrix, args):
    '''Returns the inverse 
    of M matrix'''
    if args.verbose:
        msg = bcolors.OKBLUE+"Computing the inverse of the M Matrix"+bcolors.ENDC
        print (msg)
    return np.linalg.inv(M_matrix)

def bicubic_interpolation(args, X, rs, inverse=False):
    '''Generate the bicubic interpolated image'''
    if args.verbose:
        msg = bcolors.OKBLUE+"Generating the interpolated image"+bcolors.ENDC
        print (msg)
    if inverse:
        X_cap = dip.sampling.resample(X, rs, interp='bicubic',crop=True, crop_size=(512,512))
    else:    
        X_cap = dip.sampling.resample(X, rs, interp='bicubic')
    return X_cap

def difference (args, IMG_cap):
    '''Find the difference between the original and reconstructed signal'''
    if args.verbose:
        msg = bcolors.OKBLUE+"Finding the difference between the images"+bcolors.ENDC
        print (msg)
    IMG_org = load_image(args)
    return np.abs(IMG_org - IMG_cap)

def metrics(args, IMG_cap):
    '''Perform Metric Calculation'''
    if args.verbose:
        msg = bcolors.OKBLUE+"Finding the difference between the images"+bcolors.ENDC
        print (msg)
    IMG_org = load_image(args)
    PSNR = dip.metrics.PSNR(IMG_cap, IMG_org, 255)
    SSIM = dip.metrics.SSIM(IMG_cap, IMG_org)
    msg1 = bcolors.OKGREEN+"PSNR: "+str(PSNR)+" dB"+bcolors.ENDC
    msg2 = bcolors.OKGREEN+"SSIM: "+str(SSIM[0])+bcolors.ENDC
    print (msg1)
    print (msg2)

def arg_correction(args):
    '''Correct the args in case of shrinking'''
    PI = 3.14
    if args.verbose:
        msg = bcolors.OKBLUE+"Checking the arguments"+bcolors.ENDC
    if args.h_zoom < 0:
        args.h_zoom = -args.h_zoom
    else:
        args.h_zoom =(1.0/args.h_zoom)
    if args.v_zoom < 0:
        args.v_zoom = -args.v_zoom
    else:
        args.v_zoom = (1.0/args.v_zoom)
    args.rot = ((args.rot*PI)/180.0) 
    return args
    
def parser():
    '''Parsing the input Argument'''
    img_path = "/home/harshbhate/Pictures/lena.png"
    parser = argparse.ArgumentParser(description="Bicubic Interpolation of an Image")
    parser.add_argument("-v","--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-p","--path",
                        help="Enter the path to the image. Default is the lenna image",
                        default="/home/harshbhate/Desktop/cameraman.png")
    parser.add_argument("-s","--save",
                        help="Enter the path to the image. Default is the lenna image",
                        default="/home/harshbhate/Pictures/")
    parser.add_argument("h_zoom", 
                        help="Enter the amount of zoom along the rows. Enter values between 0 and 1 for shrink.",
                        type=float)
    parser.add_argument("v_zoom", 
                        help="Enter the amount of zoom along the columns. Enter values between 0 and 1 for shrink.",
                        type=float)
    parser.add_argument("rot",
                        help="Enter the amount of rotation in degree. Positive for counter-clockwise, negative for clockwise",
                        type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    '''Main'''
    #Declaring the Argument Parser
    args = parser()
    args = arg_correction(args)
    #Loading the image
    IMG = load_image(args)
    #Computing the Resampling matrix based on input argument
    M = M_matrix(args)
    #Performing Bicubic Interpolation
    IMG_cap = bicubic_interpolation(args, IMG, M)
    #Computing the inverse of Resampling matrix
    L = L_matrix(M, args)
    #Reconstructing the Image
    IMG_recons = bicubic_interpolation(args, IMG_cap, L, True)
    #Finding the difference between the original and reconstructed image
    IMG_diff = difference(args, IMG_recons)
    #Running metrics between original and reconstructed iamge
    metrics(args, IMG_recons)
    #Displaying the images
    IMAGES = [IMG, IMG_cap, IMG_recons, IMG_diff]    
    display_image(args, IMAGES)