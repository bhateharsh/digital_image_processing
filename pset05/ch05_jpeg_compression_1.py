import dippykit as dip
import numpy as np
import cv2
import pdb

img_path = "/home/harshbhate/Codes/DIP/images/lena.png"
# STEP 1: Loading the image
# ============================ EDIT THIS PART =================================
im = dip.image_io.im_read(img_path)

# STEP 2: Converting to YCrCb
# ============================ EDIT THIS PART =================================
im = dip.utilities.rgb2ycbcr(im) # HINT: Look into dip.rgb2ycbcr

# STEP 3: Keep only the luminance channel
# ============================ EDIT THIS PART =================================
im = im[:,:,0]

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.figure()
dip.subplot(2, 2, 1)
dip.imshow(im, 'gray')
dip.title('Grayscale image', fontsize='x-small')

# STEP 4: Calculating the image entropy
# ============================ EDIT THIS PART =================================
H = dip.metrics.entropy(im)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("Entropy of the grayscale image = {:.2f} bits/pixel".format(H))

# STEP 5: Coding of the original image
# ============================ EDIT THIS PART =================================
byte_seq, _,_,_ = dip.coding.huffman_encode(im.flatten())
l,b = im.shape
im_bit_rate = (len(byte_seq)*8.0)/float(l*b)
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("Bit rate of the original image = {:.2f} bits/pixel"
      .format(im_bit_rate))

# STEP 6: Subtract 127
# ============================ EDIT THIS PART =================================
# Change im to a float for computations
im = im.astype(float)
im = im - 127.0
# STEP 7: Block-wise DCT
block_size = (8, 8)
# ============================ EDIT THIS PART =================================
im_DCT = dip.utilities.block_process(im, dip.dct_2d, block_size)
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.subplot(2, 2, 3)
dip.imshow(im_DCT, 'gray')
dip.title("Block-wise DCT coefficients - Blocksize = {}x{}"
          .format(*block_size), fontsize='x-small')

c = 5
Q_table = dip.JPEG_Q_table_luminance
      
# STEP 8: Quantization
def step8(X):
      '''Apply X/(cQ), where X is a subblock'''
      c = 5
      Q_table = dip.JPEG_Q_table_luminance
      denominator = c*Q_table
      v = np.round(X/denominator).astype(int)#   v.astype(is
    #   pdb.set_trace()
      return v
# ============================ EDIT THIS PART =================================
im_DCT_quantized = dip.utilities.block_process(im_DCT, step8, block_size)
# im_DCT_quantized = dip.float_to_im(np.array(im_DCT_quantized))
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.subplot(2, 2, 4)
dip.imshow(im_DCT_quantized, 'gray')
dip.title('Quantized DCT coefficients: c={}'.format(c), fontsize='x-small')

# STEP 9: Entropy Coding
q_bit_stream, q_bit_stream_length, q_symbol_code_dict, _ = \
        dip.huffman_encode(im_DCT_quantized.flatten())#.reshape(-1))
# ============================ EDIT THIS PART =================================
q_bit_rate = q_bit_stream_length/float(l*b)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("Bit rate of the compressed image = {:.2f} bits/pixel"
      .format(q_bit_rate))

# STEP 10: Saving the bitstream to a binary file
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
bit_stream_file = open("CompressedSunset.bin", "wb")
q_bit_stream.tofile(bit_stream_file)
bit_stream_file.close()

# STEP 11-i: Read the binary file
# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
bit_stream_file = open("CompressedSunset.bin", "rb")
q_bit_stream = np.fromfile(bit_stream_file, dtype='uint8')
bit_stream_file.close()

# STEP 11-ii: Decoding
# ============================ EDIT THIS PART =================================
im_DCT_quantized_decoded = dip.huffman_decode(q_bit_stream,
        q_symbol_code_dict)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
im_DCT_quantized_decoded = im_DCT_quantized_decoded[:im.size]
im_DCT_quantized_reconstructed = im_DCT_quantized_decoded.reshape(im.shape)

# STEP 12: Dequantization
# ============================ EDIT THIS PART =================================
def step12(X):
      '''Apply X*(cQ), where X is a subblock'''
      c = 5
      Q_table = dip.JPEG_Q_table_luminance
      numerator = c*Q_table 
      return np.round(np.multiply(X,numerator)).astype(int)

im_DCT_reconstructed = dip.utilities.block_process(im_DCT_quantized_reconstructed, step12, block_size)  

# STEP 13: Inverse DCT
# ============================ EDIT THIS PART =================================
im_reconstructed = dip.utilities.block_process(im_DCT_reconstructed, dip.transforms.idct_2d, block_size)

# STEP 14: Add 127 to every pixel
# ============================ EDIT THIS PART =================================
im_reconstructed = im_reconstructed + 127

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
dip.subplot(2, 2, 2)
dip.imshow(im_reconstructed, 'gray')
dip.title('Reconstructed image', fontsize='x-small')

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
im = im + 127

# STEP 15: Calculating MSE and PSNR
# ============================ EDIT THIS PART =================================
MSE = dip.metrics.MSE(im, im_reconstructed)
PSNR = dip.metrics.PSNR(im_reconstructed, im, 256)

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
print("MSE = {:.2f}".format(MSE))
print("PSNR = {:.2f} dB".format(PSNR))

dip.show()

