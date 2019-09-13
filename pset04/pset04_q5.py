import dippykit as dip
import numpy as np
from sklearn import feature_extraction as feat

def load_image():
    '''Loads the airplane and Brussels Image'''
    #Image Path
    img1_path = "/home/harshbhate/Pictures/airplane_downsample_gray_square.jpg"
    img2_path = "/home/harshbhate/Pictures/brussels_downsample_gray_square.jpg"
    #Read Image
    IMG1 = dip.image_io.im_read(img1_path)
    IMG2 = dip.image_io.im_read(img2_path)
    #Convert to float
    IMG1 = dip.im_to_float(IMG1)
    IMG2 = dip.im_to_float(IMG2)
    #Normalize
    IMG1 *= 255        
    IMG2 *= 255
    #Return
    return [IMG1, IMG2]

def patch_extractor(X,B):
    '''Extract Patches of size B*B'''
    M,N = X.shape
    rows = np.arange(0,M,B)
    columns = np.arange(0,N,B)
    patches = []
    for i in columns:
        up_col = i + B
        low_col = i
        for j in rows:
            up_row = j + B
            low_row = j
            blk = X[low_col:up_col, low_row:up_row]
            patches.append(blk)
    return np.array(patches)

def reconstruct_patches(patches):
    '''Reconstruct the image from patches'''
    M = 200
    N = 200
    B = 10
    rows = np.arange(0,M,B)
    columns = np.arange(0,N,B)
    image = np.zeros((200,200))
    idx = 0
    for i in columns:
        up_col = i + B
        low_col = i
        for j in rows:
            up_row = j + B
            low_row = j
            image[low_col:up_col, low_row:up_row] = patches[idx]
            idx = idx + 1
    return image

def mean_patches(patches, k):
    '''Compute the mean'''
    recons = []
    Us = []
    M = 200
    N = 200
    vectors = np.reshape(patches, (400,100))
    vectors = np.transpose(vectors)
    for X in patches:    
        u = np.mean(X)  # You need to calculate u
        U = np.array([u]*100)
        U = np.reshape(U, (100,))
        U = np.transpose(U)
        Us.append(U)
    Us = np.transpose(np.array(Us))
    Y = vectors - Us
    Rff = (1/(M*N))*np.matmul(Y, np.transpose(Y))
    eigenvectors, eigenvalues,_ = np.linalg.svd(Rff)
    phi = eigenvectors
    phi_trans = np.transpose(eigenvectors)
    temp = np.matmul (phi, phi_trans)
    temp = np.matmul (temp, Y)
    x_recons = temp + Us
    x_recons = np.reshape(x_recons, (400,10,10))
    print (x_recons)
    return x_recons

def KLT_blocks(X, k, B):
    '''Vectorize images into blocks perform KLT and reconstruct image'''
    #X_blocks = feat.image.extract_patches_2d(X, (10,10))
    #X_hat = feat.image.reconstruct_from_patches_2d(X_blocks, (200,200))
    patches = patch_extractor(X,10)
    pat = mean_patches(patches, 1) 
    IMG = reconstruct_patches(pat)
    display_image([IMG,IMG,IMG,IMG])
    
def display_image(IMAGES):
    '''Display the Images and save'''
    IMAGES = [dip.float_to_im(IMG/255) for IMG in IMAGES]
    dip.figure()
    dip.subplot(2, 2, 1)
    dip.imshow(IMAGES[0], 'gray')
    dip.title('Airplane Original Image')
    dip.subplot(2, 2, 2)
    dip.imshow(IMAGES[1], 'gray')
    dip.title('Brussels Original Image')
    dip.subplot(2, 2, 3)
    dip.imshow(IMAGES[2], 'gray')
    dip.title('Reconstructed Airplane Image')
    dip.subplot(2, 2, 4)
    dip.imshow(IMAGES[3], 'gray')
    dip.title('Reconstructed Brussels Image')
    dip.show()


if __name__=="__main__":
    '''Main Function'''
    IMG1, IMG2 = load_image()
    KLT_blocks(IMG2,1,10)
    #display_image([IMG1,IMG2,IMG1,IMG2])
