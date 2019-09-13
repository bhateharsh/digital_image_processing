import dippykit as dip
import numpy as np
import pdb

# Step 1: Defining the image
X = np.array([[1, 1, 1, 1, 1],
              [1, 1, 2, 2, 2],
              [0, 2, 1, 0, 0],
              [0, 2, 1, 0, 1]])


M, N = X.shape  # Reading the dimensions of the image
msg = "Shape of X: "

# Step 2: Calculating the mean vector
# ============================ EDIT THIS PART =================================
u = np.mean(X, axis=-1)  # You need to calculate u
U = np.vstack([u,u,u,u,u])
U = np.transpose(U)

# Step 3: Subtracting the mean
# ============================ EDIT THIS PART =================================
Y = X-U  # You need to calculate Y

# Step 4: Calculating the autocorrelation/covariance matrix
# ============================ EDIT THIS PART =================================
Ry = np.matmul(Y,np.transpose(Y))/float(N)  # You need to calculate Ry

# Step 5: Finding the eigenvectors
#pdb.set_trace()
eigenvectors, eigenvalues,_ = np.linalg.svd(Ry)
#e,w = np.linalg.eig(Ry)
#V = np.array(eigenvectors)
V = np.transpose(eigenvectors)
#V = np.transpose(w)
# ============================ EDIT THIS PART =================================

# Calculate the eigenvectors and put them in as columns of the matrix V.
# You may use the function "np.linalg.svd" which is the same as
# "np.linalg.eig" for positive semi-definite matrices but it orders the
# vectors in a descending order of the eigenvalues


# Step 6: Define the transformation matrix
# ============================ EDIT THIS PART =================================
A = np.transpose(V)  # Define A

# STEP 7: Calculating the KLT
# ============================ EDIT THIS PART =================================
Z = np.dot(A, Y)  # Calculate the KLT of the X


# STEP 7: Verification of results:
# ============================ EDIT THIS PART =================================
Rz = np.matmul(Z, np.transpose(Z))/float(N)

# Step 8: Inverse Transform:
# ============================ EDIT THIS PART =================================
X_rec = np.matmul(np.transpose(A), Z) + U

# ===================!!!!! DO NOT EDIT THIS PART !!!!!=========================
ReconstructionError = np.linalg.norm(X - X_rec, ord='fro')
print (Z)
print (Rz)
print (X_rec)
print('Reconstruction Error:', ReconstructionError)