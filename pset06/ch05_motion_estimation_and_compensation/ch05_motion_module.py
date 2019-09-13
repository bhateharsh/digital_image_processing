import dippykit as dip
import numpy as np


def MVES(im_P: np.ndarray, im_I: np.ndarray, MB_size: int, p: int) \
        -> (np.ndarray, np.ndarray):
    f_height = im_I.shape[0]
    f_width = im_I.shape[1]
    MVs_y = np.zeros((int(f_height / MB_size), int(f_width / MB_size)),
                     dtype=int)
    MVs_x = np.zeros((int(f_height / MB_size), int(f_width / MB_size)),
                     dtype=int)
    MAD = np.zeros((int(f_height / MB_size), int(f_width / MB_size)))
    for i in range(int(f_height / MB_size)):
        for j in range(int(f_width / MB_size)):

            err_min = np.inf
            for m in range(-p, p+1):
                for n in range(-p, p+1):
                    ii = MB_size * i
                    jj = MB_size * j
                    ref_blk_y = ii + m
                    ref_blk_x = jj + n
                    if ref_blk_y < 0 or (ref_blk_y + MB_size) > f_height or \
                            ref_blk_x < 0 or (ref_blk_x + MB_size) > f_width:
                        continue

                    # EDIT THIS PART
                    # Compute the mean absolute difference (MAD) between the
                    #  two blocks
                    # Part a.i
                    curr_MB = im_P[ii:(ii+MB_size),jj:(jj+MB_size)]
                    ref_MB = im_I[ref_blk_y:(ref_blk_y+MB_size),ref_blk_x:(ref_blk_x+MB_size)]
                    if(curr_MB.shape != ref_MB.shape):
                        continue
                    err = dip.MAD(curr_MB, ref_MB)
                    # EDIT THIS PART
                    # Check whether the current error is less than the
                    # err_min value, which represents the minimum error seen
                    # so far
                    # If the current error is less than err_min, reassign
                    # err_min to the current error and record the associated
                    #  motion vector (derived from m and n)
                    # Part a.ii
                    if (err < err_min):
                        err_min = err
                        dy = m
                        dx = n

            MVs_y[i, j] = dy
            MVs_x[i, j] = dx

    return MVs_y, MVs_x


def motion_comp(im_I: np.ndarray, MVs_y: np.ndarray, MVs_x: np.ndarray,
                MB_size: int) -> np.ndarray:
    f_height = im_I.shape[0]
    f_width = im_I.shape[1]
    im_compensated = np.zeros_like(im_I)

    for i in range(int(f_height / MB_size)):
        for j in range(int(f_width / MB_size)):
            ii = MB_size * i
            jj = MB_size * j
            
            # EDIT THIS PART
            dx = MVs_x[i,j]  # Part b.i
            dy = MVs_y[i,j]  # Part b.i
            ref_blk_y = ii+dy  # Part b.ii
            ref_blk_x = jj+dx # Part b.ii
            im_compensated[ii:(ii + MB_size), jj:(jj + MB_size)] = im_I[ref_blk_y:(ref_blk_y+MB_size),ref_blk_x:(ref_blk_x+MB_size)] # Part b.iii

    return im_compensated

