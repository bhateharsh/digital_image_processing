import numpy as np
import dippykit as dip
from timeit import default_timer
from ch06_OF_utils import flow_to_color, read_flow_file
from ch06_OF_algos import horn_schuck, lucas_kanade


filepaths = ('images/Drop/',
             'images/GrassSky/',
             'images/RubberWhale/',
             'images/Urban/')

def run_optical_flow(filepath_ind: int, OF_alg: int, param: int=100,
                     display: bool=True):

    frame_1 = dip.im_read(filepaths[filepath_ind] + 'frame1.png')[:, :, :3]
    frame_2 = dip.im_read(filepaths[filepath_ind] + 'frame2.png')[:, :, :3]
    residual = np.abs(frame_1.astype(float) - frame_2.astype(float)) \
            .astype(np.uint8)
    frame_1_gray = dip.rgb2gray(frame_1)
    frame_2_gray = dip.rgb2gray(frame_2)
    PSNR_val = dip.PSNR(frame_1_gray, frame_2_gray)
    if display:
        # Plot the initial images
        dip.figure()
        dip.subplot(1, 3, 1)
        dip.imshow(frame_1)
        dip.title('Frame 1', fontsize='x-small')
        dip.subplot(1, 3, 2)
        dip.imshow(frame_2)
        dip.title('Frame 2', fontsize='x-small')
        dip.subplot(1, 3, 3)
        dip.imshow(residual)
        dip.title('Residual - PSNR: {:.2f} dB'.format(PSNR_val), fontsize='x-small')

    # Convert to grayscale for analysis
    frame_1 = dip.im_to_float(frame_1_gray)
    frame_2 = dip.im_to_float(frame_2_gray)
    start_time = default_timer()

    # ============================ EDIT THIS PART =============================
    mask_x = np.array([[-1, 1], [-1,1]])
    mask_y = np.array([[-1, -1], [1,1]])
    mask_t_2 = np.array([[-1, -1], [-1,-1]])
    mask_t_1 = np.array([[1, 1], [1,1]])
    dIx = dip.convolve2d(frame_1, mask_x, mode='same', like_matlab=True)
    dIy = dip.convolve2d(frame_1, mask_y, mode='same', like_matlab=True)
    dIt = dip.convolve2d(frame_1, mask_t_1, mode='same', like_matlab=True) + dip.convolve2d(frame_2, mask_t_2, mode='same', like_matlab=True)
    
    # ==========!!!!! DO NOT EDIT ANYTHING BELOW THIS !!!!!====================

    # Instantiate blank u and v matrices
    u = np.zeros_like(frame_1)
    v = np.zeros_like(frame_1)

    if 0 == OF_alg:
        print('The optical flow is estimated using Horn-Schuck...')
        u, v = horn_schuck(u, v, dIx, dIy, dIt, param)
    elif 1 == OF_alg:
        print('The optical flow is estimated using Lucas-Kanade...')
        u, v = lucas_kanade(u, v, dIx, dIy, dIt, param)
    else:
        raise ValueError('OF_alg must be either 0 or 1')

    end_time = default_timer()

    # Determine run time
    duration = end_time - start_time
    clock = [int(duration // 60), int(duration % 60)]
    print('Flow estimation time was {} minutes and {} seconds'
            .format(*clock))

    # Downsample for better visuals
    stride = 10
    m, n = frame_1.shape
    x, y = np.meshgrid(range(n), range(m))
    x = x.astype('float64')
    y = y.astype('float64')

    # Downsampled u and v
    u_ds = u[::stride, ::stride]
    v_ds = v[::stride, ::stride]

    # Coords for downsampled u and v
    x_ds = x[::stride, ::stride]
    y_ds = y[::stride, ::stride]

    # Estimated flow
    estimated_flow = np.stack((u, v), axis=2)

    # Read file for ground truth flow
    ground_truth_flow = read_flow_file(filepaths[filepath_ind] + 'flow1_2.flo')
    u_gt_orig = ground_truth_flow[:, :, 0]
    v_gt_orig = ground_truth_flow[:, :, 1]
    u_gt = np.where(np.isnan(u_gt_orig), 0, u_gt_orig)
    v_gt = np.where(np.isnan(v_gt_orig), 0, v_gt_orig)


    # Downsampled u_gt and v_gt
    u_gt_ds = u_gt[::stride, ::stride]
    v_gt_ds = v_gt[::stride, ::stride]
    if display:
        # Plot the optical flow field
        dip.figure()
        dip.subplot(2, 2, 1)
        dip.imshow(frame_2, 'gray')
        dip.quiver(x_ds, y_ds, u_ds, v_ds, color='r')
        dip.title('Estimated', fontsize='x-small')
        dip.subplot(2, 2, 2)
        dip.imshow(frame_2, 'gray')
        dip.quiver(x_ds, y_ds, u_gt_ds, v_gt_ds, color='r')
        dip.title('Ground truth', fontsize='x-small')
        # Draw colored velocity flow maps
        dip.subplot(2, 2, 3)
        dip.imshow(flow_to_color(estimated_flow))
        dip.title('Estimated', fontsize='x-small')
        dip.subplot(2, 2, 4)
        dip.imshow(flow_to_color(ground_truth_flow))
        dip.title('Ground truth', fontsize='x-small')

    # Normalization for metric computations
    normalize = lambda im: (im - np.min(im)) / (np.max(im) - np.min(im))
    un = normalize(u)
    un_gt = normalize(u_gt)
    un_gt[np.isnan(u_gt_orig)] = 1
    vn = normalize(v)
    vn_gt = normalize(v_gt)
    vn_gt[np.isnan(v_gt_orig)] = 1

    # Error calculations and displays
    EPE = ((un - un_gt) ** 2 + (vn - vn_gt) ** 2) ** 0.5
    AE = np.arccos(((un * un_gt) + (vn * vn_gt) + 1) /
                   (((un + vn + 1) * (un_gt + vn_gt + 1)) ** 0.5))
    EPE_nan_ratio = np.sum(np.isnan(EPE)) / EPE.size
    AE_nan_ratio = np.sum(np.isnan(AE)) / AE.size
    EPE_inf_ratio = np.sum(np.isinf(EPE)) / EPE.size
    AE_inf_ratio = np.sum(np.isinf(AE)) / AE.size
    print('Error nan ratio: EPE={:.2f}, AE={:.2f}'
            .format(EPE_nan_ratio, AE_nan_ratio))
    print('Error inf ratio: EPE={:.2f}, AE={:.2f}'
            .format(EPE_inf_ratio, AE_inf_ratio))
    EPE_avg = np.mean(EPE[~np.isnan(EPE)])
    AE_avg = np.mean(AE[~np.isnan(AE)])
    print('EPE={:.2f}, AE={:.2f}'.format(EPE_avg, AE_avg))

    if display:
        dip.show()

    return clock, EPE_avg, AE_avg


if __name__ == '__main__':
    # Set this variable to choose which image set to use
    # 0 - Drop
    # 1 - GrassSky
    # 2 - RubberWhale
    # 3 - Urban
    filepath_ind = 0

    # Set this variable to choose which optical flow algorithm to use
    # 0 - Horn-Schuck
    # 1 - Lucas-Kanade
    OF_alg = 1

    run_optical_flow(filepath_ind, OF_alg,5)