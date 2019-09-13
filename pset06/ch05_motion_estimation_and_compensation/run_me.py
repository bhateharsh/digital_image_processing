import numpy as np
import cv2
import dippykit as dip
from matplotlib.axes import Axes
from multiprocessing import Queue
import ch05_motion_module as student_submission


filepath_1 = "mobile_cif.yuv"
filepath_2 = "mother-daughter_cif.yuv"

# =============================================================================
# CHANGE THESE VALUES TO ALTER THE RENDERING
# =============================================================================
# FILEPATH: The path for the file to render. If your files are the current
# directory, then the paths defined above (filepath_1, filepath_2) will
# work. Otherwise, you'll have to define the paths yourself.
#
# AUTOPLAY: If set to False, you will be prompted before each frame is
# rendered.
#
# VERBOSE: If set to True, various information about the frame reading
# process will be displayed in the console. VERBOSE=True and AUTOPLAY=False
# is generally a bad combo.
#
# NUM_FRAMES: The number of frames to be read from the original image
FILEPATH = filepath_1
AUTO_PLAY = True
VERBOSE = False
NUM_FRAMES = 10


height = 288
width = 352
frame_length = int(height * width * 1.5)
MB_size = 16
p = 7
shape = (int(height * 1.5), width)
file = open(FILEPATH, 'rb')


def add_frames_to_queue(queue: Queue):
    yuv_data_old = None
    for frame_num in range(1, NUM_FRAMES+1):
        try:
            if VERBOSE:
                print('Attempting to read frame {}...'.format(frame_num))
            raw_data = file.read(frame_length)
            yuv_data = np.frombuffer(raw_data, dtype=np.uint8).reshape(shape)
            if VERBOSE:
                print('Frame {} successfully read!'.format(frame_num))
        except Exception as e:
            print('Error:')
            print(e)
            yuv_data = None
        if yuv_data is not None:
            if yuv_data_old is not None:
                im_I = yuv_data_old[:height, :]
                im_P = yuv_data[:height, :]
                yuv_data_old = yuv_data
            else:
                yuv_data_old = yuv_data
                continue

            frame_orig = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2RGB_I420)

            if VERBOSE:
                print('Calculating motion vectors...')
            MVs_y, MVs_x = student_submission.MVES(im_P, im_I, MB_size, p)
            if VERBOSE:
                print('Motion vectors calculated.')

            if VERBOSE:
                print('Compensating I-Frame using motion vectors...')
            im_compensated = \
                student_submission.motion_comp(im_I, MVs_y, MVs_x, MB_size)
            if VERBOSE:
                print('I-Frame compensated.')

            if VERBOSE:
                print('Overlaying motion vectors onto P-Frame...')
            frame_MVs = np.stack((im_P, im_P, im_P), axis=2)
            for i in range(int(height / MB_size)):
                for j in range(int(width / MB_size)):
                    ii = MB_size * i
                    jj = MB_size * j
                    dy = MVs_y[i, j]
                    dx = MVs_x[i, j]
                    cv2.arrowedLine(frame_MVs, (jj, ii),
                                    (jj + 2 * dy, ii + 2 * dx),
                                    (255, 0, 0))
            if VERBOSE:
                print('Motion vectors overlayed.')

            frame_diff = np.abs(im_compensated.astype(int) - im_P.astype(int))
            frame_diff = np.stack((frame_diff, frame_diff, frame_diff), axis=2)
            frame_diff = frame_diff.astype(np.uint8)

            pad = 255 * np.ones((height, 20, 3), dtype=np.uint8)

            im_dat = np.hstack((frame_orig, pad, frame_MVs, pad, frame_diff))
            title = 'Frame {}'.format(frame_num)
            queue.put((im_dat, title))
    queue.put(None)


def render_image(ax: Axes, data):
    ax.imshow(data[0])
    ax.axis('off')
    ax.set_title(data[1], fontsize='x-small')
    ax.figure.tight_layout()


if __name__ == '__main__':
    dip.setup_continuous_rendering(render_image, add_frames_to_queue,
                                   auto_play=AUTO_PLAY)

