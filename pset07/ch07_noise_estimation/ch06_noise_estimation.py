
import numpy as np
import dippykit as dip

def noise_estimation(im: np.ndarray):
    """
    Given an image, this function determines the variance in its pixel
    values and displays a histogram of the image along with a fitted normal
    distribution.
    """
    # Calculate histogram
    im_hist, bin_edges = np.histogram(im, 30)
    bin_centers = bin_edges[:-1] + (np.diff(bin_edges) / 2)
    # Normalize histogram to make a PDF
    im_hist = im_hist.astype(float)
    im_hist /= np.sum(im_hist)
    # Calculate the mean and variance values
    mean = np.sum(im_hist * bin_centers)
    var = np.sum(im_hist * (bin_centers ** 2)) - (mean ** 2)
    # Calculate the values on the normal distribution PDF
    norm_vals = np.exp(-((bin_centers - mean) ** 2) / (2 * var)) \
            / np.sqrt(2 * np.pi * var)
    # Normalize the norm_vals
    norm_vals /= np.sum(norm_vals)
    # Rescale variance
    var /= (255 ** 2)
    print('Variance: {}'.format(var))
    dip.figure()
    dip.bar(bin_centers, im_hist)
    dip.plot(bin_centers, norm_vals, 'r')
    dip.legend(['Fitted Gaussian PDF', 'Histogram of image'])
    dip.xlabel('Pixel value')
    dip.ylabel('Occurrence')
    dip.show()

def main():
    ######## PART (a): EDIT HERE ########
    img = img = dip.im_read("WiseonRocks_noise_1.png")
    # img = dip.im_to_float(img)
    # img = img*255
    print ("The shape of IMG is %s \n" %str(img.shape))
    ######## PART (b): EDIT HERE ########
    noise_estimation(img)
    
    ######## PART (c)-(e): EDIT HERE ########
    SIZE = 100
    print ("Image 1")
    flatRegion = img[0:SIZE, 100:(100+SIZE)]
    notFlatRegion = img[250:(250+SIZE), 150:(150+SIZE)]
    noise_estimation(flatRegion)
    noise_estimation(notFlatRegion)

    print ("Image 2")
    img = img = dip.im_read("WiseonRocks_noise_2.png")
    flatRegion = img[0:SIZE, 100:(100+SIZE)]
    noise_estimation(img)
    notFlatRegion = img[250:(250+SIZE), 150:(150+SIZE)]
    noise_estimation(flatRegion)
    noise_estimation(notFlatRegion)

    print ("Image 3")
    img = img = dip.im_read("WiseonRocks_noise_3.png")
    flatRegion = img[0:SIZE, 100:(100+SIZE)]
    noise_estimation(img)
    notFlatRegion = img[250:(250+SIZE), 150:(150+SIZE)]
    noise_estimation(flatRegion)
    noise_estimation(notFlatRegion)

    print ("Image 4")
    img = img = dip.im_read("WiseonRocks_noise_4.png")
    flatRegion = img[0:SIZE, 100:(100+SIZE)]
    noise_estimation(img)
    notFlatRegion = img[250:(250+SIZE), 150:(150+SIZE)]
    noise_estimation(flatRegion)
    noise_estimation(notFlatRegion)

    print ("Image 5")
    img = img = dip.im_read("WiseonRocks_noise_5.png")
    flatRegion = img[0:SIZE, 100:(100+SIZE)]
    noise_estimation(img)
    notFlatRegion = img[250:(250+SIZE), 150:(150+SIZE)]
    noise_estimation(flatRegion)
    noise_estimation(notFlatRegion)

if __name__ == '__main__':
    main()
    
