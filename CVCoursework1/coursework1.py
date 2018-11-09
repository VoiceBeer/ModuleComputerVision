# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt


#   Template convolution
def template_convolve(image, weight):
    # get image dimensions and template dimensions
    image_rows = image.shape[0]
    image_cols = image.shape[1]

    template_rows, template_cols = weight.shape

    # half of template rows and cols
    htr = int(np.floor(template_rows / 2))
    htc = int(np.floor(template_cols / 2))

    # set an output as black. (initialize an same size array with original image,
    # since this assignment said "The border pixels should be set to 0."
    # so give value 0 to this array as Mark said in his book)
    result = np.zeros((image_rows, image_cols), np.double)

    # convolve the template
    # these two loop is to find the centre point of template in the image
    for i in range(htr, image_rows-htr):
        for j in range(htc, image_cols-htc):
            # use matrix calculation in python is faster than the code in the Mark's book,
            # and remember to reverse the template, which means,
            # it's not like O11 * W11, O12 * W12, actually it's O11 * W33, O12 * W32
            # and give the value to the result
            result[i, j] = np.sum(image[i-htr: i+htr+1, j-htc: j+htc+1] * reverse(weight))/255.0

    return result


# Split the image into three channels, convolve each of them, and return the merge of all
def convolve(image, weight):
    # If the image is a color image
    if len(image.shape) == 3:
        # Split the image into three images
        b, g, r = cv2.split(image)

        # Convolve each of them
        convolved_b = template_convolve(b, weight)
        convolved_g = template_convolve(g, weight)
        convolved_r = template_convolve(r, weight)

        # Merge
        new_img = cv2.merge((convolved_b, convolved_g, convolved_r))

        return new_img

    else:
        return template_convolve(image, weight)


def gaussian(sigma):
    # winsize: size of template (odd, integer)
    # sigma: variance of Gaussian function

    winsize = int(8.0 * sigma + 1.0)
    if (winsize % 2 == 0):
        winsize += 1

    centre = np.floor(winsize/2)

    # initiate sum
    sum = 0.0

    template = np.zeros((winsize, winsize), np.double)

    for i in range(0, winsize):
        for j in range(0, winsize):
            template[j, i] = np.exp(-(((j-centre) * (j-centre)) +
                                      ((i-centre) * (i-centre))) / (2*sigma*sigma))
            sum += template[j, i]
    template = template / sum
    return template


def make_border(image, kernal):

    # get the half of the kernal size and image's shape
    p = int(np.floor(kernal.shape[0] / 2))
    image_rows = image.shape[0]
    image_cols = image.shape[1]

    # get the needed part of image
    image = image[p: image_rows-p, p: image_cols-p]

    # make border with 0
    image = cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    return image


# Reverse left and right
def reverse_lr(a):
    return a[::-1]


# Reverse upper and lower
def reverse(mat):
    return np.array(reverse_lr(list(map(reverse_lr, mat))))


if __name__ == '__main__':
    # image1 = cv2.imread('./output/hybrid_image.bmp')
    #
    # fig = plt.figure(16)
    #
    # plt.subplot(2, 2, 1)
    # plt.imshow(image1)
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(cv2.pyrDown(image1))
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(cv2.pyrDown(cv2.pyrDown(image1)))
    # plt.axis('off')
    #
    # plt.subplot(2,2,4)
    # plt.imshow(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(image1))))
    # plt.axis('off')
    #
    # plt.show()

    # read the images and tranform them to the double type
    image1 = cv2.imread('./data/fish.bmp') / 1.0
    image2 = cv2.imread('./data/submarine.bmp') / 1.0


    # initiate gaussian kernal
    kernal = gaussian(4)

    # Remove the high frequencies from image1 by blurring it. The amount of
    # blur that works best will vary with different image pairs
    low_frequencies = convolve(image1, kernal)
    cv2.normalize(low_frequencies, low_frequencies, 0.0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow("low", low_frequencies)


    # Remove the low frequencies from image2. The easiest way to do this is to
    # subtract a blurred version of image2 from the original version of image2.
    # This will give you an image centered at zero with negative values.
    high_frequencies = make_border(image2, kernal)/255.0 - convolve(image2, kernal)
    cv2.normalize(high_frequencies, high_frequencies, 0.0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow("high frequencies", high_frequencies)


    # Combine the high frequencies and low frequencies
    hybrid_image = low_frequencies + high_frequencies
    # cv2.imshow("hybrid", hybrid_image)
    #
    #
    #
    cv2.normalize(low_frequencies, low_frequencies, 0.0, 255.0, cv2.NORM_MINMAX)
    cv2.normalize(high_frequencies, high_frequencies, 0.0, 255.0, cv2.NORM_MINMAX)
    cv2.normalize(hybrid_image, hybrid_image, 0.0, 255.0, cv2.NORM_MINMAX)
    cv2.imwrite('./output/low_frequencies_fish.bmp', low_frequencies)
    cv2.imwrite('./output/high_frequencies_submarine.bmp', high_frequencies)
    cv2.imwrite("./output/hybrid_image_fish_submarine.bmp", hybrid_image)

    cv2.waitKey(0)