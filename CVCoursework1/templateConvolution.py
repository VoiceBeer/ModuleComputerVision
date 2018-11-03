# -*- coding: utf-8 -*-
import numpy as np
import cv2

#   Template convolution
def template_convolve(image, template):

    # get image dimensions and template dimensions
    image_rows = image.shape[0]
    image_cols = image.shape[1]

    template_rows, template_cols = template.shape

    # half of template rows and cols
    htr = int(np.floor(template_rows / 2))
    htc = int(np.floor(template_cols / 2))

    # set an output as black. (initialize an same size array with original image,
    # since this assignment said "The border pixels should be set to 0." so give value 0 to this array
    # as Mark said in his book)
    result = np.zeros((image_rows, image_cols), np.uint8)

    # convolve the template
    # The indexes of start point and end point are different than the codes in Mark's book
    # Because 0 is the first index, I don't know whether 0 is the first index in Matlab.
    for x in range(htc, image_cols-htc): # for loop from the (left border - htc + 1) to the (right border - htc)
        for y in range(htr, image_rows-htr): # for loop from the (upper border - htr + 1) to the (lower border - htr)
            sum = 0.0 # initialize the sum
            for iwin in range(0, template_cols): # Traverse from 1 to cols
                for jwin in range(0, template_rows): # Traverse from 1 to rows
                    # Pixel values are multiplied by the corresponding weighting coefficient and added to an overall sum
                    # The sum (usually) evaluates a new value for the centre pixel (where the template is centred) and this becomes the pixel in the output image
                    # Remember to reverse the template, which means, it's not like O11 * W11, O12 * W12, actually
                    # It's O11 * W33, O12 * W32
                    # print(y+jwin-htr, x+iwin-htc, template_rows-jwin-1, template_cols-iwin-1)
                    # print("sum: ", image[y+jwin-htr, x+iwin-htc] * template[template_rows-jwin-1, template_cols-iwin-1])
                    sum += image[y+jwin-htr, x+iwin-htc] * template[template_rows-jwin-1, template_cols-iwin-1]
                     #print(sum)
            # print("------")
            result[y, x] = sum # Give the new value to the new image

    print("result: ", result)
    return result

# Split the image into three channels, convolve each of them, and return the merge of all
def convolve(image, template):

    # If the image is a color image
    if len(image.shape) == 3:
        # Split the image into three images
        b, g, r = cv2.split(image)

        # Convolve each of them
        convolved_b = template_convolve(b, template)
        convolved_g = template_convolve(g, template)
        convolved_r = template_convolve(r, template)

        # Merge
        newImg = cv2.merge((convolved_b, convolved_g, convolved_r))

        return newImg
    else:
        return template_convolve(image, template)

def gaussian(winsize, sigma):

    # winsize: size of template (odd, integer)
    # sigma: variance of Gaussian function
    centre = np.floor(winsize/2)

    # initiate sum
    sum = 0.0

    template = np.zeros((winsize, winsize), np.double)

    for i in range(0, winsize):
        for j in range(0,winsize):
            template[j, i] = np.exp(-(((j-centre) * (j-centre)) +
                                      ((i-centre) * (i-centre))) / (2*sigma*sigma))
            sum += template[j, i]
    template = template / sum
    return template

if __name__ == '__main__':
    # image = np.array([[0] * 3] * 5)
    # template = np.array([[0] * 4] * 3)
    # template_convolve(image, template)
    # image = np.array([[10, 20, 30, 40, 50],
    #                  [20, 30, 40, 50, 60],
    #                 [30, 40, 10, 20, 30],
    #                  [40, 50, 60, 70, 80],
    #                  [50, 60, 70, 80, 90]])

    image = cv2.imread('./data/cat.bmp') / 1.0
    # print("image: ", image)
    # print("shape: ", image.shape)

    # b, g, r = cv2.split(image)
    # print("image r: ", r)
    # # cv2.imshow("Red 1", r)
    # # cv2.imshow("Blue 1", b)
    # # cv2.imshow("Green 1", g)
    # cv2.waitKey(0)


    # template = np.array([[1, 2, 1],
    #                     [0, 0, 0],
    #                     [-1, -2, -1]])
    # print("template: ", template)

    # cv2.imshow("Result", convolve(image, template))
    # cv2.waitKey(0)
    # print(gaussian(5, 1.0))

    template = gaussian(5, 1.0)
    print(template)
    cv2.imshow("Result", convolve(image, template))
    cv2.waitKey(0)