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
    result = np.array([[0] * image_rows] * image_cols)

    # convolve the template
    # The indexes of start point and end point are different than the codes in Mark's book
    # Because 0 is the first index, I don't know whether 0 is the first index in Matlab.
    for x in range(htc + 1, image_cols - htc + 1): # for loop from the (left border - htc + 1) to the (right border - htc)
        for y in range(htr + 1, image_rows - htr + 1): # for loop from the (upper border - htr + 1) to the (lower border - htr)
            sum = 0 # initialize the sum
            for iwin in range(1, template_cols + 1): # Traverse from 1 to cols
                for jwin in range(1, template_rows + 1): # Traverse from 1 to rows
                    # Pixel values are multiplied by the corresponding weighting coefficient and added to an overall sum
                    # The sum (usually) evaluates a new value for the centre pixel (where the template is centred) and this becomes the pixel in the output image
                    # Remember to reverse the template, which means, it's not like O11 * W11, O12 * W12, actually
                    # It's O11 * W33, O12 * W32
                    #print(y+jwin-htr-2, x+iwin-htc-2, template_rows-jwin, template_cols-iwin)
                    sum = sum + image[y+jwin-htr-2, x+iwin-htc-2] * template[template_rows-jwin, template_cols-iwin]
            #print("------")
            result[y - 1, x - 1] = sum # Give the new value to the new image

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

if __name__ == '__main__':
    # image = np.array([[0] * 3] * 5)
    # template = np.array([[0] * 4] * 3)
    # template_convolve(image, template)
    image = np.array([[10, 20, 30, 40, 50],
                     [20, 30, 40, 50, 60],
                    [30, 40, 10, 20, 30],
                     [40, 50, 60, 70, 80],
                     [50, 60, 70, 80, 90]])

    # image = cv2.imread('./data/bicycle.bmp') / 255.0
    # print("image: ", image)
    # print("shape: ", image.shape)

    # b, g, r = cv2.split(image)
    # print("image r: ", r)
    # # cv2.imshow("Red 1", r)
    # # cv2.imshow("Blue 1", b)
    # # cv2.imshow("Green 1", g)
    # cv2.waitKey(0)


    template = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    print("template: ", template)

    print(convolve(image, template))