# -*- coding: utf-8 -*-
"""
Created on Thur May 20 01:49:32 2021

@author: manavmehta
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_PDF_CDF(img):
    img_x = img.shape[0]
    img_y = img.shape[1]
    freq = [0]*(256)
    for row in img:
        for pixel in row:
            freq[int(pixel)] = freq[int(pixel)] + 1
        
    pdf = (np.array(freq))/(img_x*img_y) #divide by total number of pixels
    
    # first element as it is
    cdf = [pdf[0]]
    for i in range(1 , len(pdf)):
        cdf.append(cdf[i-1] + pdf[i])
    
    cdf = np.array(cdf)
    
    plt.plot(pdf)
    plt.plot(cdf)
    plt.show()# plotting the pdf & cdf for corresponding color

    return pdf,cdf


def equalize(img):

    eq_img = np.zeros(img.shape)
    pdf,cdf = get_PDF_CDF(img)
    
    # get the equalized (L-1)*cdf column
    # print("Shape = ", img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            eq_img[i][j] = 255 * (cdf[img[i][j]])
    get_PDF_CDF(eq_img)
    return eq_img


filename = "moon.jpg"
# filename = "city.jpg"
img = cv2.imread(filename)

# print("Shape = ", img.shape) # (194, 259, 3)

# Separate out RGB channels -> equalize and unite them back
processed_img = np.array([equalize(img[:,:,0]) , equalize(img[:,:,1]) , equalize(img[:,:,2])])
# This gives appended rows so need to transpose

final = np.zeros(img.shape)

# reshaping -> taking the transpose to get the needed np shape for writing
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        final[i][j][0] = processed_img[0][i][j]
        final[i][j][1] = processed_img[1][i][j]       
        final[i][j][2] = processed_img[2][i][j]

cv2.imwrite('enhanced_'+filename , final)
