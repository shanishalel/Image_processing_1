"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib.pyplot as plt
from numpy.linalg import linalg

from scipy import misc
import numpy as np
import cv2
import scipy
from scipy import misc
import math

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206134033


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation==1 :
        image=cv2.imread(filename)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return gray/np.max(gray) #normalized

        """just for me 
        cv2.imshow('Original image',image)
        cv2.waitKey(0)

        cv2.imshow('Gray image', gray)
        cv2.waitKey(0)
        """

    if representation==2 :
        image=cv2.imread(filename)
        color=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return color/np.max(color) #normalized


        """just for me 
        cv2.imshow('Original image',image)
        cv2.waitKey(0)

        cv2.imshow('rgb image', color)
        cv2.waitKey(0)
        """

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation==1 :
        plt.imshow(imReadAndConvert(filename,1),cmap='gray' )
        plt.show()

    if representation==2 :
        plt.imshow( imReadAndConvert(filename, 2))
        plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    transform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])
    #multiply
    # if we will delete the T.copy we will get photo that is more yellow
    new_img = np.dot(imgRGB, transform.T.copy())
    return new_img


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    #the matrix we should multiptly yiq to get rgb (from wiki)
    transform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])

    #Compute the (multiplicative) inverse of a matrix
    ans = np.linalg.inv(transform)
    #if we will delete the T.copy we will get photo that is more yellow
    new_img = np.dot(imgYIQ, ans.T.copy())
    return new_img


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    #check if it is gray or rgb
    if len(imgOrig.shape) ==2: # is gray
        gray=True

    if len(imgOrig.shape) == 3:  # is RGB
        gray = False

    if gray==False:#rgb
        YIQ = transformRGB2YIQ(imgOrig)
        imgOrig = YIQ[:, :, 0] #takes only y chanel


    #The minimum pixel value will be mapped to the minimum output value(alpha), and the maximum pixel
    # value will be mapped to the maximum output value(beta).
    #NORM_MINMAX calculates along the lines of ((pixel_value - alpha)/(beta - alpha)) * beta
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')

    #calculate the image histogram
    orig_hist = np.histogram(imgOrig.flatten(), bins=256)[0]
    #calculate the normalized cumulative sum
    cs = np.cumsum(orig_hist)

    new_img = cs[imgOrig]
    #normlazied the cumsum
    new_img = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX)
    new_img = new_img.astype('uint8')

    new_hist = np.histogram(new_img.flatten(), bins=256)[0]

    #if an RGB image is given we should operate on y channel of yiq image
    #and then convert back from yiq to rgb
    if gray==False: #RGB
        YIQ[:, :, 0] = new_img / (new_img.max() - new_img.min())
        new_img = transformYIQ2RGB(YIQ)

    return new_img, orig_hist, new_hist


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):

    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if imOrig.shape[-1] == 3:
        isRGB=True
        imgYIQ = transformRGB2YIQ(imOrig)
        imOrig = np.copy(imgYIQ[:, :, 0])
    else:
        isRGB=False
        imgYIQ=None
        imOrig=np.copy(imOrig)

    #switch to int
    if np.amax(imOrig) <= 1:
        imOrig = imOrig * 255
    imOrig = imOrig.astype('uint8')
    histOrg, bin_edges = np.histogram(imOrig, 256, [0, 255])

    im_shape = imOrig.shape
    z = init_bound(nQuant)
    qImage_list = list()
    errors = list()

    for i in range(nIter):
        new_img = np.zeros(im_shape)
        q = np.zeros(nQuant)
        q,new_img=finding_q(q,z,histOrg,imOrig,new_img)
        #finding the error
        mse = finding_error(imOrig / 255.0, new_img / 255.0)
        errors.append(mse)

        if isRGB:
            y_chanel=new_img / 255.0
            imgYIQ[:, :, 0] = y_chanel
            rgb_img = transformYIQ2RGB(imgYIQ)
            new_img = rgb_img
        qImage_list.append(new_img)

        #By the q[i-1]+q[i]/2
        z=fix_z(z,q)

        if len(errors) >= 2:
            if np.abs(errors[-1] - errors[-2]) <= 0.000001:
                break

    return qImage_list, errors


def init_bound(nQuant: int) -> np.ndarray:
    """
    function that init the boundary as we study from tirgul
    :param nQuant: number of colors
    :return:
    """
    s = int(255 / nQuant)
    z = np.zeros(nQuant + 1, dtype=int)
    for i in range(1, nQuant):
        z[i] = z[i - 1] + s
    z[nQuant] = 255
    return z


def finding_error(old: np.ndarray, new: np.ndarray) -> float:
    """
    function that found the value of the error - from the tirgul
    mse= sqrt(sum(imgold-imgnew)^2)/allpixels
    :param old:old img
    :param new:new img
    :return:
    """
    all_pixels = old.size
    sub = np.subtract( old,new)
    pix_sum = np.sum(np.square(sub))
    return np.sqrt(pix_sum) / all_pixels


def finding_q(q,z,histOrg,imOrig,new_img):
    """
    function that finding the q by :
    (z[i] to z[i+1] : g*h(g))/(z[i] to z[i+1] h(g))
    :param q:
    :param z:
    :param histOrg:
    :param imOrig:
    :param new_img:
    :return:
    """
    for i in range(len(q)):
        if i == len(q) - 1:
            j = z[i + 1] + 1
        else:
            j = z[i + 1]
        array = np.arange(z[i], j)
        q[i] = np.average(array, weights=histOrg[z[i]:j])
        #fill the new img with q[
        condition = np.logical_and(imOrig >= z[i], imOrig < j)
        new_img[condition] = q[i]
    return q,new_img


def fix_z(z, q):
    """
        function that found the z according to : z[i]=q[i-1]-q[i]\2
    :param z: z
    :param q: q
    :return: z after fixing the boundry
    """
    for bound in range(1, len(z) - 1):
        z[bound] = (q[bound - 1] + q[bound]) / 2
    return z