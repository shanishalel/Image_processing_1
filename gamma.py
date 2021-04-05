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
import cv2
import numpy as np
from ex1_utils import LOAD_GRAY_SCALE
import cv2 as cv
import argparse
import ex1_utils

#nothing
def on_trackbar(val):
    pass


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    Run=True
    img = cv2.imread(img_path)

    if rep == 1:  # gray scale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    title_window = 'GammaGui screenshot'
    trackbar='Gamma'
    cv2.namedWindow(title_window)


    # I will able the user to put values between 0 to 10
    cv2.createTrackbar(trackbar, title_window, 1, 10, on_trackbar)

    while Run: #the gamma will always correct herself according the user choosen
        #return trackbar position
        gamma = cv2.getTrackbarPos(trackbar, title_window)

        #because we asked that the slider value be from 0 to 2 with resolution 0.01
        #we will fix the value we will get
        gamma = gamma/10#cause we get values from 0 to 10 we will divide by 10 and then multiply by (2-0.01)

        gamma=gamma* (2 - 0.01)

        new_img=img/255.0
        g_array = np.full(new_img.shape, gamma)
        new_img = np.power(new_img, g_array)

        cv2.imshow(title_window, new_img)
        k = cv2.waitKey(1000) #wait
        #if we want to close it
        if k == 27:#esc
            break
        if cv2.getWindowProperty(title_window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows() #if we will stop

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
