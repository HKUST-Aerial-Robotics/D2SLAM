import cv2 as cv
import numpy as np

def calibPhotometric(img, photometric, is_rgb=True):
    if not is_rgb:
        ret = img.copy()
        if len(img.shape) == 3:
            ret = cv.cvtColor(ret, cv.COLOR_BGR2GRAY)
        ret = ret.astype(float)/photometric
    else:
        #Divide by photometric per channel
        ret = img.copy().astype(float)
        for i in range(img.shape[2]):
            ret[:,:,i] = ret[:,:,i]/photometric*0.7
    ret = np.clip(ret, 0, 255).astype(np.uint8)
    return ret

def calibPhotometricImgsIndividual(imgs, photometrics, is_rgb=True):
    photometric_calibed = []
    if photometrics is not None:
        #Convert to grayscale
        for i in range(len(imgs)):
            calibed = calibPhotometric(imgs[i], photometrics[i], is_rgb=is_rgb)
            photometric_calibed.append(calibed)
    else:
        photometric_calibed = imgs
    return photometric_calibed