import cv2 as cv
import numpy as np
from scipy.signal import fftconvolve

def find_stop_sign(T, I, threshold=0.95):
    #convert to grayscale
    T = cv.cvtColor(T, cv.COLOR_BGR2GRAY)
    I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
    
    #subtract means of images to remove constant brightness/darkness region bias
    T = T - np.mean(T)
    #Subtracts local means of each region in I since we are sliding T over regions of I,
    #not just placing it over the whole image
    I = I - fftconvolve(I, np.ones(T.shape) / T.size, mode='same')
    
    #template is only 64x64, so will be smaller than any image we use it on
    #so no need to pad image
    h, w = T.shape
    #get fft of T&I
    T_fourier = np.fft.fft2(T, s=I.shape)
    I_fourier = np.fft.fft2(I)
    
    result = I_fourier * np.conj(T_fourier)
    
    #inverse fft to get correlation map
    correlation = np.real(np.fft.ifft2(result))

    #want to normalize correlation to use thresholding when detecting stop sign
    #need to divide by template and window energies
    T_energy = np.sqrt(np.sum(T**2))
    I2 = I**2
    window_energy = np.sqrt(fftconvolve(I2, np.ones(T.shape), mode='same'))
    correlation_normed = correlation / (T_energy * window_energy + 1e-10)

    if np.max(correlation_normed) > threshold: 
        #get bbox
        y, x = np.unravel_index(np.argmax(correlation_normed), correlation_normed.shape)
        #account for offset
        y = (y - h + 1)
        x = (x - w + 1)
        return np.array([y, x, h, w]).astype(int)
    #does not meet our threshold
    return None
    
    