from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import preprocessing


def calibrate(scienceFrame, darkFrame, biasFrame, flatField):
    science_data = scienceFrame[0].data
    dark_data = darkFrame[0].data
    bias_data = biasFrame[0].data
    flat_data = flatField[0].data

    flat_norm = flat_data / flat_data.mean()

    data = (science_data - dark_data) / flat_norm

    # Your code should return two variables.
    #    hdul = Header Data Unit of the science image
    #    data = variable containing the data from the science image after calibration.

    hdul = scienceFrame

    return hdul, data

