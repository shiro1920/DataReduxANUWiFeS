from astropy.io import fits
import numpy as np

def load_fitsfile(path):
    
    hdu = fits.open(path)
    data     = hdu[0].data
    data_var = hdu[1].data
    head     = hdu[0].header
    wave     = head['CRVAL3'] + np.arange(head['NAXIS3'])*head['CDELT3']

    return head, wave, data, data_var