import numpy as np
from astropy.io import fits

def wave_grid(fileobj):
    wave_spec_min = np.zeros(len(fileobj))
    wave_spec_max = np.zeros(len(fileobj))
    for k in np.arange(len(fileobj)):
        hdu       = fits.open(fileobj[k])
        head      = hdu[0].header
        wave_spec = head['CRVAL3'] + np.arange(head['NAXIS3'])*head['CDELT3']
        wave_spec_min[k] = np.min(wave_spec)
        wave_spec_max[k] = np.max(wave_spec)
    
    idmax     = np.argmax(wave_spec_min)
    hdu       = fits.open(fileobj[idmax])
    head      = hdu[0].header
    wave_spec = head['CRVAL3'] + np.arange(head['NAXIS3'])*head['CDELT3']
    id        = np.where(wave_spec < np.min(wave_spec_max))
    new_wave  = wave_spec[id]
    
    id_head_new_wave = idmax
    head_NAXIS3      = len(new_wave)
    return new_wave, id_head_new_wave, head_NAXIS3