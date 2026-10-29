import pyphot
from astropy.io import fits
import numpy as np
from pyphot import unit

def produce_broadband(head, data, wave, filename, filter_name, save2fits=True):
    
    cube_spec = data
    size      = cube_spec.shape
    wave_spec = wave

    #####broad-band r-band
    lib       = pyphot.get_library()
    f         = lib[filter_name]
    img_bband = np.zeros([size[1], size[2]])
        
    for id_x in np.arange(size[2]-1):   
        for id_y in np.arange(size[1]-1):
            flux_spec             = cube_spec[:, id_y, id_x]
            flux_spec[np.isnan(flux_spec)] = 0.0

            wv = wave_spec*unit['AA']
            fl = flux_spec*unit['erg/s/cm**2/AA']
            flux_syn              = f.get_flux(wv, fl) #erg/s/cm**2/AA
            
            img_bband[id_y, id_x] = flux_syn.value   #erg/s/cm**2/AA
    
    if save2fits:
        hdu = fits.PrimaryHDU(header=head, data=img_bband)
        hdu.writeto(filename, overwrite=True)    
    return img_bband