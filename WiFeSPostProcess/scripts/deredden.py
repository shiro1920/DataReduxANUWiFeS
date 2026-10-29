from scripts.loadfile import load_fitsfile
from scripts.sex2dd import sex2dd
import sfdmap
import extinction
import numpy as np
from astropy.io import fits

def deRedden(lam, flux, var, ra, dec, dustmap_path):
    
    """Correct the Galactic extinction"""
    m = sfdmap.SFDMap(dustmap_path)

    Rv = 3.1
    Av      = m.ebv(ra, dec) * Rv
    Amag    = extinction.ccm89(lam, Av, Rv, unit='aa', out=None)
    flux_unred = extinction.remove(Amag, flux)
    var_unred = var * (flux_unred/flux)**2
    
    return flux_unred, var_unred


def unRedDatacube(filename, dustmap_path, root_path):
    head, wave, data, data_var = load_fitsfile(filename)
    obj_name = head['OBJECT']
    radec = '%s %s' % (head['RA'], head['DEC'])
    ra, dec = sex2dd(radec)
    num_xspa, num_yspa = head['NAXIS1'], head['NAXIS2']

    data_unred, data_var_unred = np.full_like(data, np.nan), np.full_like(data_var, np.nan)
    for idx in range(num_xspa):
        for idy in range(num_yspa):
            flux_red, var_red     = data[:, idy, idx], data_var[:, idy, idx]
            wave_AA = 10**wave
            flux_unred, var_unred = deRedden(wave_AA, flux_red, var_red, ra, dec, dustmap_path)
            data_unred[:, idy, idx]     = flux_unred
            data_var_unred[:, idy, idx] = var_unred


    hdu_prim       = fits.PrimaryHDU(data=data_unred,   header=head)
    hdu_err        = fits.ImageHDU(data=data_var_unred, header=head)
    hdulist        = fits.HDUList([hdu_prim, hdu_err])
    
    save_file = root_path+'/'+obj_name+'_deRedden.fits'
    hdulist.writeto(save_file, overwrite=True)    

    return 

        
    