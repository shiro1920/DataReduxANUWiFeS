from astropy.io import fits
import numpy as np

def define_strongskyline(data, head, cut, degree=3):
    wave_spec    = head['CRVAL3'] + np.arange(head['NAXIS3'])*head['CDELT3']
    num_xspa     = head['NAXIS1']
    num_yspa     = head['NAXIS2']
    
     
    wave_skyline = np.zeros(len(wave_spec))
    flux         = np.median(data[:,1:-1,1:-1], axis=(1,2))
    
    id_nonan   = ~np.isnan(flux)
    
    poly_para = np.polyfit(wave_spec[id_nonan], flux[id_nonan], deg=degree)
    flux_model= np.polyval(poly_para, wave_spec)
    
    for m in np.arange(50):
        std = np.std(flux[id_nonan]-flux_model[id_nonan])
        id  = np.where( np.abs(flux[id_nonan]-flux_model[id_nonan]) < 3*std)
        
        poly_para = np.polyfit(wave_spec[id_nonan], flux[id_nonan], deg=degree)
        flux_model= np.polyval(poly_para, wave_spec)
    
    std        = np.std(flux[id_nonan]-flux_model[id_nonan])
    id         = np.where( np.abs(flux-flux_model) > cut*std)
    wave_skyline[id]= 1
    
    return wave_skyline