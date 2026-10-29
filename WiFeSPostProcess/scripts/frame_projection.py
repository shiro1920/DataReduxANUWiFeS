import numpy as np
from astropy.io import fits

def create_new_head_formosaic(head, head_mosaic):
    hdu         = fits.PrimaryHDU()
    new_head    = hdu.header
    
    new_head['CDELT1'],  new_head['CDELT2']   = head_mosaic['CDELT1'], head_mosaic['CDELT2']
    new_head['CDELT3']                        = head['CDELT3']
    
    new_head['CRPIX1'], new_head['CRPIX2']  = head_mosaic['CRPIX1'], head_mosaic['CRPIX2']
    new_head['CRPIX3']                      = 0
    
    new_head['CTYPE1'], new_head['CTYPE2']  = head_mosaic['CTYPE1'], head_mosaic['CTYPE2'] 
    new_head['CTYPE3']                      = 'wave'
    
    new_head['CUNIT1'], new_head['CUNIT2']  = 'deg', 'deg'  
    new_head['CUNIT3']                      = 'logAngstrom'
    
    new_head['CRVAL1'], new_head['CRVAL2']  = head_mosaic['CRVAL1'], head_mosaic['CRVAL2']
    new_head['CRVAL3']                      = head['CRVAL3']

    return new_head

def stack_all_with_compress3D(stack, img):
    id           = np.isnan(stack[0,:,:,:])
    stack[0,id]  = img[id]
    img[id]      = np.nan

    n_slice = stack.shape[0]
    for i_slice in range(n_slice-1):
        id                   = ~np.isnan(stack[i_slice,:,:,:]) & np.isnan(stack[i_slice+1,:,:,:])
        stack[i_slice+1, id] = img[id]
        img[id]              = np.nan
        
    return stack 


def stack_all_with_compress2D(stack, img):
    id           = np.isnan(stack[0,:,:])
    stack[0,id]  = img[id]
    img[id]      = np.nan

    n_slice = stack.shape[0]
    for i_slice in range(n_slice-1):
        id                   = ~np.isnan(stack[i_slice,:,:]) & np.isnan(stack[i_slice+1,:,:])
        stack[i_slice+1, id] = img[id]
        img[id]              = np.nan
        
    return stack 

    