import glob
import numpy as np
from astropy.io import fits
from reproject import reproject_exact
from scripts.sex2dd import sex2dd


def create_head2dwcs(head, pos, PA, pix_scale, num_xspa, num_yspa):
    ###a blank header
    hdu         = fits.PrimaryHDU()
    head2dwcs   = hdu.header
    ###position
    coord_deg  = sex2dd(pos)  #

    head2dwcs['CRPIX1'], head2dwcs['CRPIX2'] = num_xspa/2.0, num_yspa/2.0
    head2dwcs['CRVAL1'], head2dwcs['CRVAL2'] = coord_deg[0], coord_deg[1]
    head2dwcs['CTYPE1'], head2dwcs['CTYPE2'] = 'RA---TAN', 'DEC--TAN'

    ###increments
    head2dwcs['CD1_1'], head2dwcs['CD1_2'] = (-pix_scale*np.cos(PA), 'deg'), (-pix_scale*np.sin(PA), 'deg')
    head2dwcs['CD2_1'], head2dwcs['CD2_2'] = (-pix_scale*np.sin(PA), 'deg'), (pix_scale*np.cos(PA), 'deg')

    head2dwcs['CUNIT1'], head2dwcs['CUNIT2']= 'deg', 'deg'  
    return head2dwcs

###date-obs
def return_date_obs(fileobj):
    date_obs = ''
    air_mass = ''
    exptime  = 0.0
    for k in np.arange(len(fileobj)):
        file_ind   = fileobj[k]
        hdu_ind    = fits.open(file_ind)
        head_ind   = hdu_ind[0].header    
        date_obs   = date_obs + head_ind['DATE-OBS']+', '
        air_mass   = air_mass + str(head_ind['airmass'])+', '

        if head_ind['WIFESOBS'] == 'ClassicalEqual':
            exptime = exptime + float(head_ind['exptime'])
        if head_ind['WIFESOBS'] == 'NodAndShuffle':
            exptime = exptime + float(head_ind['EXPTIME']) + float(head_ind['SEXPTIME'])

    return date_obs, (exptime, 'sec, on-source'), (air_mass, 'airmass')

def update_wcs(filename, pix_scale, grism_list, root_path):
    
    hdu        = fits.open(filename)
    data       = hdu[0].data
    data_var   = hdu[1].data
    head       = hdu[0].header
    
    num_xspa   = head['NAXIS1']
    num_yspa   = head['NAXIS2']
    num_wv     = head['NAXIS3']
    objname    = head['object']
    det_wave   = head['CDELT3']
    
    pos         = head['RA']+' '+head['DEC'] 
    PA          =  -head['TELPAN']/180.0*np.pi    
    head2dwcs = create_head2dwcs(head, pos, PA, pix_scale, num_xspa, num_yspa)

    size_x           = np.abs(num_xspa*np.cos(PA)) + np.abs(num_yspa*np.sin(PA)) 
    size_y           = np.abs(num_xspa*np.sin(PA)) + np.abs(num_yspa*np.cos(PA)) 
    size_x, size_y   = int(size_x), int(size_y)
    data_rep         = np.zeros([size_y, size_x])

    head_rep = create_head2dwcs(head, pos, PA, pix_scale, num_xspa, num_yspa)
    head_rep['CRPIX1'], head_rep['CRPIX2'] = size_x/2, size_y/2
    head_rep['CD1_1'], head_rep['CD1_2']   = -(head2dwcs['CD1_1']**2 + head2dwcs['CD1_2']**2)**0.5, 0.0
    head_rep['CD2_1'], head_rep['CD2_2']   = 0.0, (head2dwcs['CD2_1']**2 + head2dwcs['CD2_2']**2)**0.5 
    hdu_rep = fits.PrimaryHDU(data=data_rep, header=head_rep)

    ## rotate for individual wavelength
    data_new = np.zeros([num_wv, size_y, size_x])
    data_var_new = np.zeros([num_wv, size_y, size_x])
    for id_wv in np.arange(0, num_wv):
        data_2d = data[id_wv,:,:]
        ## bad pixels
        id      = np.where(np.abs(data_2d) > 1.0e5)
        id      = np.concatenate(np.array(id))
        data_2d[id] = data_2d[id-1]
        ##
        hdu_2d                 = fits.PrimaryHDU(data=data_2d, header=head2dwcs)
        data_new2d, footprint  = reproject_exact(hdu_2d, hdu_rep.header)
        
        data_new[id_wv,:,:]    = data_new2d
    
        hdu_2d                     = fits.PrimaryHDU(data=data_var[id_wv,:,:], header=head2dwcs)
        data_var_new2d, footprint  = reproject_exact(hdu_2d, hdu_rep.header)
        data_var_new[id_wv,:,:]    = data_var_new2d
        
    #### update header
    # hdu         = fits.PrimaryHDU()
    new_head    = hdu[0].header
    
    new_head['CDELT1'],  new_head['CDELT2']   = head_rep['CD1_1'], head_rep['CD2_2']
    new_head['CDELT3']                        = det_wave
    
    new_head['CRPIX1'], new_head['CRPIX2']  = head_rep['CRPIX1'], head_rep['CRPIX2']
    new_head['CRPIX3']                      = 0
    
    new_head['CTYPE1'], new_head['CTYPE2']  = head_rep['CTYPE1'], head_rep['CTYPE2'] 
    new_head['CTYPE3']                      = 'wave'
    
    new_head['CUNIT1'], new_head['CUNIT2']  = head_rep['CUNIT1'], head_rep['CUNIT2']
    new_head['CUNIT3']                      = 'Angstrom'
    
    new_head['CRVAL1'], new_head['CRVAL2']  = head2dwcs['CRVAL1'], head2dwcs['CRVAL2']
    new_head['CRVAL3']                      = head['CRVAL3']
    
    new_head['OBJECT']   = head['OBJECT']
    new_head['OBSERVER'] = head['OBSERVER']
    new_head['INSTRUME'] = head['INSTRUME']
    new_head['TELESCOP'] = head['TELESCOP']

    new_head['GRATINGR'] = ', '.join(grism_list)
    new_head['CAMERA']   = 'WiFeSRed,WiFeSBlue'
    
    for grism in grism_list:
        file_grism = glob.glob(root_path + "/reduced_file/" + objname + '_' + grism + '*.fits')
        new_head['DATE-'+grism[0]], new_head['exp-'+grism[0]], new_head['am-'+grism[0]] = return_date_obs(file_grism)


    hdu_prim       = fits.PrimaryHDU(data=data_new, header=new_head)
    hdu_err        = fits.ImageHDU(data=data_var_new, header=new_head)
    hdulist        = fits.HDUList([hdu_prim, hdu_err])
    
    save_file = root_path+'/'+objname+'_updated_wcs.fits'
    hdulist.writeto(save_file, overwrite=True)    

    return hdulist



    












