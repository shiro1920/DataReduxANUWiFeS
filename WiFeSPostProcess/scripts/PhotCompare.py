from astropy.io import fits
import numpy as np
from scripts.sex2dd import sex2dd
from astropy.coordinates import SkyCoord
import csv
from astropy import wcs
from photutils.aperture import aperture_photometry, RectangularAperture
import pyphot

def PhotCompare(fake_bbimg_path, bbimg_path, pointing, blank_sky_path, bbimg_pixsize, filter_name):

    pix_size = bbimg_pixsize
    pointing_deg = sex2dd(pointing)
    
    ## fake broadband image 
    f_bbimg = fits.open(fake_bbimg_path)
    f_img_data = f_bbimg[0].data
    f_img_head = f_bbimg[0].header
    PA = f_img_head['TELPAN'] if f_img_head['TELPAN'] > 0 else f_img_head['TELPAN'] + 360
    obs_PA = PA * np.pi / 180
    FoV_size_x, FoV_size_y = f_img_data.shape[1], f_img_data.shape[0]
    FoV_cen = SkyCoord(pointing_deg[0], pointing_deg[1], unit='deg')
    fimg_flux = np.nansum(f_img_data)
    
    ## archival broadband image 
    bbimg = fits.open(bbimg_path)
    img_data = bbimg[0].data
    img_head = bbimg[0].header
    img_wcs = wcs.WCS(img_head, naxis = 2)
    img_xc, img_yc = img_wcs.world_to_pixel(FoV_cen)
    img_aper_size_x, img_aper_size_y = FoV_size_x/pix_size, FoV_size_y/pix_size
    img_aperture = RectangularAperture([img_xc, img_yc], img_aper_size_x, img_aper_size_y, theta=obs_PA) 

    #- broadband image blank sky 
    reg_info = list(csv.reader(open(blank_sky_path)))[3]
    sky_cen_ra, sky_cen_dec = float(reg_info[0][4:]), float(reg_info[1])        ## in unit of deg
    sky_cen_coord = SkyCoord(sky_cen_ra, sky_cen_dec, unit='deg')               ## in unit of deg
    sky_cen_xc, sky_cen_yc = img_wcs.world_to_pixel(sky_cen_coord)
    width, height = np.abs(float(reg_info[2][:-1])) / pix_size, np.abs(float(reg_info[3][:-1])) / pix_size      ## in unit of pxiel 
    sky_xmin, sky_xmax = int(sky_cen_xc - width/2), int(sky_cen_xc + width/2)
    sky_ymin, sky_ymax = int(sky_cen_yc - height/2), int(sky_cen_yc + height/2)
    #- substract sky of broad band image
    img_sky = np.nanmean(img_data[sky_ymin:sky_ymax, sky_xmin:sky_xmax])
    img_subsky = img_data - img_sky

    phot_table = aperture_photometry(img_subsky, img_aperture)
    phot_bb    = float(phot_table['aperture_sum'])
    ZPAPPROX   = img_head['ZPAPPROX']
    ABmag      = ZPAPPROX - 2.5*np.log10(phot_bb)
    lib       = pyphot.get_library()
    f          = lib[filter_name]
    img_flux   = 10**(-ABmag/2.5)*3630.0*1.0e-23*3.0e18/f.leff.value**2   

    ## flux ratio
    fake2bbimg_ratio = fimg_flux/img_flux

    ## result 
    result = {
        'fake_bbimg_flux':fimg_flux, 
        'bbimg_flux':img_flux,
        'fake2bbimg_ratio':fake2bbimg_ratio
    }

    return result

