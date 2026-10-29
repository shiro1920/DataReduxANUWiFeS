#!/usr/bin/env python
# Zhiyuan - 2025/10/08
# develope environment, python 3.10.16

__all__ = [
    "FileSetup",
    "SkySubstraction",
    "CombineEpoch",
    "GenerateBroadbandImages",
    "GrismFluxNormalize",
    "CombineGrism",
    "UpdateWCS",
    "DeRedden",
    "CombineMultiFoVs",
    "PipelineStep",
    "IFUPipeline",
]


import os
import glob
import yaml
import pyphot
import csv
import logging 
import numpy as np
import astropy as AP
import astropy.units as u
import warnings

from astropy import wcs
from astropy.table import Table
from astropy.io import ascii,fits
from astropy.coordinates import SkyCoord
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_adaptive

from scripts.loadfile import load_fitsfile
from scripts.mmm import mmm
from scripts.sex2dd import sex2dd
from scripts.find_skylines import define_strongskyline
from scripts.generate_wave_grid import wave_grid
from scripts.spec_projection import re_norm, resample_spec, resample_spec_fluxconv  
from scripts.gene_bbimg import produce_broadband
from scripts.PhotCompare import PhotCompare
from scripts.update_wcs import update_wcs
from scripts.deredden import unRedDatacube
from scripts.frame_projection import create_new_head_formosaic, stack_all_with_compress2D, stack_all_with_compress3D



## gengerate log 
def setup_advanced_logging(
    log_dir='logs/',
    prefix='WiFeS_post_process',
    log_level=logging.INFO,
    console_output=False,
    separate_error_log=True,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5
):
    
    """
    Log configuration, supporting log rotation, 
    separate error logging, etc.
    
    Args:
        log_dir: saving folder of log files
        prefix: user defined file name prefix
        log_level: log level, default:logging.INFO
        console_output: if print logs in console, default: False
        separate_error_log: if generate separate error log file, default: True
        max_bytes: max size of single log file, default: 10 MB
        backup_count: max file copys of log files, default: 5
    
    Returns:
        dict: includign logger and log file paths
    """
    from logging.handlers import RotatingFileHandler
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_log = log_path / f"{prefix}_{timestamp}.log"
    error_log = log_path / f"{prefix}_{timestamp}_error.log"
    
    # create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # main format of log file 
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # single format for console
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # main log file 
    main_handler = RotatingFileHandler(
        info_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    main_handler.setLevel(log_level)
    main_handler.setFormatter(detailed_formatter)
    logger.addHandler(main_handler)
    
    # error log file
    if separate_error_log:
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    # print logs in console
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    logger.info("=" * 70)
    logger.info("Log File Initialization Completed!")
    logger.info(f"Main Log File: {info_log.absolute()}")
    if separate_error_log:
        logger.info(f"Error Log File: {error_log.absolute()}")
    logger.info(f"[34m Log Level: {logging.getLevelName(log_level)}")
    logger.info("=" * 70)
    
    return {
        'logger': logger,
        'info_log': str(info_log),
        'error_log': str(error_log) if separate_error_log else None,
        'log_dir': str(log_path)
    }


## pipeline step parent class
class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.

    Args:

    """
    
    def __init__(self, name: str, n_jobs=1):
        self.name = name
        self.logger = logging.getLogger(f"Pipeline.{name}")
        
        # Explicitly serial
        if n_jobs == 1:
            self.n_jobs = 1

        # Use all cores
        elif n_jobs == -1:
            self.n_jobs = cpu_count()

        # Auto: use half cores
        elif n_jobs == 0:
            self.n_jobs = max(1, cpu_count() // 2)  

        # Limit to available cores
        elif n_jobs > cpu_count():
            self.logger.warning(
                f" Requested {n_jobs} cores but only {cpu_count()} available, "
                f"using {cpu_count()} cores"
            )
            self.n_jobs = cpu_count()
        # User defined cpu numbers but limit to the total cpus 
        else:
            self.n_jobs = min(n_jobs, cpu_count()) 
    
    @abstractmethod
    def run(self, obj_name, root_path, pointing, **kwargs):
        """Execute processing step"""
        pass
    
    def __call__(self, obj_name, root_path, pointing, **kwargs):
        self.logger.info(f"➤ Start: {self.name} for {obj_name}")
        start_time = datetime.now()
        try:
            result = self.run(obj_name, root_path, pointing, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✓ Complete: {self.name} ({duration:.2f}s)")
            return result
        except Exception as e:
            self.logger.error(f"✗ Error: {self.name} - {str(e)}")
            raise


## Step >> File setup, rename files, calculate exposure time
class FileSetup(PipelineStep):
    """
    Step 1: File setup, rename files, calculate exposure time

    Args:
        PipelineStep (class): abstract base class 
    """

    def __init__(self, observatory='Siding Spring Observatory', n_jobs=1):
        super().__init__("FileSetup", n_jobs=n_jobs)

        self.observatory  = str(observatory)
    
    def run(self, obj_name, root_path, pointing, **kwargs):
        """
        run file setup

        Args:
            obj_name (string): object name, example: 'NGC1800'
            root_path (string): the main path of all files and process, the 
            pointing (string): _description_
        """

        ## 
        sci_core_name = str(obj_name)
        sky_core_name = str(obj_name) + 'sky'
        pos_ra, pos_dec = sex2dd(pointing)

        ## load telescope location, default is 'Siding Spring Observatory'
        SSO = AP.coordinates.EarthLocation.of_site(self.observatory)

        ## default exposure time
        grism_list = []
        exp_U, exp_B, exp_R, exp_I = 0.0, 0.0, 0.0, 0.0

        ## load all frames
        frame_path = root_path + '/pipeline_frames/'
        filelist = glob.glob(frame_path + '*.fits')

        ## loop over all frames
        for k in tqdm(np.arange(len(filelist)), desc="Rename Files"):

            filename = filelist[k]
            self.logger.info(f"  Frame {filename}")
            
            head, wave, data, data_var = load_fitsfile(filename)

            ## rename head['object'] according to the angular seperation
            radec = '%s %s' % (head['RA'], head['DEC'])
            ra, dec = sex2dd(radec)
            angsep = 3600.0*((dec-pos_dec)**2 + (np.cos(np.radians(dec))*(ra-pos_ra))**2)**0.5

            angsep_cut = 10 
            head['object'] = sci_core_name if np.abs(angsep) < angsep_cut else sky_core_name
            core_name      = sci_core_name if np.abs(angsep) < angsep_cut else sky_core_name

            self.logger.debug(f"  Object: {head['object']}, angsep: {angsep:.2f}\"")

            ## correct the heliocentric velocity 
            T_radec = '%s %s' % (head['RA'], head['DEC'])
            T_ra, T_dec = sex2dd(T_radec)

            obsdate    = head['DATE-OBS'][0:10]
            pos        = AP.coordinates.SkyCoord(ra=T_ra*u.deg, dec=T_dec*u.deg)
            barycorr   = pos.radial_velocity_correction(obstime=AP.time.Time(obsdate), location=SSO) #in m/s
            z          = barycorr.value/3.0e8 
            self.logger.info(f"  Barycentric correction: {barycorr.value:.2f} m/s")
            
            head['CRVAL3'] = head['CRVAL3']*(1.0+z)
            head['CDELT3'] = head['CDELT3']*(1.0+z)

            ## exposure time 
            grism    = ''

            ## distinguish observation mode and load exp time from header
            if head['WIFESOBS'] == 'NodAndShuffle':
                sci_exp = head['EXPTIME']
                sky_exp = head['SEXPTIME']
                exp = sci_exp + sky_exp
            if head['WIFESOBS'] == 'ClassicalEqual':
                exp = head['exptime']

            ## calculate exposure time 
            if head['CAMERA'] == 'WiFeSBlue':

                if head['BEAMSPLT'] == 'RT560':         ## B3000
                    grism = head['GRATINGB']
                    grism_list.append(grism)
                    if 'sky' not in head['object']:
                        exp_B += exp

                elif head['BEAMSPLT'] == 'RT615':       ## B7000
                    grism = head['GRATINGB']
                    grism_list.append(grism)
                    if 'sky' not in head['object']:
                        exp_B += exp

                elif head['BEAMSPLT'] == 'RT480':       ## U7000
                    grism = head['GRATINGB']
                    grism_list.append(grism)
                    if 'sky' not in head['object']:
                        exp_U += exp


            if head['CAMERA'] == 'WiFeSRed':

                if head['BEAMSPLT'] == 'RT560':        ## R3000  
                    grism = head['GRATINGR']
                    grism_list.append(grism)
                    if 'sky' not in head['object']:
                        exp_R += exp

                elif head['BEAMSPLT'] == 'RT615':       ## I7000
                    grism = head['GRATINGR']
                    grism_list.append(grism)
                    if 'sky' not in head['object']:
                        exp_I += exp

                elif head['BEAMSPLT'] == 'RT480':       ## U7000
                    grism = head['GRATINGR']
                    grism_list.append(grism)
                    if 'sky' not in head['object']:
                        exp_R += exp

            ## save the result
            hdu_prim       = fits.PrimaryHDU(data=data, header=head)
            hdu_var        = fits.ImageHDU(data=data_var, header=head)
            hdulist        = fits.HDUList([hdu_prim, hdu_var])
            newfile  = root_path+'/'+core_name+'_'+grism+'_'+head['DATE-OBS']+'.fits'
            hdulist.writeto(newfile, overwrite=True)

            self.logger.info(f"  rename frame done!")

        unique_grism = list(set(grism_list))
        with open(root_path + '/logs/exposure_time.txt', 'w') as f:
            f.write(
                'Total exposure time in seconds: \n' + \
                'U: ' + str(exp_U) + 
                ', B: ' + str(exp_B) + 
                ', R: ' + str(exp_R) + 
                ', I: ' + str(exp_I) +  
                '\ngrisms,' + ''.join(str(i)+',' for i in unique_grism)
            )
        f.close()

        self.logger.info(f"- calculate total exposure time done!")
        self.logger.info(f"- Total exposure times - U:{exp_U}s, B:{exp_B}s, R:{exp_R}s, I:{exp_I}s")

        reduced_path = Path(root_path+'/reduced_file/')
        reduced_path.mkdir(parents=True, exist_ok=True)

        return
    
## Step >> subtract sky for blank sky observation mode
class SkySubstraction(PipelineStep):
    """Subtract sky for blank sky observation mode"""
    
    def __init__(self, n_jobs=1):
        super().__init__("SkySubtraction", n_jobs=n_jobs)

    def subtract_sky(self, data, data_var, head_data, filesky, grism):

        mjd_sky = np.zeros(len(filesky))
        for id in np.arange(len(filesky)):
            hdu         = fits.open(filesky[id])[0]
            head        = hdu.header
            mjd_sky[id] = head['MJD-OBS']
            
        mjd_obs = head_data['MJD-OBS']
        id_min  = np.argmin(np.abs(mjd_obs-mjd_sky))
        hdu     = fits.open(filesky[id_min])[0]
        data_sky= hdu.data
        head_sky= hdu.header
        num_wv  = head_sky['NAXIS3'] 
        
        frac_skyerr_dataerr = np.zeros(num_wv)
        for k in np.arange(num_wv):
            sky_img = data_sky[k,:,:]
            id_nonan = ~np.isnan(sky_img)
            
            if np.sum(id_nonan) > 50:
                skymed, skystd, skew = mmm(sky_img)
                n_pixel = sky_img.shape[0]*sky_img.shape[1]
                skystd_mean = skystd/(n_pixel-1)**0.5
                data[k,:,:]          = data[k,:,:]- skymed
                data_var[k,:,:]      = np.abs(data_var[k,:,:]) + skystd_mean**2

                # ## compare sky level with flux error 
                frac_skyerr_toterr   = skystd_mean/np.median(data_var[k,:,:]**0.5)
                frac_skyerr_dataerr[k] = frac_skyerr_toterr
            else:
                self.logger.info(f"sky pixels < 50 for wave grid={k}, setting data/data_var to nan")
                data[k,:,:]          = np.nan
                data_var[k,:,:]      = np.nan
                
        #mask out strong sky lines
        if grism != 'U7000': 
            cut = 10
            if grism == 'I7000': 
                cut = 4
                
            wave_skyline     = define_strongskyline(data_sky, head_sky, cut, degree=3)
            id               = np.where(wave_skyline == 1)
            data_var[id,:,:] = -data_var[id,:,:]
        
        return data, data_var, frac_skyerr_dataerr
        

    def run(self, obj_name, root_path, pointing, **kwargs):
        """Execute sky subtraction"""

        ## load grisms
        exp_file = Path(root_path + '/logs/exposure_time.txt')
        if not exp_file.exists():
            self.logger.error(f"✗ Exposure time file not found: {exp_file}")
            return
        
        grism_list = list(csv.reader(open(exp_file)))[2][1:-1]

        for id_grism in np.arange(len(grism_list)):
            fileobj = glob.glob(root_path + obj_name + '_' + grism_list[id_grism]+"*.fits")
            filesky = glob.glob(root_path + obj_name + 'sky_' +grism_list[id_grism]+"*.fits")
        
            if len(fileobj) > 0:
                for k in tqdm(np.arange(len(fileobj)), desc="Substract Sky"):
                    filename = fileobj[k]
                    aa       = filename.split('/')[-1]
                    corename = aa.split('.fits')[0]
                    
                    head, wave, data, data_var = load_fitsfile(filename)

                    data_sub, data_var_sub, frac_skyerr_dataerr = self.subtract_sky(data, data_var, head, filesky, grism_list[id_grism])
                    
                    hdu_prim       = fits.PrimaryHDU(data=data_sub, header=head)
                    hdu_var        = fits.ImageHDU(data=data_var_sub, header=head)
                    hdulist        = fits.HDUList([hdu_prim, hdu_var])
                    hdulist.writeto(root_path + '/'+corename+'_substract_sky.fits', overwrite=True)

                    data = Table()
                    data['wave_grid']      = wave
                    data['skyerr2dataerr'] = frac_skyerr_dataerr
                    ascii.write(data, root_path + '/'+corename+'_skyerr2dataerr.dat', format='fixed_width', overwrite=True)

                    self.logger.info(f"sky subtraction done!  {filename}")

        os.system('mv '+root_path+'/*.fits'+' '+root_path+'/reduced_file/')
        os.system('mv '+root_path+'/*.dat'+' '+root_path+'/reduced_file/')
        os.system('mv '+root_path+'/reduced_file/'+ '*substract_sky*.fits'+' '+root_path+'/')

        return

## Step >> Combine single frames into one datacube 
class CombineEpoch(PipelineStep):
    """
    """

    def __init__(self, n_jobs=2):
        super().__init__("CombineEpoch", n_jobs=n_jobs)

    def combine_epoch(self, args):

        obj_name, root_path, grism = args

        fileobj            = glob.glob(root_path + "/*" + grism + "*.fits")
        head               = fits.getheader(fileobj[0])
        num_xspa, num_yspa = head['NAXIS1'], head['NAXIS2']
        
        ## define new wave grid that is the shortest one
        new_wave, id_head_new_wave, head_NAXIS3 = wave_grid(fileobj)
        
        num_wave         = len(new_wave)
        num_obj          = len(fileobj)
        data_all         = np.zeros((num_obj, num_wave, num_yspa, num_xspa))
        data_var_all     = np.zeros((num_obj, num_wave, num_yspa, num_xspa))
        strline_all      = np.zeros((num_obj, num_wave, num_yspa, num_xspa))
        
        ## loop over all files 
        for id_file in np.arange(0, num_obj):
            self.logger.info(f'-- {grism}: {fileobj[id_file]}')
            filename      = fileobj[id_file]
            head, old_wave, data, data_var = load_fitsfile(filename)

            for idx in np.arange(0, num_xspa):
                for idy in np.arange(0, num_yspa):
                    old_flux      = data[:,idy,idx]
                    old_var       = data_var[:,idy,idx]
                    strline       = np.zeros(len(old_wave))
                    
                    id = np.where(old_var < 0)
                    old_var[id] = -old_var[id]
                    strline[id] = 1.0
                    
                    ##replace nan with zero
                    num_nan = np.sum(np.isnan(old_flux))
                    if num_nan > 0:
                        #print('# of wave elements are nan, setting to zero and strline to 1.0:', num_nan)
                        id           = ( np.isnan(old_flux) | np.isnan(old_var) )
                        old_flux[id] = 0.0
                        old_var[id]  = 0.0
                        strline[id]  = 1.0 
            
                    ##resample flux-converved
                    old_err           = old_var**0.5
                    new_flux, new_err = resample_spec_fluxconv(new_wave, old_wave, old_flux, old_err, extrap_treatment='zero')
                    new_var           = new_err**2
                    new_strline, ignore = resample_spec_fluxconv(new_wave, old_wave, strline, strline, extrap_treatment='zero')
                    
                    data_all[id_file, :, idy, idx]      = new_flux
                    data_var_all[id_file, :, idy, idx]  = new_var
                    strline_all[id_file, :, idy, idx]   = new_strline 

        
        data_combine     = np.mean(data_all, axis=0)                        #combine through mean
        data_var_combine = np.sum(data_var_all, axis=0)/len(fileobj)**2     #
        strline_combine  = np.max(strline_all, axis=0)      
        id               = np.where( strline_combine > 0 )
        data_var_combine[id] = -np.abs(data_var_combine[id])

        ## update header 
        head['CRVAL3'] = new_wave[0] 
        head['NAXIS3'] = len(new_wave)
        head['CDELT3'] = new_wave[1]-new_wave[0]

        ## save to fits file
        hdu_prim       = fits.PrimaryHDU(data=data_combine, header=head)
        hdu_err        = fits.ImageHDU(data=data_var_combine, header=head)
        hdulist        = fits.HDUList([hdu_prim, hdu_err])
        hdulist.writeto(root_path + '/'+obj_name+'_combined_'+grism+'.fits', overwrite=True)

        self.logger.info(f"{obj_name}: {grism} combination done!")

        return


    def run(self, obj_name, root_path, pointing, **kwargs):

        ## load grisms
        grism_list = list(csv.reader(open(root_path + '/logs/exposure_time.txt')))[2][1:-1]
        num_grisms = len(grism_list)

        args_list = [
                (obj_name, root_path, grism_list[i])
                for i in range(len(grism_list))
            ]

        with Pool(self.n_jobs) as pool:
            results = list(
                tqdm(pool.imap(self.combine_epoch, args_list),
                total=num_grisms,
                desc="Combine Epoch")
            )

        for grism in grism_list:
            os.system('mv '+root_path+'/*'+obj_name+'_'+grism+'*sky*.fits'+' '+root_path+'/reduced_file/')

        return

## Step >> produce fake broadband image 
class GenerateBroadbandImages(PipelineStep):

    def __init__(self, filter_name=None, n_jobs=1):
        super().__init__("GenerateBroadbandImages", n_jobs=n_jobs)
        self.filter_name = filter_name 

    def run(self, obj_name, root_path, pointing, **kwargs):

        file_list = glob.glob(root_path + '/*.fits')

        if len(file_list) > 1:
            for i in range(len(file_list)):
                file_name = file_list[i]
                if 'U7000' in file_name:
                    filter_name = self.filter_name if self.filter_name is not None else 'SkyMapper_v'
                    grism = 'U7000'
                if 'B7000' in file_name:
                    filter_name = self.filter_name if self.filter_name is not None else 'SkyMapper_g'
                    grism = 'B7000'
                if 'R7000' in file_name:
                    filter_name = self.filter_name if self.filter_name is not None else 'SkyMapper_r'
                    grism = 'R7000'
                if 'I7000' in file_name:
                    filter_name = self.filter_name if self.filter_name is not None else 'SkyMapper_i'
                    grism = 'I7000'
                if 'B3000' in file_name:
                    filter_name = self.filter_name if self.filter_name is not None else 'SkyMapper_g'       ## 待完善
                    grism = 'B3000'
                if 'R3000' in file_name:
                    filter_name = self.filter_name if self.filter_name is not None else 'SkyMapper_r'       ## 待完善         
                    grism = 'R3000'


                save_file_name = root_path + '/' + obj_name + '_' + grism + '_' + filter_name + '_fakeimg.fits'
                head, wave, data, data_var = load_fitsfile(file_name)
                img_data = produce_broadband(head, data, wave, save_file_name, filter_name, save2fits=True)

        elif len(file_list) == 1:

            filter_list = ['SkyMapper_v', 'SkyMapper_g', 'SkyMapper_r', 'SkyMapper_i']
            head, wave, data, data_var = load_fitsfile(file_list[0])
            wave = 10**wave if wave[0] < 10 else wave

            if self.filter_name == None:
                self.logger.info(f"No input filter name, use dafult filters of skymapper v, g, r, i.")

                for filter_name in filter_list:
                    save_file_name = root_path + '/' + obj_name + '_' + filter_name + '_fakeimg.fits'
                    img_data = produce_broadband(head, data, wave, save_file_name, filter_name, save2fits=True)
            else:
                if len(self.filter_name) > 1:

                    filter_list = list(self.filter_name)
                    for filter_name in filter_list:
                        save_file_name = root_path + '/' + obj_name + '_' + filter_name + '_fakeimg.fits'
                        img_data = produce_broadband(head, data, wave, save_file_name, filter_name, save2fits=True)
                        
                else:
                    filter_name = self.filter_name
                    save_file_name = root_path + '/' + obj_name + '_' + filter_name + '_fakeimg.fits'
                    img_data = produce_broadband(head, data, wave, save_file_name, filter_name, save2fits=True)

        else:
            self.logger.info('No file input')

        return
    
## Step >> estimate normolized factor for each grism datacube by compare with the skymapper broadband image 
class GrismFluxNormalize(PipelineStep):
    def __init__(self, filter_name=None, n_jobs=1):
        super().__init__("GrismFluxNormalize", n_jobs=n_jobs)
        self.filter_name = filter_name

    def run(self, obj_name, root_path, pointing, **kwargs):

        grism_list = list(csv.reader(open(root_path + '/logs/exposure_time.txt')))[2][1:-1]

        if self.filter_name == None:

            self.logger.info(f"No input filter name, use dafult filters of skymapper v, g, r, i.")
            bbimg_path = root_path + '/broadband_image/'

            if len(grism_list) == 4:
                filter_list = ['SkyMapper_v', 'SkyMapper_g', 'SkyMapper_r', 'SkyMapper_i']
                grisms = grism_list

            elif len(grism_list) == 2:
                filter_list = ['SkyMapper_g', 'SkyMapper_r']
                grisms = grism_list

            phot_results = Table()
            phot_results['grism'] = grisms
            phot_results['fake_img_flux'] = np.zeros(len(grisms))
            phot_results['bb_img_flux'] = np.zeros(len(grisms))
            phot_results['norm_factor'] = np.zeros(len(grisms)) 

            for i in range(len(filter_list)):
                fake_bbimg = glob.glob(root_path + '*' + filter_list[i] + '*.fits')[0]
                bbimg      = glob.glob(bbimg_path + '*' + filter_list[i] + '*.fits')[0]
                blank_sky  = glob.glob(bbimg_path + '*.reg')[0]
                pix_size   = 0.5 
                grism_result = PhotCompare(fake_bbimg, bbimg, pointing, blank_sky, pix_size, filter_list[i])
                phot_results['fake_img_flux'][i] = grism_result['fake_bbimg_flux']
                phot_results['bb_img_flux'][i] = grism_result['bbimg_flux']
                phot_results['norm_factor'][i] = grism_result['fake2bbimg_ratio']
            
            ascii.write(phot_results, root_path + '/logs/' + obj_name + 'grisms_normlize_factor.dat', format='fixed_width', overwrite=True)

        return 

## Step >> Combine all grisms into one datacube
class CombineGrism(PipelineStep):

    def __init__(self, n_jobs=4):
        super().__init__("CombineGrism", n_jobs=n_jobs)

    
    def CombineGrism(self, args):
        idx, root_path = args
    
        ## load data
        file_all = glob.glob(root_path+'/*combined*.fits')
        grism_list = list(csv.reader(open(root_path + '/logs/exposure_time.txt')))[2][1:-1]
        norm_df = ascii.read(glob.glob(root_path + '/logs/*norm*.dat')[0], format='fixed_width')
        norm_grism = norm_df['grism']
        norm_factors = norm_df['norm_factor']

        for file in file_all:
    
            if 'U7000' in file:
                head_U, wave_U, data_U, data_var_U = load_fitsfile(glob.glob(root_path+'/*U7000*.fits')[0])
                #remove about 10 A to avoid the anomaly at the edge
                data_U           = data_U[0:-30,:,:]       
                data_var_U       = data_var_U[0:-30,:,:]
                wave_U           = wave_U[0:-30]
                head_U['NAXIS3'] = head_U['NAXIS3']-30
                id_U = np.where(norm_grism == 'U7000')
                norm_U = norm_factors[id_U]
            
            if 'B3000' in file:
                head_B, wave_B, data_B, data_var_B = load_fitsfile(glob.glob(root_path+'/*B3000*.fits')[0])
                id_B = np.where(norm_grism == 'B3000')
                norm_B = norm_factors[id_B]
                
            if 'B7000' in file:
                head_B, wave_B, data_B, data_var_B = load_fitsfile(glob.glob(root_path+'/*B7000*.fits')[0])
                id_B = np.where(norm_grism == 'B7000')
                norm_B = norm_factors[id_B]
                
            if 'R3000' in file:
                head_R, wave_R, data_R, data_var_R = load_fitsfile(glob.glob(root_path+'/*R3000*.fits')[0])
                id_R = np.where(norm_grism == 'R3000')
                norm_R = norm_factors[id_R]
            
            if 'R7000' in file:
                head_R, wave_R, data_R, data_var_R = load_fitsfile(glob.glob(root_path+'/*R7000*.fits')[0])
                id_R = np.where(norm_grism == 'R7000')
                norm_R = norm_factors[id_R]
            
            if 'I7000' in file:
                head_I, wave_I, data_I, data_var_I = load_fitsfile(glob.glob(root_path+'/*I7000*.fits')[0])
                id_I = np.where(norm_grism == 'I7000')
                norm_I = norm_factors[id_I]

        ## set new wave grid
        wave_min, wave_max    = 3500, 9000
        dlogwave              = 1.0/7000.0/np.log(10)/2
        nwave                 = ( np.log10(wave_max) - np.log10(wave_min) )/dlogwave
        logwavenew            = np.log10(wave_min) + np.arange(nwave) * dlogwave
        new_wave              = 10**logwavenew
        num_wave              = len(new_wave)
        num_xspa, num_yspa    = head_R['NAXIS1'], head_R['NAXIS2']

        ## combine grism process
        data_givenx         = np.zeros((num_yspa, num_wave))
        data_var_givenx     = np.zeros((num_yspa, num_wave))
        for idy in np.arange(0, num_yspa):
            fl_B  = data_B[:,idy,idx] / norm_B
            var_B = data_var_B[:,idy,idx] / norm_B

            if len(grism_list) > 2:
                fl_U, var_U   = re_norm(wave_B, fl_B, wave_U, data_U[:,idy,idx] / norm_U, data_var_U[:,idy,idx] / norm_U)
                fl_R, var_R   = re_norm(wave_B, fl_B, wave_R, data_R[:,idy,idx] / norm_R, data_var_R[:,idy,idx] / norm_R)
                fl_I, var_I   = re_norm(wave_R, fl_R, wave_I, data_I[:,idy,idx] / norm_I, data_var_I[:,idy,idx] / norm_I)
        
                new_fl1, new_var1, new_strline1 = resample_spec(new_wave, wave_U, fl_U, var_U)
                new_fl2, new_var2, new_strline2 = resample_spec(new_wave, wave_B, fl_B, var_B)
                new_fl3, new_var3, new_strline3 = resample_spec(new_wave, wave_R, fl_R, var_R)
                new_fl4, new_var4, new_strline4 = resample_spec(new_wave, wave_I, fl_I, var_I)
        
                #
                weight1 = np.zeros(len(new_wave))
                id      = np.where(new_var1 > 0)
                weight1[id] = 1.0/new_var1[id]
        
                weight2 = np.zeros(len(new_wave))
                id      = np.where(new_var2 > 0)
                weight2[id] = 1.0/new_var2[id]
        
                weight3 = np.zeros(len(new_wave))
                id      = np.where(new_var3 > 0)
                weight3[id] = 1.0/new_var3[id]
        
                weight4 = np.zeros(len(new_wave))
                id      = np.where(new_var4 > 0)
                weight4[id] = 1.0/new_var4[id]
        
                new_fl  = ( new_fl1*weight1 + new_fl2*weight2 + new_fl3*weight3 + new_fl4*weight4) / (weight1 + weight2 + weight3 + weight4)
                new_var = ( new_var1*weight1**2 + new_var2*weight2**2 + new_var3*weight3**2 + new_var4*weight4**2) / (weight1 + weight2 + weight3 + weight4)**2
                new_strline = new_strline1 + new_strline2 + new_strline3 + new_strline4
                id          = np.where(new_strline > 0)
                new_var[id] = -new_var[id]

            else:
                
                fl_R, var_R   = re_norm(wave_B, fl_B, wave_R, data_R[:,idy,idx] / norm_R, data_var_R[:,idy,idx] / norm_R)
        
                new_fl2, new_var2, new_strline2 = resample_spec(new_wave, wave_B, fl_B, var_B)
                new_fl3, new_var3, new_strline3 = resample_spec(new_wave, wave_R, fl_R, var_R)
        
                #
                weight2= np.zeros(len(new_wave))
                id     = np.where(new_var2 > 0)
                weight2[id] = 1.0/new_var2[id]
        
                weight3= np.zeros(len(new_wave))
                id     = np.where(new_var3 > 0)
                weight3[id] = 1.0/new_var3[id]
        
                new_fl = (new_fl2*weight2 + new_fl3*weight3)/( weight2 + weight3)
                
                new_var= (new_var2*weight2**2 + new_var3*weight3**2)/(weight2 + weight3)**2
                
                new_strline = new_strline2 + new_strline3
        
                id          = np.where(new_strline > 0)
                new_var[id] = -new_var[id]
                

            data_givenx[idy,:]     = np.transpose(new_fl)
            data_var_givenx[idy,:] = np.transpose(new_var)

        return data_givenx, data_var_givenx


    def run(self, obj_name, root_path, pointing, **kwargs):

        head, wave, data, data_var = load_fitsfile(glob.glob(root_path+'/*combined*.fits')[0])
        num_xspa, num_yspa = head['NAXIS1'], head['NAXIS2']

        ## set new wave grid
        wave_min, wave_max    = 3500, 9000
        dlogwave              = 1.0/7000.0/np.log(10)/2
        nwave                 = ( np.log10(wave_max) - np.log10(wave_min) )/dlogwave
        logwavenew            = np.log10(wave_min) + np.arange(nwave) * dlogwave

        args_list = [
                (idx, root_path)
                for idx in range(num_xspa)
            ]

        with Pool(self.n_jobs) as pool:
            results = np.array(list(
                tqdm(pool.imap(self.CombineGrism, args_list),
                total=num_xspa,
                desc="Combine Epoch")
            ))

        data_combine     = np.transpose(results[:,0,:,:])
        data_var_combine = np.transpose(results[:,1,:,:])

        head['CRVAL3'] = (logwavenew[0], ', Reference wavelength in logarithm')
        head['NAXIS3'] = len(logwavenew) 
        head['CDELT3'] = (logwavenew[1] - logwavenew[0], ', Wavelength step in logarithm' )

        hdu_prim       = fits.PrimaryHDU(data=data_combine, header=head)
        hdu_err        = fits.ImageHDU(data=data_var_combine, header=head)
        hdulist        = fits.HDUList([hdu_prim, hdu_err])
        hdulist.writeto(root_path+'/'+obj_name+'.fits', overwrite=True)
        os.system('mv '+root_path+'/*'+obj_name + '*combined*.fits'+' '+root_path+'/reduced_file/')

        return
    
## Step >> correct PA, update header with wcs info and exposure time, airmass
class UpdateWCS(PipelineStep):

    def __init__(self, n_jobs=1):
        super().__init__("UpdateWCS", n_jobs=n_jobs)

        

    def run(self, obj_name, root_path, pointing, **kwargs):
        warnings.filterwarnings("ignore", message="Calling _reproject_celestial")

        file_path  = glob.glob(root_path + '/' + obj_name + '.fits')[0]
        grism_list = list(csv.reader(open(root_path + '/logs/exposure_time.txt')))[2][1:-1]
        pix_scale  = 1.0/3600.0          #arcsec ANU/WiFeS, in unit of degree

        hdu = update_wcs(file_path, pix_scale, grism_list, root_path)

        os.system('mv '+root_path+'/' +obj_name+'.fits'+' '+root_path+'/reduced_file/')

        return
    
## Step >> correct galactic extinction
class DeRedden(PipelineStep):

    def __init__(self, n_jobs=1):
        super().__init__("DeRedden", n_jobs=n_jobs)

        
    def run(self, obj_name, root_path, pointing, **kwargs):

        dustmap_path = 'sfddata/'
        file_name = glob.glob(root_path + '/' + '*wcs*.fits')[0]
        unRedDatacube(file_name, dustmap_path, root_path)
        os.system('mv '+root_path+'/' +obj_name+'*wcs*.fits'+' '+root_path+'/reduced_file/')
        
        return

## Step >> combine multi field-of-view of single target
class CombineMultiFoVs(PipelineStep):

    def __init__(self, n_jobs=1, num_obs=10,):
        super().__init__("CombineMultiFoVs", n_jobs=n_jobs)

        # the maximum visit for single pixel, default is 10, 
        self.num_visit = num_obs

    def generate_mosic_frames(self, obj_name, root_path, **kwargs):

        pix_scale =  1.0/3600.0                     #arcsec
        file_list = glob.glob(root_path + '*deRedden*.fits')
        img_shape = (fits.getdata(file_list[0]).shape[1], fits.getdata(file_list[0]).shape[2])

        tar_info = []
        for i in range(len(file_list)):
            file_info = [img_shape]
            file_info.append(wcs.WCS(fits.getheader(file_list[i]), naxis=2))
            tar_info.append(tuple(file_info))

        wcs_out, shape_out = find_optimal_celestial_wcs(tar_info)
        ra_cen, dec_cen = wcs_out.wcs.crval[0], wcs_out.wcs.crval[1]
        size_y, size_x = shape_out[0], shape_out[1]

        ## create a wcs for the final mosaic; wcs is north-up, left-east
        hdu           = fits.PrimaryHDU()
        head_mosaic   = hdu.header
        head_mosaic['CRPIX1'], head_mosaic['CRPIX2'] = size_x/2.0, size_y/2.0
        head_mosaic['CRVAL1'], head_mosaic['CRVAL2'] = ra_cen, dec_cen
        head_mosaic['CTYPE1'], head_mosaic['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
        head_mosaic['CDELT1'], head_mosaic['CDELT2'] = (-pix_scale, 'deg'), (pix_scale, 'deg')

        data_mosaic     = np.zeros([size_y, size_x])
        hdu_mosaic      = fits.PrimaryHDU(data=data_mosaic, header=head_mosaic)
        exp_mosaic      = np.zeros([size_y, size_x])
        visit_mosaic    = np.zeros([size_y, size_x])

        ## reproject individual frames
        for i in range(len(file_list)):

            ## load wcs info for every frame
            file_name  = file_list[i]
            head, wave, data, data_var = load_fitsfile(file_name)

            head2dwcs = wcs.WCS(head, naxis=2)

            data_new, footprint      = reproject_adaptive((data, head2dwcs), hdu_mosaic.header, conserve_flux=True)
            data_var_new, footprint  = reproject_adaptive((data_var, head2dwcs), hdu_mosaic.header, conserve_flux=True)

            exp_mosaic   = footprint[0,:,:] * head['exptime']
            visit_mosaic = footprint[0,:,:]

            ##save the result
            new_head = create_new_head_formosaic(head, head_mosaic)
            hdu_prim       = fits.PrimaryHDU(data=data_new, header=new_head)
            hdu_err        = fits.ImageHDU(data=data_var_new, header=new_head)
            hdu_visit      = fits.ImageHDU(data=visit_mosaic, header=new_head)
            hdu_exp        = fits.ImageHDU(data=exp_mosaic, header=new_head)
            hdulist        = fits.HDUList([hdu_prim, hdu_err, hdu_visit, hdu_exp])
            core_name = file_name.split('/')[-1].split('_')[0]
            hdulist.writeto(root_path+core_name+'_Mosic.fits', overwrite=True)  

            self.logger.info(f"Create Mosic Image for {file_name} Done! ")

        ## move old files to reduced files
        os.system('mv '+root_path+'/' +obj_name+'*deRedden*.fits'+' '+root_path+'/reduced_file/')

        return
    
    def combine_FoVs(self, obj_name, root_path, **kwargs):

        file_list = glob.glob(root_path + '*Mosic*.fits')

        # data, var, visit time, exptime
        hdu = fits.open(file_list[0])                  
        num_wv, size_y, size_x = (hdu[0].data).shape

        # assume the maximum visit is 10 for a pixel
        num_visit   = self.num_visit
        stack_flux  = np.zeros([num_visit, num_wv, size_y, size_x]) + np.nan
        stack_var   = np.zeros([num_visit, num_wv, size_y, size_x]) + np.nan
        stack_visit = np.zeros([num_visit, size_y, size_x]) + np.nan
        stack_exp   = np.zeros([num_visit, size_y, size_x]) + np.nan

        for ifile in np.arange(len(file_list)):

            self.logger.info(f"stack file: {file_list[ifile]}")

            hdu          = fits.open(file_list[ifile])
            head         = hdu[0].header
            flux_map     = hdu[0].data
            var_map      = hdu[1].data
            visit_map    = hdu[2].data
            exptime_map  = hdu[3].data

            stack_flux  = stack_all_with_compress3D(stack_flux, flux_map)    
            stack_var   = stack_all_with_compress3D(stack_var, var_map)    
            stack_visit = stack_all_with_compress2D(stack_visit, visit_map)    
            stack_exp   = stack_all_with_compress2D(stack_exp, exptime_map)    


        ## combine all and save to single fits file
        head = hdu[0].header

        final_visit = np.nansum(stack_visit, axis=0)
        fits.writeto(root_path+obj_name+'_visit.fits', final_visit, header=head, overwrite=True)

        final_exp   = np.nansum(stack_exp, axis=0)
        fits.writeto(root_path+obj_name+'_exp.fits', final_exp, header=head, overwrite=True)

        final_flux  = np.nanmedian(stack_flux, axis=0)
        fits.writeto(root_path+obj_name+'_flux.fits', final_flux, header=head, overwrite=True)

        final_var     = np.nanmedian(stack_var, axis=0)
        fits.writeto(root_path+obj_name+'_var.fits', final_var, header=head, overwrite=True)

        self.logger.info(f"Combine {len(file_list)} FoVs for {obj_name} Done! ")

        os.system('mv '+root_path+'/' +obj_name+'*Mosic*.fits'+' '+root_path+'/reduced_file/')

        return 

    def run(self, obj_name, root_path, pointing, **kwargs):

        self.generate_mosic_frames(obj_name, root_path)
        self.combine_FoVs(obj_name, root_path)
        
        return

## ANU WiFeS data post-process main class
class IFUPipeline:
    """
    Post-processing pipeline for ANU/WiFeS datacubes reduced by *pywifes*.

    This pipeline performs a series of optional post-processing steps for 
    WiFeS datacubes, including automatic renaming, sky subtraction, grism combination,
    WCS correction, flux normalization, and extinction correction.

    Each processing step is modular and can be enabled or skipped independently.
    Intermediate results are saved under ``reduced_file/``, and all process logs
    are automatically recorded in ``logs/``.

    **Available processing steps:**

        - ``FileSetup`` — Automatically rename FITS files and separate science and blank-sky frames  
        - ``SkySubstraction`` — Subtract sky background for science frames  
        - ``CombineEpoch`` — Combine science frames of multiple exposures under the same grism  
        - ``GenerateBroadbandImages`` — Create broadband images  
        - ``GrismFluxNormalize`` — Calculate flux normalization factor  
        - ``CombineGrism`` — Combine datacubes of different grisms  
        - ``UpdateWCS`` — Update WCS and correct telescope PA  
        - ``DeRedden`` — Correct galactic extinction  
        - ``CombineMultiFoVs`` — Combine multiple field-of-view observations

    :param str obj_name: 
        Target object name. Example: ``'NGC1800'``.
    :param str root_path:
        Root path containing all input and output files. The raw pyWiFeS-reduced data
        should be under ``<root_path>/pipeline_frames/``.
        Example: ``'/NGC1800_files/'``.
    :param str pointing:
        Observing pointing coordinates in format ``'RA Dec'``.
        Example: ``'20:09:31.77 -61:51:01.70'``.

    :return: Final science datacube and reduction logs.
    :rtype: tuple

    **Example (single object):**

    .. code-block:: python

        from WiFeS_PostProcess import *

        obj_name  = 'IC4951'
        pointing  = '20:09:31.77 -61:51:01.7'
        root_path = 'obj_frames/'

        pipeline = IFUPipeline()

        # Configure logging
        log_info = pipeline.setup_advanced_logging(
            log_dir = root_path + 'logs/',
            prefix = obj_name + '_post_process',
            log_level = logging.INFO,
            console_output=False,
            separate_error_log=True,
            max_bytes=10*1024*1024,
            backup_count=5
        )

        # Add processing steps
        pipeline.add_step(FileSetup())
        pipeline.add_step(SkySubstraction())
        ...

        pipeline.run(obj_name=obj_name, root_path=root_path, pointing=pointing)

    **Example (multiple objects):**

    .. code-block:: python

        from WiFeS_PostProcess import *

        targets = [
            ('IC4951', 'IC4951_test/', '20:09:31.77 -61:51:01.7'),
            ('IC0559', 'IC0559_test/', '09:44:43.89 +09:36:54.0'),
        ]

        for obj_name, root_path, pointing in targets:
            pipeline = IFUPipeline()
            log_info = pipeline.setup_advanced_logging(
                log_dir=root_path + '/logs/',
                prefix=obj_name + '_post_process',
                log_level=logging.INFO,
                console_output=False,
                separate_error_log=True,
                max_bytes=10*1024*1024,
                backup_count=5
            )

            pipeline.add_step(FileSetup())
            pipeline.add_step(SkySubstraction())
            ...

            pipeline.run(obj_name=obj_name, root_path=root_path, pointing=pointing)

    Revison History:
        v1.0: developed by Zhiyuan Zheng, 2025/10/08
    """

    
    def __init__(self, config_file=None, log_file=None, log_level=logging.INFO):

        self.steps    = []
        self.config   = None
        if config_file:
            self.load_config(config_file)

    def setup_advanced_logging(self, log_dir='logs', **kwargs):
        """
        log configuration
        
        Args:
            log_dir: log save path
            **kwargs: other parameters
        """
        self.log_info = setup_advanced_logging(log_dir=log_dir, **kwargs)
        self.logger = logging.getLogger("IFUPipeline")
        return self.log_info
    
    def load_config(self, config_file):
        """ run pipeline by loading configuration files """
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger.info(f"from configuration file: {config_file}")
    
    def add_step(self, step):
        """add process steps"""
        self.steps.append(step)
        return self
    
    def run(self, obj_name, root_path, pointing, **kwargs):
        """run pipelins"""
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"data post-process started")
        self.logger.info(f"total {len(self.steps)} steps")
        self.logger.info(f"=" * 60)
        
        for i, step in enumerate(self.steps, 1):
            self.logger.info(f"\n ➤ Step {i}/{len(self.steps)}: {step.name}")
            self.logger.info(f"-" * 60)
            
            data = step(obj_name, root_path, pointing, **kwargs)
        
        self.logger.info(f"\n" + "=" * 60)
        self.logger.info(f"data post-process done!")
        self.logger.info(f"=" * 60)
        
        return data


