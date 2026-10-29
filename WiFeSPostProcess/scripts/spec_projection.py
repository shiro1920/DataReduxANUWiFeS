import numpy as np

def resample_spec_fluxconv(newwave, wave, flux, err, extrap_treatment = None):
    #newwave in AA
    #wave in AA
    #flux in erg/s/cm^2/AA
    #err  in erg/s/cm^2/AA
    #extrap_treatment = None: fill outside grids with nan
    #                 = zero: fill outside grids with zero
    def define_spec_edge(x):
        inside_edges = (x[1:] + x[:-1]) / 2
        edges = np.insert(inside_edges, 0, 2 * x[0] - inside_edges[0])
        edges = np.append(edges, 2 * x[-1] - inside_edges[-1])
        return edges

    ######define edges
    orig_edges = define_spec_edge(wave)
    fin_edges  = define_spec_edge(newwave)

    orig_low = orig_edges[:-1]
    fin_low  = fin_edges[:-1]
    orig_upp = orig_edges[1:]
    fin_upp  = fin_edges[1:]


    ########got the matrix of each original pixel's contributin to the final pixel
    l_inf = np.where(orig_low > fin_low[:, np.newaxis],
                         orig_low, fin_low[:, np.newaxis])
    l_sup = np.where(orig_upp < fin_upp[:, np.newaxis],
                         orig_upp, fin_upp[:, np.newaxis])

    ##   f=sum(f*resamp_detlam)/sum(resamp_detlam)
    ##   err = ( sum(err**2 * resamp_detlam * orig_detlam) )**2/sum(resamp_detlam)
    resamp_detlam   = (l_sup - l_inf).clip(0)  #the contribution in wavelength of each wave-grid to newwave-grid
    orig_detlam     = (orig_upp - orig_low) 
    resamp_mat      = resamp_detlam*orig_detlam # 

    left_clip = np.where(fin_edges[:-1] - orig_edges[0] < 0, 0, 1)
    right_clip = np.where(orig_edges[-1] - fin_edges[1:] < 0, 0, 1)
    keep_overlapping_matrix = left_clip * right_clip

    resamp_detlam   *= keep_overlapping_matrix[:, np.newaxis]
    resamp_mat      *= keep_overlapping_matrix[:, np.newaxis]
    
    tot_flux    = np.sum(flux * resamp_detlam, axis=-1)
    tot_wv      = np.sum(resamp_detlam, axis=-1)
    id          = tot_wv > 0
    newflux     = np.full(len(tot_wv), np.nan)
    newflux[id] = tot_flux[id]/tot_wv[id]
    
    var        = err**2
    tot_err    = np.sum( var * resamp_mat, axis=-1)**0.5
    newerr     =  np.full(len(tot_wv), np.nan)
    newerr[id] = tot_err[id]/tot_wv[id]
    
    if extrap_treatment == 'zero':
        idout = (fin_low < orig_low[0]) | (fin_upp > orig_upp[-1])
        newflux[idout] = 0.0
        newerr[idout]  = 0.0
    return newflux, newerr

def re_norm(wv_fd, fl_fd, wv_nonfd, fl_nonfd, var_nonfd): #fd=fiducial
    wv_fd_min = np.min(wv_fd)
    wv_fd_max = np.max(wv_fd)
    
    wv_nonfd_min = np.min(wv_nonfd)
    wv_nonfd_max = np.max(wv_nonfd)
    
    if wv_fd_min > wv_nonfd_min:
        id_fd    = np.where( wv_fd < wv_nonfd_max)
        id_nonfd = np.where( wv_nonfd > wv_fd_min)
        norm     = np.abs(np.median(fl_fd[id_fd])/np.median(fl_nonfd[id_nonfd]))
        
    if wv_fd_min < wv_nonfd_min:
        id_fd    = np.where( wv_fd > wv_nonfd_min)
        id_nonfd = np.where( wv_nonfd < wv_fd_max)
        norm     = np.abs(np.median(fl_fd[id_fd])/np.median(fl_nonfd[id_nonfd]))

    fl_nonfd = fl_nonfd*norm
    var_nonfd= var_nonfd*norm**2
    return fl_nonfd, var_nonfd

def resample_spec(new_wave, old_wave, old_flux, old_var):
    old_strline     = np.zeros(len(old_wave))
    id              = np.where(old_var < 0)
    old_var[id]     = -old_var[id] 
    old_strline[id] = 1.0
        
    old_err               = old_var**0.5
    new_fl, new_err       = resample_spec_fluxconv(new_wave, old_wave, old_flux, old_err, extrap_treatment='zero')
    new_var               = new_err**2
    new_strline, ignore   = resample_spec_fluxconv(new_wave, old_wave, old_strline, old_strline, extrap_treatment='zero')
    
    return new_fl, new_var, new_strline
