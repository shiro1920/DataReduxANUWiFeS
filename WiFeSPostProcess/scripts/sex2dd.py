import numpy as np
def sex2dd(radec_str):
    """
    Converts sexagesimal coordinates strings like 'hh:mm:ss.ss +-dd:mm:ss.s'
    into a decimal degrees tuples (rr.rr, dd.dd).
    """
    ra_hms, dec_dms = radec_str.split()
    # fix ra
    ra_hh, ra_mm, ra_ss = ra_hms.split(':')
    ra_dd = 15.0*(float(ra_hh)
                  +float(ra_mm)/60.0
                  +float(ra_ss)/3600.0)
    # fix dec
    if dec_dms[0] == '-':
        sign = -1.0
    else:
        sign = 1.0
    dec_hh, dec_mm, dec_ss = dec_dms[1:].split(':')
    dec_dd = sign*(abs(float(dec_hh))
                   +float(dec_mm)/60.0
                   +float(dec_ss)/3600.0)
    return (ra_dd, dec_dd)