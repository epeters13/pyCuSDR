# Original Author    : Edwin G. W. Peters @ sdr-Surface-Book-2
#   Creation date    : Mon Jul 12 15:36:46 2021 (+1000)
#   Email            : edwin.peters@unsw.edu.au
# ------------------------------------------------------------------------------
# Last-Updated       : Mon Jul 12 15:50:28 2021 (+1000)
#           By       : Edwin G. W. Peters @ sdr-Surface-Book-2
# ------------------------------------------------------------------------------
# File Name          : freq_from_rangerate.py
# Description        : 
# ------------------------------------------------------------------------------
# Copyright          : Insert license
# ------------------------------------------------------------------------------

import scipy.constants

def rangerate_from_freq(freq,Fc):
    """
    Gpredict seems to not be able to configure the LO as I want. Instead of wasting time, just assume their default LO to recover the range rate, and we can do what we need to
    """
    # print(f'freq {freq} Fc {Fc}')
    dopp_at_IF = freq - Fc
    return dopp_at_IF*scipy.constants.speed_of_light/Fc

def freq_from_rangerate(rangerate,Fc):
    # print(f'rangerate {rangerate} Fc {Fc}')
    return Fc + rangerate/scipy.constants.speed_of_light*Fc
