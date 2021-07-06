# Copyright: (c) 2021, Edwin G. W. Peters

import numpy as np
from lib.filters import gaussianFilter

"""
GMSK modulation of the data
"""

def gmskMod(bits,spSym,bw=0.5,nTaps=None,gain=1):
    """
    [waveform, phase] = gmskMod(bits,spSym,bw,nTaps,gain) 
    
    Does a GMSK modulation of the input bits
    
    input:
    \tbits -- raw bits to be coded
    \tspSym -- oversampling factor
    \tbw -- filter bandwidth [default 0.5 symbols]
    \tTaps -- number of filter taps [default 4 * spSym]
    \tgain -- Gaussian filter gain [default 1]

    returns:
    \tmodulated waveform
    \tphase of modulated waveform
    """

    if not min(bits) < 0:
        bits = bits*2-1
    # if any(bits>1) or any(bits<0):
    #     raise ValueError('bits expected to contain only 0\'s and 1\'s')
    
    if nTaps is None:
        nTaps = 4 * spSym
        
    filt = gaussianFilter(gain,bw,spSym,nTaps)*np.pi/2/spSym

    # interpolate and pulse shape
    filtBits = np.convolve(filt,np.repeat(bits,spSym))

    pulseShape = np.cumsum(filtBits)

    return np.exp(1j*pulseShape), pulseShape, len(filt)
