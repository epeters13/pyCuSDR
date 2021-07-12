# Copyright: (c) 2021, Edwin G. W. Peters
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np

def rrcosfilter(beta,span,spsym):
    """
    Python version of matlabs rcosdesign function
    Contains rcosfilter and rrcosfilter

    B = rcosfilter(BETA, SPAN, SPS) returns raised cosine FIR
    filter coefficients, B, with a rolloff factor of BETA. The filter is
    truncated to SPAN symbols and each symbol is represented by SPS
    samples. The filter energy is one.
    B = rrcosfilter(BETA, SPAN, SPS) returns root raised cosine FIR filter 
    coefficients

    Author: Edwin Peters
    Date: 2018-01-10
    """
    
    delay = span * spsym/2
    t = np.arange(-delay,delay+1)/spsym

    # Find mid-point

    idx1 = np.where(t == 0)

    b = np.zeros(len(t))
    
    if idx1[0].size > 0:
        b[idx1] = -1 / (np.pi*spsym) * (np.pi*(beta-1) - 4*beta)

    # Find non-zero denominator indices
    idx2 = np.where(np.abs(np.abs(4*beta*t) - 1) < np.sqrt(np.finfo(float).eps))
    if idx2[0].size > 0:
        b[idx2] = 1/(2*np.pi*spsym) * (
            np.pi * (beta+1) * np.sin(np.pi*(beta+1)/(4*beta))
            - 4*beta * np.sin(np.pi*(beta-1)/(4*beta))
            + np.pi*(beta-1) * np.cos(np.pi*(beta-1)/(4*beta))
        )

    # Fill in the zero denominator indices
    ind = np.arange(0,len(t))
    idx12 = np.concatenate((idx1[0],idx2[0]))
    ind = np.delete(ind,idx12)
    nind = t[ind]

    b[ind] = -4*beta/spsym * (np.cos((1+beta)*np.pi*nind) +
                              np.sin((1-beta)*np.pi*nind) / (4*beta*nind)) / (
                                  np.pi * ((4*beta*nind)**2-1))
    
        
    # Normalize filter energy
    return b/np.sqrt(np.sum(b**2))
    

def gaussianFilter(gain,BT,spSym,nTaps):
    """
    Returns Gaussian filter taps.
    Inputs:
      gain:   normalized filter gain
      BT:     bitrate to bandwidth ratio 
      spSym:  samples per symbol
      nTaps:  number of filter taps
    
    Output:
      filter taps

    Example:
      taps = gaussianFilter(1,0.5,8,32)
      returns 32 filter taps for a filter with 8 samples per symbol and 0.5 bitrate to bandwidth ratio, ang gain 1

    Author: Edwin Peters
    Date: 2018-04-08
    """

    a = np.sqrt(np.log(2)/2)/BT
    t = np.linspace(-.5*nTaps,.5*nTaps-1,nTaps)/spSym

    ft = np.sqrt(np.pi)/a *np.exp(-(np.pi**2*(t)**2)/a**2)
    ft /= np.sum(ft) * gain # normalize filter

    return ft
