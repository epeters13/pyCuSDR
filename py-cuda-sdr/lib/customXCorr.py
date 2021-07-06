# Copyright: (c) 2021, Edwin G. W. Peters

import numpy as np

def customXCorr(a,b,N=None):
    """
    Faster xcorr function than numpy's. Done by using FFT's instead of conventional convolution
    """

    Na = len(a)
    Nb = len(b)
    if N is None:
        N = np.max([Na,Nb])

    A = np.fft.fft(a,N)
    B = np.fft.fft(b,N)

    return np.fft.ifft(A*np.conj(B),N)


def customXCorrFast(a,b):
    """
    Ensures zero padding to a prime of the longest of a and b before calling customXCorr
    """

    La, Lb = len(a), len(b)

    Nfft = int(2**np.ceil(np.log2(max((La,Lb)))))

    return customXCorr(a,b,Nfft)
